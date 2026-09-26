"""Evaluation-only of the shared public SSG epoch50 checkpoint. Never train."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.data import GTSRBTrainOnly
from trustml.convnext_gtsrb.evaluation import autoattack_inputs, evaluate
from trustml.convnext_gtsrb.model import sha256
from .private import load_selected
from .rectangular import SOURCE_SHA

STAGES = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
DEV_HASH = 'da0b5fa6364952d92f3d6e9356eaf4f23dcfe1a6b06871e1aab07be8bce28a24'


def checked_mask(value, previous):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != previous.shape:
        raise ValueError('invalid attack mask dtype/shape')
    if np.any(value & ~previous):
        raise ValueError('attack stage resurrects failed examples')
    return value.copy()


def full_aa(model, data, output, batch_size=64, *, checkpoint_sha256=SOURCE_SHA,
            expected_clean=None):
    from autoattack import AutoAttack
    from autoattack.state import EvaluationState
    output.mkdir(parents=True, exist_ok=False)
    ids = data.split['indices']['dev']
    x, y = autoattack_inputs(*data.batch(ids), 'cuda')
    model.eval()
    if x.dtype != torch.float32 or any(m.training for m in model.modules()):
        raise RuntimeError('FP32/eval required')
    clean = []
    for i in range(0, len(ids), batch_size):
        with torch.no_grad():
            logits = model(x[i:i+batch_size])
        if logits.shape[1] != 43 or not torch.isfinite(logits).all():
            raise RuntimeError('invalid clean logits')
        clean.extend(logits.argmax(1).eq(y[i:i+batch_size]).cpu().tolist())
    clean = np.asarray(clean, dtype=bool)
    if len(ids) != 4020 or data.split['hashes']['dev'] != DEV_HASH:
        raise RuntimeError('ordered public dev differs from locked reference')
    if expected_clean is None:
        if checkpoint_sha256 != SOURCE_SHA or int(clean.sum()) != 2169:
            raise RuntimeError('shared pretrain clean count differs from verified reference')
    elif not np.array_equal(clean, np.asarray(expected_clean, dtype=bool)):
        raise RuntimeError('clean predictions differ from verified final checkpoint')
    arrays = dict(sample_ids=np.asarray(ids), clean_correct=clean)
    stage_results = []
    last = [clean, time.monotonic()]

    class CapturedState(EvaluationState):
        def add_run_attack(self, attack):
            super().add_run_attack(attack)
            mask = checked_mask(self.robust_flags.detach().cpu().numpy(), last[0])
            arrays['after_' + attack] = mask
            now = time.monotonic()
            stage_results.append(dict(attack=attack, survivors=int(mask.sum()),
                accuracy=float(100*mask.mean()), seconds=now-last[1]))
            last[:] = [mask, now]
            np.savez_compressed(output/'stage_masks.npz', **arrays)
            write_json(output/'stage_metrics.json', stage_results)
            self.to_disk(force=True)
            print(json.dumps(dict(phase='aa_stage_complete', **stage_results[-1])), flush=True)

    started = time.monotonic()
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard',
        seed=42, device='cuda', log_path=str(output/'autoattack.log'))
    if adversary.attacks_to_run != STAGES:
        raise RuntimeError('unexpected pinned standard attack suite')
    # Observe stage masks; do not alter attack parameters, RNG, or survivor selection.
    with patch('autoattack.autoattack.EvaluationState', CapturedState):
        adv = adversary.run_standard_evaluation(x, y, bs=batch_size,
                                              state_path=output/'state.json')
    max_linf = float((adv-x).abs().max())
    if not torch.isfinite(adv).all() or max_linf > 8/255+1e-6 or adv.min()<0 or adv.max()>1:
        raise RuntimeError('invalid adversarial images')
    posthoc = []
    for i in range(0, len(ids), batch_size):
        with torch.no_grad():
            logits = model(adv[i:i+batch_size])
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite posthoc logits')
        posthoc.extend(logits.argmax(1).eq(y[i:i+batch_size]).cpu().tolist())
    state = EvaluationState.from_disk(output/'state.json')
    attack_time = checked_mask(state.robust_flags.numpy(), clean)
    # Stages with zero survivors may be skipped by stock AutoAttack.
    if set(STAGES)-set(state.run_attacks) and attack_time.any():
        raise RuntimeError('incomplete AutoAttack with surviving examples')
    posthoc = np.asarray(posthoc, dtype=bool) & clean
    final = attack_time & posthoc
    arrays.update(aa_attack_time_correct=attack_time, aa_posthoc_correct=posthoc,
                  aa_correct=final)
    np.savez_compressed(output/'predictions.npz', **arrays)
    mismatch = int(np.count_nonzero(attack_time != posthoc))
    result = dict(status='PASS' if not mismatch else 'NUMERICAL_WARNING', examples=len(ids),
        clean=float(100*clean.mean()), aa=float(100*final.mean()), stages=stage_results,
        attack_time_posthoc_mismatches=mismatch, max_linf=max_linf,
        seconds=time.monotonic()-started, full_autoattack=True,
        attack_seed=42, epsilon=8/255, norm='Linf', precision='FP32', model_mode='eval',
        autoattack_version='0.1 / a39220048b3c9f2cca9a4d3a54604793c68eca7e',
        ordered_dev_hash=DEV_HASH, checkpoint_sha256=checkpoint_sha256,
        optimizer_steps_added=0, official_test_loaded=False)
    write_json(output/'metrics.json', result)
    return result


def run(args):
    configure_gpu()
    torch.manual_seed(42)
    np.random.seed(42)
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    args.output.mkdir(parents=True, exist_ok=False)
    model, architecture = load_selected(Path('gtsrb_inputs'), args.checkpoint, SOURCE_SHA)
    model.cuda().eval()
    protocol = json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())
    data = GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),
                         Path('gtsrb_inputs/splits.json'), protocol['augmentation'])
    if data.split['hashes']['dev'] != DEV_HASH:
        raise RuntimeError('locked real-dev split mismatch')
    write_json(args.output/'architecture.json', architecture)
    # Exact existing 1024-record diagnostic, separate from the full-dev AA.
    diagnostic = evaluate(model, data, args.output/'diagnostic_1024', final=True)
    result = full_aa(model, data, args.output/'aa_4020')
    write_json(args.output/'summary.json', dict(diagnostic_1024=diagnostic, aa_4020=result))
    write_json(args.output/'files.json', {str(p.relative_to(args.output)):sha256(p)
        for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/pretrain_real_dev_aa'))
    run(p.parse_args())

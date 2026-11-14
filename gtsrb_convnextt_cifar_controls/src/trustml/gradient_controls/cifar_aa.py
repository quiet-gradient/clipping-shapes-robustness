"""CIFAR AA8 adapter: frozen wrapper, explicit stage smoke, saved per-record masks."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

from .io import canonical, sha, write_json

COMPONENTS = ('apgd-ce', 'apgd-t', 'fab-t', 'square')


def load_evaluation_model(cfg, checkpoint, device):
    import torch
    from trustml.cifar_interventions.runtime import model_for
    # model_for fixes GN irrespective of whether the DP mechanism is enabled.
    model = model_for(cfg, device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    if payload.get('config',{}).get('experiment',{}).get('name') != str(cfg.experiment.name):
        raise RuntimeError('checkpoint/config run mismatch')
    return model.to(device).float().eval()


def aa_configuration(split, count, log_path):
    if split not in ('test','dev') or count != (10000 if split=='test' else 5000):
        raise ValueError('full frozen split required')
    return dict(epsilon=8/255,norm='Linf',version='standard',batch_size=128,seed=42,
                expected_examples=count,require_full_test=(split=='test'),verbose=False,
                capture_component_trace=True,log_path=str(log_path))


def validate_masks(ids, masks, expected_ids):
    import numpy as np
    if ids.dtype.kind not in 'iu' or not np.array_equal(ids, expected_ids):
        raise RuntimeError('sample IDs/order mismatch')
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError('duplicate sample IDs')
    previous = np.ones(len(ids), dtype=bool)
    for name in ('clean', *COMPONENTS, 'final'):
        mask = masks[name]
        if mask.shape != previous.shape or mask.dtype != np.bool_ or np.any(mask & ~previous):
            raise RuntimeError('invalid/nonmonotone per-example masks: ' + name)
        previous = mask
    if not np.array_equal(masks['square'], masks['final']):
        raise RuntimeError('final/Square mask mismatch')


def smoke_all_components(model, public_loader, cfg, device, output):
    """Exercise each standard stage on the SAME clean-correct public examples.

    This separate technical smoke avoids falsely passing when an earlier stage
    kills all survivors. It is not an estimate of ensemble robustness.
    """
    import torch
    from autoattack import AutoAttack
    from omegaconf import OmegaConf
    from trustml.attacks.autoattack import AutoAttackAttack, _NormalizedModel
    wrapper = AutoAttackAttack(OmegaConf.create(dict(epsilon=8/255, version='standard',
        norm='Linf', seed=42, batch_size=128, verbose=False)), mean=cfg.data.mean, std=cfg.data.std)
    prepared = wrapper._prepare_inputs(public_loader)
    wrapped = _NormalizedModel(model, wrapper.mean, wrapper.std).to(device).eval()
    diagnostics = wrapper._diagnose_prepared(model, wrapped, prepared, device)
    if not diagnostics['validation_passed']:
        raise RuntimeError('AA smoke normalization failed')
    predictions = wrapper._predict(wrapped, prepared.pixel, device, 128)
    indices = predictions.eq(prepared.labels).nonzero().flatten()[:8]
    if len(indices) < 2:
        raise RuntimeError('not enough clean-correct public examples to exercise AA')
    x, y = prepared.pixel[indices], prepared.labels[indices]
    rows = []
    for component in COMPONENTS:
        adversary = AutoAttack(wrapped, norm='Linf', eps=8/255, seed=42,
                              verbose=False, version='standard', device=device)
        adversary.attacks_to_run = [component]
        owner = {'apgd-ce': adversary.apgd, 'apgd-t': adversary.apgd_targeted,
                 'fab-t': adversary.fab, 'square': adversary.square}[component]
        original = owner.perturb
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        owner.perturb = counted
        started = time.monotonic()
        adv, labels = adversary.run_standard_evaluation(x, y, bs=128, return_labels=True)
        delta = (adv.cpu()-x).abs().max().item()
        if not calls or not torch.isfinite(adv).all() or delta > 8/255+1e-6:
            raise RuntimeError('AA smoke stage failed: '+component)
        rows.append(dict(component=component, actual_perturb_calls=len(calls),
                         sample_count=len(indices), max_linf=delta,
                         seconds=time.monotonic()-started))
    write_json(output, dict(status='PASS', technical_smoke_only=True,
                            components=rows, public_sample_positions=indices.tolist()))


def evaluate(config, checkpoint, output, split='test', smoke_only=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from omegaconf import OmegaConf
    from torchvision.datasets import CIFAR10
    from trustml.cifar_interventions.runtime import datasets
    from trustml.cifar_interventions.protocol import DEV_SHA, digest
    from trustml.data.transforms import cifar_transform
    from trustml.attacks.autoattack import AutoAttackAttack
    from trustml.blind_official_test import clean_mask

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CIFAR AA requires exactly one CUDA GPU')
    device = torch.device('cuda')
    config, checkpoint, output = Path(config), Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(config)
    model = load_evaluation_model(cfg, checkpoint, device)
    if any(p.dtype != torch.float32 for p in model.parameters()) or model.training:
        raise RuntimeError('FP32/eval invariant failed')
    _, dev, _, dev_ids = datasets(cfg)
    if digest(dev_ids) != DEV_SHA:
        raise RuntimeError('historical dev ordering changed')
    smoke_loader = DataLoader(Subset(dev, dev_ids[:128]), batch_size=128, shuffle=False, num_workers=0)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    smoke_all_components(model, smoke_loader, cfg, device, output/'attack_smoke.json')
    if smoke_only:
        return
    if split == 'test':
        dataset = CIFAR10(str(cfg.data.root), train=False, download=False,
            transform=cifar_transform(train=False, mean=tuple(cfg.data.mean),
                                      std=tuple(cfg.data.std), augmentation=False))
        ids = np.arange(10000, dtype=np.int64)
    elif split == 'dev':
        dataset, ids = Subset(dev, dev_ids), np.asarray(dev_ids, dtype=np.int64)
    else:
        raise ValueError('split must be test or dev')
    if len(dataset) != len(ids) or len(ids) != (10000 if split == 'test' else 5000):
        raise RuntimeError('evaluation coverage mismatch')
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    direct_clean = np.asarray(clean_mask(model, loader, device), dtype=bool)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    aa = AutoAttackAttack(OmegaConf.create(aa_configuration(split,len(ids),output/'autoattack.log')),
         mean=cfg.data.mean, std=cfg.data.std)
    metrics = aa.run(model, loader, device=device, progress=False)
    trace = aa.last_component_trace
    if trace is None or trace['official_component_order'] != list(COMPONENTS):
        raise RuntimeError('missing/wrong AutoAttack stage trace')
    masks = {'clean': np.asarray(trace['clean_correct'], dtype=bool),
             'final': np.asarray(trace['final_robust_correct'], dtype=bool)}
    masks.update({name: np.asarray(trace['correct_after_component'][name], dtype=bool) for name in COMPONENTS})
    validate_masks(ids, masks, ids)
    survivors = int(masks['clean'].sum())
    for component in COMPONENTS:
        if int(trace['component_reached_examples'][component]) != survivors:
            raise RuntimeError('AutoAttack stage reach mismatch')
        survivors = int(masks[component].sum())
    if not np.array_equal(direct_clean, masks['clean']):
        raise RuntimeError('direct/wrapper clean masks disagree')
    diagnostics = aa.last_output_diagnostics or {}
    if diagnostics.get('attack_time_vs_posthoc_correctness_disagreement_count') != 0:
        raise RuntimeError('attack-time/posthoc correctness mismatch requires review')
    if float(metrics['attack/autoattack_max_perturbation']) > 8/255+1e-6:
        raise RuntimeError('AA Linf bound exceeded')
    np.savez_compressed(output/'per_example.npz', sample_ids=ids, **masks)
    with np.load(output/'per_example.npz', allow_pickle=False) as restored:
        validate_masks(restored['sample_ids'], {k: restored[k] for k in masks}, ids)
    identity = dict(checkpoint_sha256=sha(checkpoint), config_sha256=sha(config), split=split,
                    ordered_sample_ids_sha256=digest(ids.tolist()), epsilon=8/255,
                    norm='Linf', version='standard', attack_seed=42, precision='FP32',
                    model_mode='eval', sample_count=len(ids), batch_size=128,
                    autoattack_wheel_sha256=sha(Path('vendor/autoattack-0.1-py3-none-any.whl')),
                    torch_version=torch.__version__,numpy_version=np.__version__,
                    wrapper_sha256=sha(Path(__import__('trustml.attacks.autoattack', fromlist=['x']).__file__)))
    write_json(output/'evaluation.json', dict(status='PASS', **identity,
        evaluation_fingerprint=hashlib.sha256(canonical(identity)).hexdigest(),
        clean_accuracy=float(masks['clean'].mean()), aa_accuracy=float(masks['final'].mean()),
        component_runtimes_seconds=trace['component_durations_seconds'],
        component_reached_examples=trace['component_reached_examples'],
        survivor_counts={k:int(v.sum()) for k,v in masks.items()},
        normalization=aa.last_diagnostics, output_diagnostics=diagnostics,
        metrics=metrics, per_example_sha256=sha(output/'per_example.npz')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('test','dev'), default='test')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    from trustml.convnext_cifar100.preflight import configure_gpu
    configure_gpu()
    evaluate(args.config, args.checkpoint, args.output, args.split, args.smoke_only)

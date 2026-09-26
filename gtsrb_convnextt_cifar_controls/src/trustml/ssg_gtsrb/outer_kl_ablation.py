"""One matched hard J/U x outer-KL quartet, public diagnostics and full dev AA.

This is a sensitivity experiment, not a replacement grid or a new selection.
All four runs share one RAM-only randomness root. Privacy is marginal per run;
the correlated quartet is NOT jointly epsilon-4 private.
"""
import argparse
import gc
import json
from pathlib import Path
import secrets
import shutil
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.common import code_identity, diagnostic_rng, tensor_digest
from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
from trustml.convnext_gtsrb.mechanism import gradients, contribution, private_release, assign_gradient
from trustml.convnext_gtsrb.model import sha256
from .campaign import check_coupling, verify_completed_run
from .night_aa import load_final
from .persistence import TransactionStore
from .private import load_selected, train, training_microbatch
from .real_dev_aa import DEV_HASH, full_aa
from .rectangular import SOURCE_SHA, fetch
from .rectangular_v2 import protocol


def manifest():
    rows = []
    # Complete the existing-convention control pair first, then the intervention.
    for convention in ('full', 'detached_clean'):
        for role in ('Joint', 'Componentwise'):
            rid = f'kl2x2_{role.lower()}_{convention}_c100_lr0050_s42'
            row = dict(run_id=rid, role=role, outer_kl_target=convention,
                       clipping_map='hard', C=1., lr=.05, beta=1.,
                       q=1/3 if role == 'Componentwise' else None,
                       optimizer='SGD', private=True, seed_label=42,
                       source_checkpoint_sha256=SOURCE_SHA,
                       rng_coupling='shared_quartet_ram_only')
            if role == 'Componentwise':
                row['matched_joint_run_id'] = f'kl2x2_joint_{convention}_c100_lr0050_s42'
            rows.append(row)
    return dict(campaign='gtsrb_outer_kl_2x2_v1', runs=rows,
                training_runs=4, full_dev_aa_evaluations=4, seeds=[42],
                optimizer_steps=3436, physical_batch=128, logical_batch=512,
                source_checkpoint_sha256=SOURCE_SHA, source_epoch=50,
                epsilon=4., delta=1e-5, noise_multiplier=1.22344970703125,
                official_test_allowed=False, hyperparameter_selection=False,
                primary_contrast='(AA_J_full-AA_U_full) - (AA_J_detached-AA_U_detached)',
                secondary_contrasts=['detached-minus-full within Joint', 'detached-minus-full within Componentwise'],
                inference='one coupled training realization; paired image/track analysis is not training-seed variance',
                privacy_scope='marginal per fixed run; NOT joint epsilon-4 for the quartet',
                existing_accountant_deviation='RDP minus PRV about .22846 exceeds original .1 crosscheck threshold; retained unchanged, not a new PASS of that threshold')


def transaction_ids():
    return ['public_kl_probe'] + [key for row in manifest()['runs']
                                 for key in (row['run_id'], row['run_id']+'_aa')]


def paired_vector_metrics(left, right):
    """Public-record comparisons; undefined zero-direction cosines remain explicit."""
    left, right = left.double(), right.double()
    nl, nr = left.norm(dim=-1), right.norm(dim=-1)
    denominator = nl * nr
    cosine = torch.where(denominator > 0, (left*right).sum(-1)/denominator,
                         torch.full_like(denominator, float('nan')))
    return dict(full_norm=nl, detached_norm=nr, cosine=cosine.clamp(-1, 1),
                difference_norm=(left-right).norm(dim=-1),
                relative_difference=torch.where(nl > 0, (left-right).norm(dim=-1)/nl,
                                                torch.full_like(nl, float('nan'))))


def public_gpu_gate(model, data, output):
    """Exercise actual physical128, both derivative conventions and J/U releases."""
    params = [p for p in model.parameters() if p.requires_grad]
    original = [p.detach().clone() for p in params]
    initial = tensor_digest(dict(model.named_parameters()))
    ids = data.split['indices']['probe'][:128]
    x, y = data.batch(ids)
    x, y = x.cuda(), y.cuda()
    uniform = public_uniform(x, [data.split['records'][i]['image_id'] for i in ids], 20260921, 'trades')
    adv, _ = attack(model, x, y, objective='kl', uniform=uniform)
    torch.cuda.reset_peak_memory_stats()
    checks = []
    for convention in ('full', 'detached_clean'):
        values = gradients(model, x, y, trades_adv=adv, outer_kl_target=convention)
        for stable in (False, True):
            reference = gradients(model, x[:2], y[:2], trades_adv=adv[:2],
                                  outer_kl_target=convention, stable_public_kl=stable)
            for key in ('g', 'h'):
                torch.testing.assert_close(values[key][:2], reference[key], rtol=5e-4, atol=1e-5)
            del reference
        for role in ('Joint', 'Componentwise'):
            update = contribution(values, role, 1., 1/3 if role == 'Componentwise' else None)
            if not torch.isfinite(update).all() or (update.norm(dim=-1) > 1+2e-5).any():
                raise RuntimeError('outer-KL GPU query bound failure')
            noise = torch.Generator(device='cuda').manual_seed(20260923)
            sanitized = private_release(update.sum(0), sigma=1.22344970703125,
                                        bound=1., generator=noise, denominator=512)
            optimizer = torch.optim.SGD(params, lr=.05, momentum=.9, nesterov=True, weight_decay=5e-4)
            assign_gradient(model, sanitized)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in params):
                raise RuntimeError('outer-KL GPU optimizer failure')
            with torch.no_grad():
                for p, original_p in zip(params, original):
                    p.copy_(original_p)
            checks.append(dict(outer_kl_target=convention, role=role,
                               max_query_norm=float(update.norm(dim=-1).max())))
            del update, sanitized, optimizer
        del values
    model.zero_grad(set_to_none=True)
    if tensor_digest(dict(model.named_parameters())) != initial:
        raise RuntimeError('public GPU gate changed initialization')
    total_memory = torch.cuda.get_device_properties(0).total_memory
    peak_memory = torch.cuda.max_memory_allocated()
    if peak_memory + 2*1024**3 > .9*total_memory:
        raise RuntimeError('physical128 public gate has insufficient GPU memory headroom; no private training started')
    result = dict(status='PASS', public_only=True, physical_batch=128, checks=checks,
                  initial_model_preserved=True, gpu=torch.cuda.get_device_name(),
                  peak_allocated_bytes=peak_memory, device_memory_bytes=total_memory,
                  memory_headroom_gate='peak+2GiB <= 90% device memory')
    write_json(output, result)
    return result


def paired_public_probe(model, data, output):
    """Same theta and SAME x_adv for both gradients; no private examples/no DP noise."""
    output.mkdir(parents=True, exist_ok=False)
    ids = data.split['indices']['probe']
    if len(ids) != 512:
        raise RuntimeError('public probe must have 512 records')
    saved, sums = {}, {}
    began = time.monotonic()
    initial = tensor_digest(dict(model.named_parameters()))
    for begin in range(0, 512, 2):
        block = ids[begin:begin+2]
        x, y = data.batch(block)
        x, y = x.cuda(), y.cuda()
        image_ids = [data.split['records'][i]['image_id'] for i in block]
        uniform = public_uniform(x, image_ids, 20260921, 'trades')
        adv, _ = attack(model, x, y, objective='kl', uniform=uniform)
        full = gradients(model, x, y, trades_adv=adv, stable_public_kl=True, outer_kl_target='full')
        detached = gradients(model, x, y, trades_adv=adv, stable_public_kl=True, outer_kl_target='detached_clean')
        torch.testing.assert_close(full['g'], detached['g'], rtol=0, atol=0)
        with torch.no_grad():
            clean_logits, adversarial_logits = model(x), model(adv)
            kl = F.kl_div(adversarial_logits.log_softmax(-1), clean_logits.softmax(-1), reduction='none').sum(-1)
        saved.setdefault('kl_value', []).extend(kl.cpu().tolist())
        vectors = {'h': (full['h'], detached['h']),
                   's': (full['g']+full['h'], detached['g']+detached['h'])}
        for role in ('Joint', 'Componentwise'):
            vectors[role] = tuple(contribution(v, role, 1., 1/3 if role == 'Componentwise' else None)
                                  for v in (full, detached))
        for label, pair in vectors.items():
            for key, values in paired_vector_metrics(*pair).items():
                saved.setdefault(label+'_'+key, []).extend(values.cpu().tolist())
            for convention, value in zip(('full', 'detached_clean'), pair):
                key = label+'_'+convention
                total = value.double().sum(0)
                sums[key] = total if key not in sums else sums[key]+total
        if begin % 64 == 0:
            print(json.dumps(dict(phase='paired_outer_kl_public_probe', records=begin+2, total=512)), flush=True)
        del full, detached, vectors
    if tensor_digest(dict(model.named_parameters())) != initial:
        raise RuntimeError('paired public probe changed model')
    arrays = {key: np.asarray(value) for key, value in saved.items()}
    if any(len(v) != 512 or np.isinf(v).any() for v in arrays.values()):
        raise RuntimeError('invalid paired probe arrays')
    aggregates = {}
    for key, values in arrays.items():
        finite = values[np.isfinite(values)]
        aggregates[key] = dict(undefined=int((~np.isfinite(values)).sum()),
                               mean=float(finite.mean()) if len(finite) else None,
                               percentiles=np.percentile(finite, [5, 25, 50, 75, 95]).tolist() if len(finite) else None)
    means = {}
    for label in ('h', 's', 'Joint', 'Componentwise'):
        values = paired_vector_metrics(*(sums[label+'_'+c].unsqueeze(0)/512 for c in ('full', 'detached_clean')))
        means[label] = {k: float(v[0]) if torch.isfinite(v[0]) else None for k, v in values.items()}
    np.savez_compressed(output/'public_comparison.npz', probe_ids=np.asarray(ids), **arrays)
    result = dict(status='PASS', examples=512, public_only=True, optimizer_steps=0,
                  no_dp_noise=True, same_adversarial_inputs=True, initial_model_preserved=True,
                  source_checkpoint_sha256=SOURCE_SHA, split_hash=data.split['split_hash'],
                  probe_hash=data.split['hashes']['probe'], outer_kl_targets=['full', 'detached_clean'],
                  per_record=aggregates, mean_update=means, seconds=time.monotonic()-began,
                  official_test_loaded=False)
    write_json(output/'comparison.json', result)
    return result


def require_empty_receipt(store, route):
    with store.request('GET', route['receipt_get']) as response:
        if response.status_code != 404:
            raise RuntimeError('existing/uncertain receipt: recover committed children, never retrain blindly')
    with store.request('GET', route['archive_get'], stream=True) as response:
        if response.status_code != 404:
            raise RuntimeError('existing/uncertain archive: reconcile persistence before any retraining')


def run(args):
    code_identity()
    configure_gpu()
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('exactly one A100 required')
    if torch.cuda.get_device_properties(0).total_memory < 70*1024**3:
        raise RuntimeError('validated physical128 requires A10080GB; refusing an unvalidated smaller GPU')
    plan = manifest()
    payload = json.loads(args.routes.read_text())
    if payload['manifest_hash'] != digest(plan) or set(payload['routes']) != set(transaction_ids()):
        raise RuntimeError('ablation manifest/routes mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    store = TransactionStore(payload['routes'], args.output/'transactions')
    for route in payload['routes'].values():
        require_empty_receipt(store, route)
    print(json.dumps(dict(phase='outer_kl_ablation_start', gpu=torch.cuda.get_device_name(),
                          training_runs=4, full_dev_aa_evaluations=4, official_test_loaded=False)), flush=True)
    from scripts.platform.gtsrb_inputs import materialize
    inputs = payload['inputs']
    materialize(args.input_archive, inputs['training_sha256'])
    archive = Path('public_selected.tar.gz')
    fetch(inputs['selected_url'], archive, inputs['selected_archive_sha256'])
    selected = Path('gtsrb_inputs/public_pretrain_selected.pt')
    with tarfile.open(archive) as tf:
        member = tf.getmember('public_pretrain_selected.pt')
        if not member.isfile():
            raise RuntimeError('invalid public pretrain archive')
        with tf.extractfile(member) as source, selected.open('xb') as destination:
            shutil.copyfileobj(source, destination)
    if sha256(selected) != SOURCE_SHA:
        raise RuntimeError('public pretrain mismatch')
    augmentation = json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data = GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),
                         Path('gtsrb_inputs/splits.json'), augmentation)
    locked = json.loads(Path('reports/ssg_gtsrb_v1/gtsrb_private_split_manifest.json').read_text())
    if data.split['hashes']['dev'] != DEV_HASH or data.split['split_hash'] != locked['split_hash']:
        raise RuntimeError('public/private split changed')
    p = protocol()
    p['campaign'] = plan['campaign']
    p['outer_kl_ablation'] = plan
    p['stage_order'] = ['public_gpu_gate', 'paired_public_probe',
                        'four_sequential_train_commit_AA_commit_children']
    p['grid_per_optimizer'] = dict(Joint=2, Componentwise=2, C=[1.], lr=[.05], q=[1/3],
                                  outer_kl_target=['full', 'detached_clean'])
    p['sgd']['learning_rates'] = [.05]
    p.pop('nonprivate', None)
    p['private']['full_autoattack_stage'] = 'IN_SAME_PARENT_AFTER_EACH_COMMITTED_TRAINING'
    p['execution'].update(backend_assignment='datasphere_only', gpu_count=1,
                          one_C_LR_group_per_job=True, artificial_walltime_hours=None)
    calibration = json.loads(Path('configs/ssg_gtsrb_rect54/calibration.json').read_text())
    privacy = json.loads(Path('configs/ssg_gtsrb_rect54/accounting.json').read_text())
    if privacy['noise_multiplier'] != plan['noise_multiplier'] or privacy['steps'] != 3436:
        raise RuntimeError('privacy configuration changed')
    if any(training_microbatch(p, row) != 128 for row in plan['runs']):
        raise RuntimeError('physical batch changed')
    write_json(args.output/'manifest.json', plan)
    write_json(args.output/'protocol.json', p)
    model, _ = load_selected(Path('gtsrb_inputs'), selected, SOURCE_SHA)
    model.cuda().eval()
    with diagnostic_rng():
        public_gpu_gate(model, data, args.output/'public_gpu_gate.json')
        paired_public_probe(model, data, args.output/'public_kl_probe')
    shutil.copy2(args.output/'public_gpu_gate.json', args.output/'public_kl_probe/public_gpu_gate.json')
    store.commit('public_kl_probe', args.output/'public_kl_probe')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    master = secrets.token_bytes(32)  # never persisted, printed or exported
    completed, attacks = [], []
    for row in plan['runs']:
        rid = row['run_id']
        target = args.output/rid
        result = train(row, p, data, calibration, privacy, Path('gtsrb_inputs'), selected, master, target)
        verify_completed_run(target, row, result)
        if completed:
            check_coupling(completed[0], result)
        store.commit(rid, target)  # weights survive any later AA/parent failure
        completed.append(result)
        write_json(args.output/'completed_trainings.json', completed)
        gc.collect()
        torch.cuda.empty_cache()
        model, architecture = load_final(target/'checkpoints/final.pt',
                                        dict(checkpoint_sha256=result['checkpoint_sha256'], config=row))
        model.cuda().eval()
        aa_target = args.output/(rid+'_aa')
        aa_target.mkdir()
        with np.load(target/'evaluation_epoch050/public_dev_predictions.npz') as predictions:
            if not np.array_equal(predictions['dev_ids'], data.split['indices']['dev']):
                raise RuntimeError('training dev sample IDs changed')
            expected_clean = predictions['clean_correct'].copy()
        write_json(aa_target/'provenance.json', dict(config=row, config_hash=digest(row),
                   checkpoint_sha256=result['checkpoint_sha256'], architecture=architecture,
                   split_hash=data.split['split_hash'], manifest_hash=digest(plan)))
        with diagnostic_rng():
            torch.manual_seed(42)
            np.random.seed(42)
            metrics = full_aa(model, data, aa_target/'aa_4020',
                              checkpoint_sha256=result['checkpoint_sha256'], expected_clean=expected_clean)
        store.commit(rid+'_aa', aa_target)
        attacks.append(dict(run_id=rid, **metrics))
        write_json(args.output/'completed_autoattack.json', attacks)
        if metrics['status'] != 'PASS':
            raise RuntimeError('AA numerical warning preserved; review before continuation')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output/'summary.json', dict(status='COMPLETE', runs=completed, autoattack=attacks,
               shared_randomness_verified=True, official_test_loaded=False,
               limitation='one paired training realization; no training-seed variance estimate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/outer_kl_ablation'))
    run(parser.parse_args())

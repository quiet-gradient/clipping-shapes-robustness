"""Frozen-weight, public-only full/stop-gradient KL diagnostic. NO training or AA."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import requests
import torch

from scripts.platform.gtsrb_inputs import materialize
from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.common import tensor_digest
from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
from trustml.convnext_gtsrb.mechanism import gradients
from trustml.convnext_gtsrb.model import sha256
from .night_aa import load_final, extract_checkpoint
from .outer_kl_ablation import paired_vector_metrics
from .persistence import TransactionStore
from .private import load_selected
from .real_dev_aa import DEV_HASH
from .rectangular import SOURCE_SHA

CHECKPOINTS = ['pretrain', 'joint', 'componentwise']
CHUNK = 64
SAMPLES = 512
SOFT_SECONDS = 1440


def transaction_ids():
    return [f'{name}_{start:03d}' for start in range(0, SAMPLES, CHUNK)
            for name in CHECKPOINTS] + ['summary']


class BoundedStore(TransactionStore):
    @staticmethod
    def request(method, url, **kwargs):
        try:
            return requests.request(method, url, timeout=(10, 30), **kwargs)
        except requests.RequestException:
            raise RuntimeError('artifact transport failed; credentials suppressed') from None


def clip(v, bound):
    return v * (bound / v.norm(dim=-1, keepdim=True).clamp_min(1e-300)).clamp(max=1)


def compare(g, hf, hd):
    """Same public records/weights/attack; C=1, beta=1, q=1/3 in both cases."""
    g, hf, hd = g.double(), hf.double(), hd.double()
    vectors = {}
    arrays = {}
    for suffix, h in [('full', hf), ('detached', hd)]:
        s = g + h
        j = clip(s, 1.)
        u = clip(g, 2/3) + clip(h, 1/3)
        sj = s / (s.norm(dim=-1, keepdim=True) + .1)
        su = (2/3)*g/(g.norm(dim=-1, keepdim=True)+(2/3)*.1)
        su = su + (1/3)*h/(h.norm(dim=-1, keepdim=True)+(1/3)*.1)
        ng, nh, ns = [v.norm(dim=-1) for v in (g, h, s)]
        arrays[f'{suffix}_cos_g_h'] = paired_vector_metrics(g, h)['cosine']
        arrays[f'{suffix}_kappa'] = ns / (ng + nh).clamp_min(1e-300)
        ag = ((2/3)/ng.clamp_min(1e-300)).clamp(max=1)
        ah = ((1/3)/nh.clamp_min(1e-300)).clamp(max=1)
        arrays[f'{suffix}_beta_eff_ratio'] = ah / ag
        arrays[f'{suffix}_JU_cosine'] = paired_vector_metrics(j, u)['cosine']
        arrays[f'{suffix}_JU_distance'] = (j-u).norm(dim=-1)
        ej, eu = (s-j).square().sum(-1), (s-u).square().sum(-1)
        residual = eu - ej - (j-u).square().sum(-1)
        arrays[f'{suffix}_projection_residual'] = residual
        if (residual < -1e-8*(1+eu+ej)).any():
            raise RuntimeError('hard projection inequality failed')
        for tag, v in [('h', h), ('s', s), ('J', j), ('U', u), ('SJ', sj), ('SU', su)]:
            if not torch.isfinite(v).all():
                raise RuntimeError('nonfinite diagnostic vector')
            if tag in {'J', 'U', 'SJ', 'SU'} and (v.norm(dim=-1) > 1+1e-8).any():
                raise RuntimeError('query bound failed')
            vectors[f'{tag}_{suffix}'] = v
    for tag in ['h', 's', 'J', 'U', 'SJ', 'SU']:
        for key, value in paired_vector_metrics(vectors[tag+'_full'], vectors[tag+'_detached']).items():
            arrays[f'{tag}_{key}'] = value
    return {k: v.cpu().numpy() for k, v in arrays.items()}, {k: v.sum(0).cpu() for k, v in vectors.items()}


def summarize(arrays):
    out = {}
    for name, values in arrays.items():
        if name == 'probe_ids':
            continue
        values = np.asarray(values, dtype=float)
        if np.isinf(values).any():
            raise RuntimeError('infinite statistic')
        finite = values[np.isfinite(values)]
        out[name] = dict(count=len(values), undefined=int(len(values)-len(finite)),
                         mean=float(finite.mean()) if len(finite) else None,
                         quantiles=dict(zip(['p05', 'p25', 'p50', 'p75', 'p95'],
                             np.quantile(finite, [.05, .25, .5, .75, .95]).tolist())) if len(finite) else None)
    return out


def fetch(route, dest, expected):
    with BoundedStore.request('GET', route, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(f'checkpoint transport HTTP {response.status_code}')
        with dest.open('xb') as stream:
            for block in response.iter_content(8*1024**2):
                stream.write(block)
    if sha256(dest) != expected:
        raise RuntimeError('source archive SHA mismatch')


def run(args):
    start = time.monotonic()
    configure_gpu()
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('one A100 required')
    plan = json.loads(args.plan.read_text())
    transport = json.loads(args.routes.read_text())
    if digest(plan) != transport['manifest_hash'] or set(transport['routes']) != set(transaction_ids()):
        raise RuntimeError('immutable plan/transport mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    store = BoundedStore(transport['routes'], args.output/'transactions')
    print(json.dumps(dict(phase='quick_kl_start', training_runs=0, optimizer_steps=0,
                         gpu=torch.cuda.get_device_name(), max_records_per_model=SAMPLES)), flush=True)
    materialize(args.input_archive, transport['inputs']['training_sha256'])
    augmentation = json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data = GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),
                         Path('gtsrb_inputs/splits.json'), augmentation)
    ids = data.split['indices']['probe']
    if len(ids) != SAMPLES or len(set(ids)) != SAMPLES or data.split['hashes']['dev'] != DEV_HASH:
        raise RuntimeError('fixed public probe changed')
    if not set(ids).issubset(data.split['indices']['dev']):
        raise RuntimeError('probe must be public dev only')
    models, initial = {}, {}
    for name in CHECKPOINTS:
        row = plan['checkpoints'][name]
        archive = args.output/(name+'_input.tar.gz')
        fetch(transport['inputs'][name+'_get'], archive, row['source_archive_sha256'])
        path = Path('gtsrb_inputs')/(name+'.pt')
        if name == 'pretrain':
            with tarfile.open(archive) as tf:
                member = tf.getmember('public_pretrain_selected.pt')
                if not member.isfile():
                    raise RuntimeError('invalid checkpoint member')
                with tf.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)
            model, _ = load_selected(Path('gtsrb_inputs'), path, SOURCE_SHA)
        else:
            extract_checkpoint(archive, path, row['checkpoint_sha256'])
            model, _ = load_final(path, row)
        # Keep only one model's activations alive; three frozen parameter sets fit easily.
        models[name] = model.cuda().eval()
        initial[name] = tensor_digest(dict(model.named_parameters()))
    arrays_by_model = {name: {} for name in CHECKPOINTS}
    sums = {name: {} for name in CHECKPOINTS}
    completed = {name: 0 for name in CHECKPOINTS}
    committed = []
    stop = False
    for offset in range(0, SAMPLES, CHUNK):
        for name in CHECKPOINTS:
            if time.monotonic()-start >= SOFT_SECONDS:
                stop = True
                break
            model = models[name]
            part = {'probe_ids': np.asarray(ids[offset:offset+CHUNK])}
            buffers = {}
            for pos in range(offset, offset+CHUNK, 8):
                batch_ids = ids[pos:pos+8]
                x, y = data.batch(batch_ids)
                x, y = x.cuda(), y.cuda()
                uniform = public_uniform(x, [data.split['records'][i]['image_id'] for i in batch_ids], 20260921, 'trades')
                adv, _ = attack(model, x, y, objective='kl', uniform=uniform)
                full = gradients(model, x, y, trades_adv=adv, stable_public_kl=True, outer_kl_target='full')
                detached = gradients(model, x, y, trades_adv=adv, stable_public_kl=True, outer_kl_target='detached_clean')
                torch.testing.assert_close(full['g'], detached['g'], rtol=0, atol=0)
                metrics, partial_sums = compare(full['g'], full['h'], detached['h'])
                for key, value in metrics.items():
                    buffers.setdefault(key, []).append(value)
                for key, value in partial_sums.items():
                    sums[name][key] = sums[name].get(key, 0) + value
                del full, detached, partial_sums, x, y, adv
            part.update({key: np.concatenate(value) for key, value in buffers.items()})
            if tensor_digest(dict(model.named_parameters())) != initial[name]:
                raise RuntimeError('diagnostic changed model parameters')
            key = f'{name}_{offset:03d}'
            target = args.output/key
            target.mkdir()
            np.savez_compressed(target/'public_comparison.npz', **part)
            write_json(target/'comparison.json', dict(checkpoint=name, records=CHUNK,
                checkpoint_sha256=plan['checkpoints'][name]['checkpoint_sha256'],
                plan_hash=digest(plan), probe_hash=digest(ids), split_hash=data.split['split_hash'],
                sample_offset=offset, metrics=summarize(part), model_unchanged=True,
                private_examples_used=0, optimizer_steps=0, official_test_loaded=False))
            store.commit(key, target)
            committed.append(key)
            completed[name] += CHUNK
            for metric, value in part.items():
                arrays_by_model[name].setdefault(metric, []).append(value)
            print(json.dumps(dict(phase='quick_kl_progress', completed=completed,
                                 elapsed_seconds=round(time.monotonic()-start, 2))), flush=True)
        if stop:
            break
    summaries = {}
    for name in CHECKPOINTS:
        values = {key: np.concatenate(value) for key, value in arrays_by_model[name].items()}
        aggregate = {}
        for tag in ['h', 's', 'J', 'U', 'SJ', 'SU'] if completed[name] else []:
            pair = paired_vector_metrics(sums[name][tag+'_full']/completed[name], sums[name][tag+'_detached']/completed[name])
            aggregate[tag] = {key: float(value) if torch.isfinite(value) else None for key, value in pair.items()}
        summaries[name] = dict(records=completed[name], per_record=summarize(values), mean_update=aggregate)
    target = args.output/'summary'
    target.mkdir()
    write_json(target/'summary.json', dict(status='PARTIAL_TIME_GUARD' if stop else 'COMPLETE',
        completed=completed, committed=committed, checkpoints=plan['checkpoints'],
        metrics=summaries, elapsed_seconds=time.monotonic()-start,
        inference='Frozen-weight geometry diagnostic; not counterfactual training or robustness accuracy.',
        official_test_loaded=False, private_examples_used=0, optimizer_steps=0))
    store.commit('summary', target)
    print(json.dumps(dict(phase='quick_kl_finished', completed=completed, elapsed_seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/quick_kl'))
    run(parser.parse_args())

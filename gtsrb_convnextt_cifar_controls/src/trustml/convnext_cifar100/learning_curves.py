"""Post-hoc public-dev curves from saved DP iterates; never train or select epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import TrainOnlyData, indices_hash
from .model import build_model, sha256
from .preflight import configure_gpu, write_json
from .runner import canonical_hash, dev_pgd


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_snapshot(model, state, config, step):
    if (state['config_hash'] != canonical_hash(config) or state['config'] != config
            or state['optimizer_steps'] != step or state['privacy_events'] != step
            or state['source_checkpoint_sha256'] != config['checkpoint_sha256']):
        raise ValueError('snapshot/config/step provenance mismatch')
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    values = state['trainable']
    if set(values) != set(expected):
        raise ValueError('snapshot trainable mask mismatch')
    with torch.no_grad():
        for name, parameter in expected.items():
            value = values[name]
            if (value.shape != parameter.shape or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError('invalid snapshot tensor')
            parameter.copy_(value)
    model.eval()


def prediction_metrics(path, expected_ids):
    with np.load(path, allow_pickle=False) as arrays:
        ids = arrays['sample_ids']
        clean, robust, losses = (arrays[k] for k in ['clean_correct', 'pgd_correct', 'clean_loss'])
        if (len(ids) != 5000 or len(np.unique(ids)) != 5000
                or not np.array_equal(ids, np.asarray(expected_ids))
                or any(x.shape != (5000,) for x in [clean, robust, losses])
                or clean.dtype != np.bool_ or robust.dtype != np.bool_
                or np.any(robust & ~clean) or not np.isfinite(losses).all()
                or np.any(losses < 0)):
            raise ValueError('public-dev prediction validation failed')
        return dict(dev_clean=100*float(clean.mean()), dev_pgd20=100*float(robust.mean()),
                    dev_ce_loss=float(losses.mean()), examples=5000, split_hash=indices_hash(ids))


def run(manifest_path, chain, checkpoint, data_root, output_root):
    device = configure_gpu()
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    data = TrainOnlyData(data_root)
    ids = data.split['indices']['dev']
    if indices_hash(ids) != manifest['dev_hash']:
        raise ValueError('ordered dev split changed')
    if sha256(Path(data_root)/'train_only.npz') != manifest['data_sha256']:
        raise ValueError('dataset changed')
    model, architecture = build_model(checkpoint, expected_sha=manifest['checkpoint_sha256'])
    model.to(device)
    write_json(output/'architecture.json', architecture)
    write_json(output/'environment.json', dict(gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, evaluation='unchanged training-bundle dev_pgd'))
    records = []
    for point in manifest['points']:
        if point['chain'] != chain:
            continue
        config = manifest['configs'][point['run_id']]
        snapshot = Path(point['path'])
        if sha256(snapshot) != point['sha256']:
            raise ValueError('snapshot SHA mismatch')
        state = torch.load(snapshot, map_location='cpu', weights_only=True)
        restore_snapshot(model, state, config, point['step'])
        del state
        child = output/point['point_id']
        child.mkdir()
        print(json.dumps(dict(event='curve_point_start', point_id=point['point_id'], step=point['step'])), flush=True)
        started = time.monotonic()
        dev_pgd(model, data, child, batch_size=64)
        metrics = prediction_metrics(child/'dev_predictions.npz', ids)
        record = dict(**point, **metrics, status='PASS', elapsed_seconds=time.monotonic()-started,
            config_hash=canonical_hash(config), prediction_sha256=sha256(child/'dev_predictions.npz'),
            attack_seed=42, steps=20, restarts=1, epsilon=8/255, precision='FP32', model_mode='eval',
            official_test_loaded=False, posthoc_diagnostic=True, checkpoint_selection='unchanged final.pt')
        write_json(child/'COMMITTED.json', record)
        records.append(record)
        write_json(output/'completed_points.json', records)
        print(json.dumps(dict(event='curve_point_complete', **record)), flush=True)
    write_json(output/'summary.json', dict(status='PASS', chain=chain, points=len(records),
        official_test_loaded=False, training_performed=False, manifest_sha256=sha256(manifest_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--chain', required=True, choices=['A', 'B'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.manifest, args.chain, args.checkpoint, args.data, args.output)


if __name__ == '__main__':
    main()

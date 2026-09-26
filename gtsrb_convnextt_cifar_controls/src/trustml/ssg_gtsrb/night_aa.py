"""Transactional full public-dev AA of verified rect57 final iterates, no test."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import torch

from scripts.platform.gtsrb_inputs import materialize
from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
from trustml.convnext_gtsrb.model import build_model, sha256
from .persistence import TransactionStore
from .real_dev_aa import full_aa, DEV_HASH
from .rectangular import fetch, SOURCE_SHA


def extract_checkpoint(archive, destination, expected_hash):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise RuntimeError('existing checkpoint differs')
        return destination
    with tarfile.open(archive) as tf:
        member = tf.getmember('checkpoints/final.pt')
        if not member.isfile():
            raise RuntimeError('invalid final checkpoint member')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tf.extractfile(member) as src, destination.open('xb') as out:
            shutil.copyfileobj(src, out)
    if sha256(destination) != expected_hash:
        raise RuntimeError('final checkpoint SHA mismatch')
    return destination


def load_final(path, row, assets=Path('gtsrb_inputs')):
    if sha256(path) != row['checkpoint_sha256']:
        raise RuntimeError('checkpoint identity mismatch')
    metadata = json.loads((assets/'assets.json').read_text())['convnext_tiny_cvst_clean.pt']
    model, architecture = build_model(assets/'convnext_tiny_cvst_clean.pt', metadata, seed=42)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['optimizer_steps'] != 3436 or state['scientific_config'] != row['config']
            or state['source_checkpoint_sha256'] != SOURCE_SHA
            or state['trainable_mask_hash'] != architecture['trainable_mask_hash']):
        raise RuntimeError('final checkpoint protocol mismatch')
    model.load_state_dict(state['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise RuntimeError('nonfinite checkpoint')
    return model.eval(), architecture


def get_final(row, route, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/(row['run_id']+'.tar.gz')
    if not archive.exists():
        fetch(route['source_get'], archive, row['source_archive_sha256'])
    elif sha256(archive) != row['source_archive_sha256']:
        raise RuntimeError('cached source archive mismatch')
    return extract_checkpoint(archive, cache/row['run_id']/'final.pt', row['checkpoint_sha256'])


def run(args):
    configure_gpu()
    if torch.cuda.device_count() != 1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('exactly one A100 required')
    plan = json.loads(args.manifest.read_text())
    routes = json.loads(args.routes.read_text())
    if routes['manifest_hash'] != digest(plan):
        raise RuntimeError('manifest changed')
    rows = plan['chains'][args.chain]
    if {r['run_id'] for r in rows} != set(routes['routes']):
        raise RuntimeError('chain route coverage mismatch')
    if not Path('gtsrb_inputs').exists():
        materialize(args.input_archive, routes['inputs']['training_sha256'])
    aug = json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data = GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'), Path('gtsrb_inputs/splits.json'), aug)
    if data.split['hashes']['dev'] != DEV_HASH:
        raise RuntimeError('split mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    store = TransactionStore(routes['routes'], args.output/'transactions')
    done = []
    start = time.time()
    for row in rows:
        rid = row['run_id']
        route = routes['routes'][rid]
        with store.request('GET', route['receipt_get']) as response:
            if response.status_code == 200:
                receipt = response.json()
                if receipt.get('status') != 'COMMITTED' or receipt.get('stage') != rid:
                    raise RuntimeError('invalid existing receipt')
                # Frozen manifest gives each key one checkpoint/config/attack identity.
                done.append(dict(run_id=rid, status='ALREADY_COMMITTED', receipt=receipt))
                continue
            if response.status_code != 404:
                raise RuntimeError('uncertain receipt; do not repeat evaluation')
        path = get_final(row, route, Path('evaluation_inputs'))
        model, architecture = load_final(path, row)
        model.cuda().eval()
        torch.manual_seed(42)
        np.random.seed(42)
        target = args.output/rid
        target.mkdir()
        write_json(target/'scientific_config.json', row['config'])
        write_json(target/'provenance.json', dict(run_id=rid, checkpoint_sha256=row['checkpoint_sha256'],
            config_sha256=digest(row['config']), manifest_hash=digest(plan),
            source_archive_sha256=row['source_archive_sha256'], architecture=architecture,
            official_test_loaded=False))
        print(json.dumps(dict(phase='aa_child_start', run_id=rid, gpu=torch.cuda.get_device_name())), flush=True)
        metrics = full_aa(model, data, target/'aa_4020', checkpoint_sha256=row['checkpoint_sha256'],
                          expected_clean=row['clean_correct'])
        receipt = store.commit(rid, target)
        done.append(dict(run_id=rid, metrics=metrics, receipt=receipt))
        write_json(args.output/'completed.json', done)
        del model
        torch.cuda.empty_cache()
        if metrics['status'] != 'PASS':
            raise RuntimeError('numerical AA warning persisted; inspect before continuing')
    write_json(args.output/'summary.json', dict(status='COMPLETE', chain=args.chain,
        completed=len(done), chain_start=start, chain_end=time.time(), total_gpu_seconds=time.time()-start))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--routes', type=Path, required=True)
    p.add_argument('--chain', required=True)
    p.add_argument('--input-archive', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/night_aa'))
    run(p.parse_args())

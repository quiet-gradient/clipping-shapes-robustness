"""Finish final public diagnostics from immutable weights; never train/resume."""
import argparse
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

import torch

from .common import code_identity, diagnostic_rng, provenance, tensor_digest
from .data import GTSRBTrainOnly, digest
from .evaluation import evaluate
from .mechanism import lr_at
from .model import build_model, sha256
from .probe import audit_geometry
from .runner import validate_config
from .validation import validate_evaluation, validate_geometry, validate_lr
from trustml.convnext_cifar100.preflight import configure_gpu, write_json


def extract_regular(archive, destination, prefix=''):
    """Fresh destination, regular files only, no traversal or symlink following."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    skipped = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('unsafe archive path')
            if prefix and not member.name.startswith(prefix):
                if member.isdir():
                    continue
                raise ValueError('unexpected archive root')
            relative = member.name[len(prefix):]
            if not member.isfile():
                if not member.isdir():
                    skipped.append(member.name)
                continue
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest)
    return skipped


def verify_training(source, expected):
    source = Path(source)
    config = json.loads((source/'config.json').read_text())
    validate_config(config)
    if digest(config) != expected['config_hash'] or config['run_id'] != expected['run_id']:
        raise ValueError('training config differs from recovery manifest')
    checkpoint = source/'checkpoints/final.pt'
    if sha256(checkpoint) != expected['checkpoint_sha256']:
        raise ValueError('final checkpoint checksum mismatch')
    updates = [json.loads(line) for line in (source/'history.jsonl').read_text().splitlines()]
    if [r['step'] for r in updates] != list(range(1, config['steps']+1)):
        raise ValueError('training incomplete: recovery must not run optimizer steps')
    for r in updates:
        if r['release_count'] != (r['step'] if config['private'] else 0):
            raise ValueError('release count mismatch in training log')
        validate_lr(r['step'], r['lr'], config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['optimizer_steps'] != config['steps'] or state['config_hash'] != digest(config) or state['config'] != config:
        raise ValueError('checkpoint identity/steps mismatch')
    if state['release_count'] != (config['steps'] if config['private'] else 0):
        raise ValueError('checkpoint release count mismatch')
    if state['source_checkpoint_hash'] != config['source_metadata']['sha256']:
        raise ValueError('checkpoint source mismatch')
    if not all(torch.isfinite(v).all() for v in state['model'].values()):
        raise ValueError('scientific failure: nonfinite trained weights')
    for name, value in state['trainable'].items():
        if not torch.equal(value, state['model'][name]):
            raise ValueError('trainable/full checkpoint mismatch')
    return config, state, updates


def recover(source, assets_root, output, expected):
    started = time.monotonic()
    source, root, output = Path(source), Path(assets_root), Path(output)
    config, state, updates = verify_training(source, expected)
    output.mkdir(parents=True, exist_ok=False)
    device = configure_gpu()
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    code = code_identity()
    environment = {'packages':{k:importlib.metadata.version(k) for k in
        ['torch','torchvision','timm','numpy','scipy','opacus','wandb','pyarrow','Pillow']},
        'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda, 'FP32':True, 'TF32':False}
    original_env = json.loads((source/'environment.json').read_text())
    if environment['packages'] != original_env['packages']:
        raise ValueError('recovery package versions differ from training environment')
    write_json(output/'recovery_environment.json', environment)
    if sha256(root/'GTSRB_Final_Training_Images.zip') != config['data_sha256']:
        raise ValueError('dataset hash mismatch')
    data = GTSRBTrainOnly(root/'GTSRB_Final_Training_Images.zip', root/'splits.json', config['augmentation'])
    if data.split['split_hash'] != config['split_hash'] or data.split['hashes']['probe'] != config['probe_hash']:
        raise ValueError('split/probe changed')
    model, architecture = build_model(root/'convnext_tiny_cvst_clean.pt', config['source_metadata'], seed=config['seed'])
    previous_arch = json.loads((source/'architecture.json').read_text())
    for key in ['trainable_mask_hash','model_architecture_digest']:
        if architecture[key] != previous_arch[key]:
            raise ValueError('architecture changed')
    model.load_state_dict(state['model'], strict=True)
    before = tensor_digest(model.state_dict())
    del state  # No optimizer is created or restored, and no optimizer.step exists.
    model.to(device).eval()
    checkpoint = source/'checkpoints/final.pt'
    identity = provenance(config, data.split, architecture, checkpoint, step=config['steps'],
        epoch=config['steps']*config['sample_rate'], lr=updates[-1]['lr'], code=code)
    identity.update(nominal_expected_epoch=50, recovery=True,
        training_code_identity=json.loads((source/'code_identity.json').read_text()))
    phases = []
    with diagnostic_rng():
        t = time.monotonic()
        metrics = evaluate(model, data, output/'evaluation_epoch050', seed=config['attack_seed'], final=True)
        attack_config = {'epsilon':config['attack_epsilon'], 'pgd_steps':20, 'pgd_restarts':1,
            'seed':config['attack_seed'], 'final_attacks':config['final_attacks'], 'subset_size':1024, 'precision':'FP32'}
        write_json(output/'evaluation_epoch050/provenance.json', {**identity, 'attack_seed':config['attack_seed'],
            'attack_config':attack_config, 'attack_hash':digest(attack_config)})
        validate_evaluation(output/'evaluation_epoch050', data.split, final=True)
        phases.append({'phase':'evaluation', 'epoch':50, 'seconds':time.monotonic()-t, 'recovery':True})
        print(json.dumps({'stage':'recovered_evaluation', 'run_id':config['run_id'], **metrics}), flush=True)
        t = time.monotonic()
        audit_geometry(model, data, architecture, config['calibration'], config, output/'geometry_epoch050',
            identity, microbatch=config['microbatch'])
        validate_geometry(output/'geometry_epoch050', data.split)
        phases.append({'phase':'geometry', 'epoch':50, 'seconds':time.monotonic()-t, 'recovery':True})
    if tensor_digest(model.state_dict()) != before or sha256(checkpoint) != expected['checkpoint_sha256']:
        raise ValueError('diagnostics modified trained weights')
    curves = json.loads((source/'gtsrb_pilot_learning_curves.json').read_text())
    if [r['epoch'] for r in curves] != config['checkpoint_epochs'][:-1]:
        raise ValueError('unexpected prior evaluations')
    row = {'epoch':50, 'expected_epoch':config['steps']*config['sample_rate'], 'step':config['steps'],
        'LR':updates[-1]['lr'], 'checkpoint_sha256':expected['checkpoint_sha256'], **metrics}
    curves.append(row)
    write_json(output/'gtsrb_pilot_learning_curves.json', curves)
    privacy = json.loads((source/'gtsrb_privacy_accounting.json').read_text())
    expected_releases = config['steps'] if config['private'] else 0
    history = []
    if config['private']:
        from opacus.accountants import RDPAccountant, PRVAccountant
        history = [(config['sigma'], config['sample_rate'], expected_releases)]
        rdp, prv = RDPAccountant(), PRVAccountant()
        rdp.history, prv.history = history.copy(), history.copy()
        privacy['achieved_rdp_epsilon_from_history'] = rdp.get_epsilon(delta=config['delta'], alphas=privacy['orders'])
        privacy['achieved_prv_epsilon_from_history'] = prv.get_epsilon(delta=config['delta'], eps_error=.1, delta_error=1e-8)
        if max(privacy['achieved_rdp_epsilon_from_history'], privacy['achieved_prv_epsilon_from_history']) > 4:
            raise ValueError('achieved epsilon exceeds 4')
    reconstruction = 'Reconstructed from verified per-step release counts, final checkpoint count and immutable training config; not original in-memory accountant serialization.'
    privacy['recovery_accountant_provenance'] = reconstruction
    write_json(output/'gtsrb_privacy_accounting.json', privacy)
    write_json(output/'accountant_history.json', {'rdp':history, 'prv':history, 'release_count':expected_releases,
        'reconstructed':True, 'provenance':reconstruction})
    compute = json.loads((source/'gtsrb_compute_ledger.json').read_text())
    compute += phases + [{'phase':'training', 'seconds':sum(r['step_seconds'] for r in updates),
        'reconstructed_from_history':True}]
    write_json(output/'gtsrb_compute_ledger.json', compute)
    recovery = {**expected, 'status':'PASS', 'recovery_code_identity':code,
        'optimizer_steps_added':0, 'releases_added':0, 'weights_unchanged':True,
        'weights_tensor_digest':before, 'seconds':time.monotonic()-started,
        'original_failure':'APGD-T check_zero_gradients grad.view on noncontiguous input',
        'change':'AA boundary only: materialize contiguous NCHW, no AA wheel or hyperparameter changes',
        'accountant_provenance':reconstruction, 'official_test_loaded':False,
        'compute_caveat':'Prior failed final-evaluation and original setup/teardown overhead not included in phase ledger.'}
    write_json(output/'recovery.json', recovery)
    final = {'status':'PASS', 'run_id':config['run_id'], 'role':config['role'], 'config_hash':digest(config),
        'optimizer_steps':config['steps'], 'release_count':expected_releases,
        'checkpoint_sha256':expected['checkpoint_sha256'], 'privacy':privacy, 'dev':row,
        'wandb_run_id':expected['wandb_run_id'], 'git_commit':identity['training_code_identity']['git_commit'],
        'recovery_git_commit':code['git_commit'], 'elapsed_seconds':updates[-1]['elapsed_seconds']+recovery['seconds'],
        'elapsed_seconds_note':'Measured through last training step plus recovery; excludes failed final attack and queue/transfer.',
        'gpu':environment['gpu'], 'official_test_loaded':False, 'checkpoint_epochs':[r['epoch'] for r in curves],
        'geometry_epochs':config['geometry_epochs'], 'final_iterate':'epoch50, unchanged trained weights, recovered diagnostics',
        'recovered':True}
    write_json(output/'summary.json', final)
    write_json(output/'artifact_manifest.json', {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(output, arcname=output.name)
    receipt = {'status':'COMMITTED', 'run_id':config['run_id'], 'config_hash':digest(config),
        'checkpoint_sha256':expected['checkpoint_sha256'], 'archive_sha256':sha256(archive),
        'archive_bytes':archive.stat().st_size, 'optimizer_steps_added':0, 'overlay_only':True}
    write_json(output.with_suffix('.COMMITTED.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--archive')
    p.add_argument('--archive-sha')
    p.add_argument('--input-root', default='recovery_inputs')
    p.add_argument('--assets-root', default='gtsrb_inputs')
    p.add_argument('--output', default='outputs/recovery')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if args.archive:
        if sha256(args.archive) != args.archive_sha:
            raise ValueError('recovery input archive checksum mismatch')
        extract_regular(args.archive, args.input_root, prefix='recovery_inputs/')
    for expected in manifest['runs']:
        for name, value in expected['input_hashes'].items():
            if sha256(Path(args.input_root)/expected['run_id']/name) != value:
                raise ValueError('recovery input file mismatch: '+name)
        recover(Path(args.input_root)/expected['run_id'], args.assets_root,
            Path(args.output)/expected['run_id'], expected)

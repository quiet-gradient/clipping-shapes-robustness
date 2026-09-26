"""Locked GTSRB confirmation/mechanism wave; no search, no test-set access.

Reuses the tested per-record maps from CIFAR, NOT its loss or training loop.
All gradients are produced by the existing GTSRB full-outer-KL implementation.
"""
import copy
import hashlib
import hmac
import json
from pathlib import Path

import torch
import yaml

from trustml.cifar_interventions.mechanism import contribution, ARMS
from trustml.convnext_gtsrb.geometry import vector_norm
from .rectangular import SOURCE_SHA

CAMPAIGN = 'gtsrb_confirmation_dmr_20260924'
SEEDS = (45, 46, 47)
LOCKS = {
    'hard': Path('configs/ssg_gtsrb_rect57_v2/hard_selection_locked.yaml'),
    'soft': Path('configs/ssg_gtsrb_autos30_v1/soft_selection_locked.yaml'),
}


def require_private_launch(decision):
    """Authorize compute separately from completing the scientific diagnostics."""
    if decision.get('privacy_crosscheck_exception_authorized') is not True:
        raise RuntimeError('private launch gate: privacy crosscheck exception not authorized')
    if (decision.get('attack_gate_closed') is not True
            and decision.get('parallel_training_while_attack_diagnostics_authorized') is not True):
        raise RuntimeError('private launch gate: attacks pending and parallel training not authorized')


def validate_intervention(row):
    arm = row['intervention']
    if (arm not in ARMS or row['role'] != ('Joint' if arm == 'J' else 'Componentwise')
            or row['beta'] != 1 or row.get('outer_kl_target') != 'full'
            or row['C'] != 1. or not 0 < row['intervention_q'] < 1
            or row['q'] != (None if arm == 'J' else row['intervention_q'])):
        raise ValueError('invalid fixed GTSRB intervention')


def intervention_query(values, row):
    validate_intervention(row)
    # Avoid materializing every counterfactual at physical128 (15.5M parameters).
    from trustml.convnext_gtsrb.mechanism import contribution as hard
    from .autos_map import query as soft
    arm=row['intervention'];q=row['intervention_q'];C=row['C']
    def branch(role):
        return (soft(values,role,row['gamma'],q) if row.get('clipping_map')=='auto_s'
                else hard(values,role,C,q))
    if arm=='J':return branch('Joint')
    if arm=='U':return branch('Componentwise')
    g,h=values['g'],values['h']
    if arm=='R':
        norm=g.norm(dim=-1,keepdim=True)+h.norm(dim=-1,keepdim=True)
        if row.get('clipping_map')=='auto_s':return (g+h)/(norm+row['gamma'])
        return (g+h)*(C/norm.clamp_min(torch.finfo(norm.dtype).tiny)).clamp(max=1)
    radius=branch('Componentwise' if arm=='D' else 'Joint').norm(dim=-1,keepdim=True)
    direction=branch('Joint' if arm=='D' else 'Componentwise')
    norm=direction.norm(dim=-1,keepdim=True)
    return direction*(radius/torch.where(norm>0,norm,torch.ones_like(norm)))


def intervention_coefficients(gram, actual):
    """Same queries expressed in (g,h,a) basis for streaming public geometry."""
    basis = torch.eye(3, device=gram.device, dtype=gram.dtype)
    g, h = [v.expand(len(gram), -1) for v in basis[:2]]
    s = g+h
    ng, nh, ns = [vector_norm(v, gram) for v in (g, h, s)]
    q, C = actual['intervention_q'], actual['C']
    tiny = torch.finfo(gram.dtype).tiny
    if actual.get('clipping_map') == 'auto_s':
        gamma = actual['gamma']
        j = s/(ns+gamma)[:, None]
        u = (1-q)*g/(ng+(1-q)*gamma)[:, None]+q*h/(nh+q*gamma)[:, None]
        r = s/(ng+nh+gamma)[:, None]
    else:
        j = s*(C/ns.clamp_min(tiny)).clamp(max=1)[:, None]
        u = (g*((1-q)*C/ng.clamp_min(tiny)).clamp(max=1)[:, None]
             +h*(q*C/nh.clamp_min(tiny)).clamp(max=1)[:, None])
        r = s*(C/(ng+nh).clamp_min(tiny)).clamp(max=1)[:, None]
    nj, nu = vector_norm(j, gram), vector_norm(u, gram)
    d = j*torch.where(nj > 0, nu/nj.clamp_min(tiny), torch.zeros_like(nj))[:, None]
    m = u*torch.where(nu > 0, nj/nu.clamp_min(tiny), torch.zeros_like(nu))[:, None]
    return dict(J=j, U=u, D=d, M=m, R=r)


def manifest():
    # Read the accepted locks; fail rather than drifting to an amended grid.
    soft = yaml.safe_load(LOCKS['soft'].read_text())
    hard = yaml.safe_load(LOCKS['hard'].read_text())
    if (hard['roles']['Joint_practical']['learning_rate'] != .1
            or hard['roles']['Componentwise']['learning_rate'] != .025
            or hard['roles']['Componentwise']['q'] != 1/3):
        raise ValueError('unexpected hard lock')
    if not soft['final_lock'] or soft['roles']['Componentwise']['learning_rate'] != .04304104621706269:
        raise ValueError('unexpected soft lock')
    if soft['roles']['Componentwise']['q'] != .25:
        raise ValueError('unexpected soft allocation')
    groups = []
    for family, lr, q in [('hard', .025, 1/3), ('soft', .04304104621706269, .25)]:
        for seed in SEEDS:
            name = f'{family}_dmr_s{seed}'
            rows = []
            for arm in ARMS:
                row = dict(run_id=f'confirm_{family}_{arm.lower()}_s{seed}', role='Joint' if arm == 'J' else 'Componentwise',
                    optimizer='SGD', C=1., lr=lr, q=None if arm == 'J' else q, beta=1, private=True,
                    source_checkpoint_sha256=SOURCE_SHA, seed_label=seed, outer_kl_target='full',
                    clipping_map='auto_s' if family == 'soft' else 'hard', intervention=arm,
                    intervention_q=q, rng_coupling='matched_secret_group', coupling_group=name,
                    joint_release_dp_claim=False)
                if family == 'soft': row['gamma'] = .1
                if arm != 'J': row['matched_joint_run_id'] = f'confirm_{family}_j_s{seed}'
                validate_intervention(row)
                rows.append(row)
            groups.append(dict(group=name, runs=rows, coupled=True))
    specs = [('hard_practical_joint', 'Joint', .1, 1., 'hard'),
             ('hard_cleandp', 'CleanDP', .1, 1., 'hard'),
             ('soft_cleandp', 'CleanDP', .08084591581898012, 1., 'auto_s'),
             ('natural', 'Natural', .05, None, 'hard')]
    for name, role, lr, C, clipping in specs:
        rows = []
        for seed in SEEDS:
            row = dict(run_id=f'confirm_{name}_s{seed}', role=role, optimizer='SGD', C=C, lr=lr,
                q=None, beta=1 if role == 'Joint' else 0, private=role!='Natural',
                source_checkpoint_sha256=SOURCE_SHA, seed_label=seed, outer_kl_target='full',
                clipping_map=clipping, rng_coupling='independent_per_run')
            if clipping == 'auto_s': row['gamma'] = .1
            rows.append(row)
        groups.append(dict(group=name, runs=rows, coupled=False))
    return dict(campaign=CAMPAIGN, training_runs=42, groups=groups, seeds=list(SEEDS),
        source_checkpoint_sha256=SOURCE_SHA, full_dev_aa_after_each_training=True,
        official_test_allowed=False, marginal_dp_only=True, joint_release_dp_claim=False,
        confirmation_reselection_allowed=False, max_parent_hours=72)


def triplet_groups(plan):
    """Transpose the six scientific five-arm blocks; never change a run config."""
    original={g['group']:g for g in plan['groups']}
    groups=[]
    for family in ['hard','soft']:
        for arm in ARMS:
            rows=[next(r for r in original[f'{family}_dmr_s{seed}']['runs']
                       if r['intervention']==arm) for seed in SEEDS]
            groups.append(dict(group=f'{family}_{arm.lower()}',runs=rows,coupled=False,
                               cross_job_coupled=True))
    return groups


def expected_stream_commitments(master, row):
    from .stages import coupling_key
    anchor=coupling_key(row)
    return {name:hashlib.sha256(str(int.from_bytes(hmac.new(master,
        f'{anchor}/{name}'.encode(),hashlib.sha256).digest()[:8],'big') % 2**63).encode()).hexdigest()
        for name in ['sampler','attack','noise','augmentation']}


def protected_keys(document, execution, plan_hash, group, *, layout):
    """Validate confidential job input without ever echoing its content."""
    wanted={r['coupling_group'] for r in group['runs']}
    if (document.get('manifest_hash')!=plan_hash or document.get('group')!=group['group']
            or execution.get('layout')!=layout or execution.get('group')!=group['group']
            or set(document.get('roots',{}))!=wanted
            or set(execution.get('root_commitments',{}))!=wanted):
        raise RuntimeError('protected RNG input identity mismatch')
    try:
        roots={k:bytes.fromhex(v) for k,v in document['roots'].items()}
    except (ValueError,TypeError):
        raise RuntimeError('invalid protected RNG input encoding') from None
    if any(len(v)!=32 or hashlib.sha256(v).hexdigest()!=execution['root_commitments'][k]
           for k,v in roots.items()):
        raise RuntimeError('protected RNG input commitment mismatch')
    expected={r['run_id']:expected_stream_commitments(roots[r['coupling_group']],r)
              for r in group['runs']}
    if execution.get('stream_commitments')!=expected:
        raise RuntimeError('protected RNG stream commitment mismatch')
    return roots


def triplet_keys(document, execution, plan_hash, group):
    return protected_keys(document, execution, plan_hash, group, layout='arm_triplets')


def public_gate(model, data, rows, output):
    from trustml.convnext_gtsrb.attacks import attack, public_uniform
    from trustml.convnext_gtsrb.mechanism import gradients, contribution as hard_query, private_release, assign_gradient
    from trustml.convnext_cifar100.preflight import write_json
    from .autos_map import query as soft_query
    params=[p for p in model.parameters() if p.requires_grad]
    saved=[p.detach().clone() for p in params]
    ids=data.split['indices']['probe'][:128]
    x,y=data.batch(ids);x,y=x.cuda(),y.cuda()
    uniform=public_uniform(x,[data.split['records'][i]['image_id'] for i in ids],20260921,'trades')
    adv,_=attack(model,x,y,objective='kl',uniform=uniform)
    torch.cuda.reset_peak_memory_stats()
    values=gradients(model,x,y,trades_adv=adv,beta=1.,outer_kl_target='full')
    single=gradients(model,x[:1],y[:1],trades_adv=adv[:1],beta=1.,outer_kl_target='full')
    for k in ['g','h']:torch.testing.assert_close(values[k][:1],single[k],rtol=5e-4,atol=1e-5)
    checks=[]
    for row in rows:
        if row['role']=='Natural':
            continue  # Unchanged unclipped CE path has existing public preflight.
        if 'intervention' in row:
            u=intervention_query(values,row)
            separate=intervention_query({k:v[:1] for k,v in values.items()},row)
            torch.testing.assert_close(u[:1],separate)
        elif row.get('clipping_map')=='auto_s':
            u=soft_query(values,row['role'],row['gamma'],row['q'])
        else:
            u=hard_query(values,'DPOnly' if row['role']=='CleanDP' else row['role'],row['C'],row['q'])
        if 'uniform_joint_scale' in row:
            u = u * row['uniform_joint_scale']
        if not torch.isfinite(u).all() or (u.norm(dim=-1)>row['C']*(1+2e-5)).any():
            raise RuntimeError('public intervention sensitivity/finite failure')
        gen=torch.Generator(device='cuda').manual_seed(20260924)
        sigma = 0. if row.get('zero_noise_control') else 1.22344970703125
        sanitized=private_release(u.sum(0),sigma=sigma,bound=row['C'],generator=gen,denominator=512)
        opt=torch.optim.SGD(params,lr=row['lr'],momentum=.9,nesterov=True,weight_decay=5e-4)
        assign_gradient(model,sanitized);opt.step()
        if not all(torch.isfinite(p).all() for p in params):raise RuntimeError('public optimizer failure')
        with torch.no_grad():
            for p,v in zip(params,saved):p.copy_(v)
        checks.append(dict(run_id=row['run_id'],max_norm=float(u.norm(dim=-1).max())))
        del u,opt,sanitized
    write_json(output,dict(status='PASS',public_only=True,physical_batch=128,
        logical_batch=512,outer_kl_target='full',checks=checks,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name()))
    for p in params:p.grad=None


def run(args, *, plan=None):
    import secrets
    import shutil
    import tarfile
    import time
    import numpy as np
    from scripts.platform.gtsrb_inputs import materialize
    from trustml.convnext_cifar100.preflight import configure_gpu, write_json
    from trustml.convnext_gtsrb.common import code_identity, diagnostic_rng
    from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
    from trustml.convnext_gtsrb.model import sha256
    from .campaign import verify_completed_run, check_coupling
    from .private import train, load_selected
    from .rectangular import fetch
    from .rectangular_v2 import protocol
    from .persistence import TransactionStore
    from .night_aa import load_final
    from .real_dev_aa import DEV_HASH, full_aa

    code_identity()
    payload=json.loads(args.routes.read_text());plan=manifest() if plan is None else plan
    if payload['manifest_hash']!=digest(plan):raise RuntimeError('confirmation manifest changed')
    layout=getattr(args,'layout','seed_blocks')
    groups=triplet_groups(plan) if layout=='arm_triplets' else plan['groups']
    group=next(g for g in groups if g['group']==args.group)
    roots=None
    if layout in {'arm_triplets', 'single_controls'}:
        if not args.rng_input:raise RuntimeError('protected matched RNG input required')
        roots=protected_keys(json.loads(args.rng_input.read_text()),payload['execution'],digest(plan),group,
                             layout=layout)
    elif getattr(args,'rng_input',None):
        raise RuntimeError('external RNG inputs only permitted for arm-triplet execution')
    if any(r['private'] for r in group['runs']):
        decision=json.loads(Path('confirmation_decision.json').read_text())
        require_private_launch(decision)
    configure_gpu()
    if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('one A100 required')
    print(json.dumps(dict(phase='confirmation_start',group=args.group,gpu=torch.cuda.get_device_name())),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'execution.json',dict(layout=layout,group=args.group,
        run_ids=[r['run_id'] for r in group['runs']],
        secret_root_location='protected_job_input_not_output' if roots else 'RAM_only',
        cross_job_coupling_audit='PENDING_SIBLING_COLLECTION' if roots else 'not_applicable',
        official_test_loaded=False))
    archive=args.input_archive or Path('training_inputs.tar')
    if not args.input_archive:fetch(payload['inputs']['training_url'],archive,payload['inputs']['training_sha256'])
    materialize(archive,payload['inputs']['training_sha256'])
    aug=json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data=GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),Path('gtsrb_inputs/splits.json'),aug)
    if data.split['hashes']['dev']!=DEV_HASH:raise RuntimeError('dev split mismatch')
    source_archive=Path('public_selected.tar.gz')
    fetch(payload['inputs']['selected_url'],source_archive,payload['inputs']['selected_archive_sha256'])
    selected=Path('gtsrb_inputs/public_pretrain_selected.pt')
    with tarfile.open(source_archive) as tf:
        m=tf.getmember('public_pretrain_selected.pt')
        if not m.isfile():raise RuntimeError('unsafe source member')
        with tf.extractfile(m) as src,selected.open('xb') as dest:shutil.copyfileobj(src,dest)
    if sha256(selected)!=SOURCE_SHA:raise RuntimeError('source checkpoint mismatch')
    p=protocol();p['campaign']=plan['campaign']
    calibration=json.loads(Path('configs/ssg_gtsrb_rect54/calibration.json').read_text())
    privacy=json.loads(Path('configs/ssg_gtsrb_rect54/accounting.json').read_text())
    if privacy['noise_multiplier']!=1.22344970703125:raise RuntimeError('noise multiplier changed')
    store=TransactionStore(payload['routes'],args.output/'transactions')
    # A lost secret cannot reproduce the coupling of previously completed siblings.
    # Preserve their artifacts and require an explicit recovery design; never retrain silently.
    for row in group['runs']:
        for key in [row['run_id'],row['run_id']+'_aa']:
            with store.request('GET',payload['routes'][key]['receipt_get']) as response:
                if response.status_code!=404:raise RuntimeError('existing/uncertain receipt; reconcile without retraining')
    with diagnostic_rng():
        model,_=load_selected(Path('gtsrb_inputs'),selected,SOURCE_SHA);model.cuda().eval()
        public_gate(model,data,group['runs'],args.output/'public_gpu_gate.json')
        del model;torch.cuda.empty_cache()
    print(json.dumps(dict(phase='confirmation_public_gate_pass',group=args.group)),flush=True)
    master=secrets.token_bytes(32)
    completed=[];started=time.time()
    def commit(key,path):
        receipt=store.commit(key,path)
        if args.mlm:
            from scripts.platform.mlm.publish_job_artifacts import publish_job_artifacts
            try:
                publish_job_artifacts([store.root/(key+'.tar.gz'),store.root/(key+'.COMMITTED.json')])
            except Exception as exc:
                write_json(store.root/(key+'.mlm_mirror_pending.json'),dict(
                    status='MLM_MIRROR_PENDING_S3_COMMITTED',error_class=type(exc).__name__))
        return receipt
    for row in group['runs']:
        rid=row['run_id'];target=args.output/rid
        run_master=roots[row['coupling_group']] if roots else master
        result=train(row,p,data,calibration,privacy,Path('gtsrb_inputs'),selected,run_master,target)
        verify_completed_run(target,row,result)
        if roots:
            if result['randomness_audit']['stream_commitments']!=payload['execution']['stream_commitments'][rid]:
                raise RuntimeError('scientific failure: cross-job RNG commitments differ')
            write_json(target/'cross_job_rng_provenance.json',dict(
                coupling_group=row['coupling_group'],
                root_commitment=payload['execution']['root_commitments'][row['coupling_group']],
                expected_stream_commitments=payload['execution']['stream_commitments'][rid],
                secret_root_location='protected_job_input_not_output',
                sibling_final_position_audit='PENDING_SIBLING_COLLECTION',joint_release_dp_claim=False))
        if group['coupled'] and completed:check_coupling(completed[0],result)
        commit(rid,target)  # Checkpoint is durable before an attack can fail.
        completed.append(result);write_json(args.output/'completed_training.json',completed)
        aa_row=dict(config=row,checkpoint_sha256=result['checkpoint_sha256'])
        model,_=load_final(target/'checkpoints/final.pt',aa_row);model.cuda().eval()
        npz=np.load(target/'evaluation_epoch050/full_dev_pgd20/public_dev_predictions.npz')
        if not np.array_equal(npz['dev_ids'],data.split['indices']['dev']):raise RuntimeError('training dev IDs differ')
        # Existing public evaluator uses the same ordered-dev records.
        expected=npz['clean_correct']
        with diagnostic_rng():
            metrics=full_aa(model,data,args.output/(rid+'_aa'),checkpoint_sha256=result['checkpoint_sha256'],expected_clean=expected)
        commit(rid+'_aa',args.output/(rid+'_aa'))
        del model;torch.cuda.empty_cache()
        if metrics['status']!='PASS':raise RuntimeError('scientific AA warning; training/AA retained')
    write_json(args.output/'summary.json',dict(status='COMPLETE',group=args.group,
        trainings=len(completed),total_gpu_seconds=time.time()-started,official_test_loaded=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,required=True);p.add_argument('--group',required=True)
    p.add_argument('--input-archive',type=Path);p.add_argument('--mlm',action='store_true')
    p.add_argument('--layout',choices=['seed_blocks','arm_triplets','single_controls'],default='seed_blocks')
    p.add_argument('--rng-input',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/confirmation'))
    run(p.parse_args())

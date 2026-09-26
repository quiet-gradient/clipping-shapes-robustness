"""Calibrate six public LR centers, then execute explicitly assigned AUTO-S runs."""
import argparse
import json
from pathlib import Path
import secrets
import shutil
import tarfile
import time

import torch

from scripts.platform.gtsrb_inputs import materialize
from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_cifar100.mechanism import clip
from trustml.convnext_gtsrb.common import diagnostic_rng
from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
from trustml.convnext_gtsrb.model import sha256
from trustml.convnext_gtsrb.mechanism import gradients, private_release, assign_gradient
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.probe import probe_batches
from .autos_map import query
from .night_aa import get_final, load_final
from .persistence import TransactionStore
from .private import load_selected, train
from .campaign import verify_completed_run
from .rectangular import fetch, SOURCE_SHA
from .rectangular_v2 import protocol
from .real_dev_aa import DEV_HASH

GAMMAS = [.01, .1, 1.]
MULTIPLIERS = [.5, 1., 2.]
QS = [('q025', .25), ('q1of3', 1/3), ('q050', .5), ('q075', .75)]
DS_RUNS = ['soft30_joint_g0100_lr100_s42', 'soft30_cleandp_g0100_lr100_s42']


def make_rows(centers):
    rows = []
    for gamma in GAMMAS:
        gt = f'g{round(1000*gamma):04d}'
        for role in ['CleanDP', 'Joint']:
            center = centers[f'{role}/{gamma:g}']['center']
            for multiplier in MULTIPLIERS:
                tag = f'lr{round(100*multiplier):03d}'
                rows.append(dict(run_id=f'soft30_{role.lower()}_{gt}_{tag}_s42', role=role,
                    clipping_map='auto_s', gamma=gamma, C=1., lr=center*multiplier,
                    lr_center=center, lr_multiplier=multiplier, q=None, beta=0 if role=='CleanDP' else 1,
                    optimizer='SGD', private=True, source_checkpoint_sha256=SOURCE_SHA, seed_label=42,
                    rng_coupling='independent_per_run'))
    for qtag,q in QS:
        for multiplier in MULTIPLIERS:
            center=centers['Joint/0.1']['center']; tag=f'lr{round(100*multiplier):03d}'
            rows.append(dict(run_id=f'soft30_componentwise_g0100_{qtag}_{tag}_s42', role='Componentwise',
                clipping_map='auto_s', gamma=.1, C=1., lr=center*multiplier,
                lr_center=center, lr_multiplier=multiplier, q=q, beta=1, optimizer='SGD',private=True,
                source_checkpoint_sha256=SOURCE_SHA, seed_label=42, rng_coupling='independent_per_run',
                matched_joint_run_id=f'soft30_joint_g0100_{tag}_s42'))
    if len(rows)!=30 or len({r['run_id'] for r in rows})!=30:
        raise RuntimeError('invalid soft grid')
    return rows


def public_gpu_gate(model,data,output):
    """Actual physical128 gradients/query/release/SGD, public records only."""
    params=[p for p in model.parameters() if p.requires_grad]
    saved=[p.detach().clone() for p in params]
    ids=data.split['indices']['probe'][:128]
    x,y=data.batch(ids); x,y=x.cuda(),y.cuda()
    uniform=public_uniform(x,[data.split['records'][i]['image_id'] for i in ids],20260921,'trades')
    adv,_=attack(model,x,y,objective='kl',uniform=uniform)
    torch.cuda.reset_peak_memory_stats()
    values=gradients(model,x,y,trades_adv=adv,beta=1.)
    refs=gradients(model,x[:2],y[:2],trades_adv=adv[:2],beta=1.)
    for name in ['g','h']:
        torch.testing.assert_close(values[name][:2],refs[name],rtol=5e-4,atol=1e-5)
    checks=[]
    for role in ['CleanDP','Joint','Componentwise']:
        for gamma in GAMMAS:
            q=.25 if role=='Componentwise' else None
            u=query(values,role,gamma,q)
            if not torch.isfinite(u).all() or (u.norm(dim=-1)>1+2e-5).any():
                raise RuntimeError('AUTO-S public query failure')
            split=torch.cat([query({k:v[i:i+1] for k,v in values.items()},role,gamma,q) for i in range(2)])
            torch.testing.assert_close(u[:2],split)
            gen=torch.Generator(device='cuda').manual_seed(20260922)
            release=private_release(u.sum(0),sigma=1.22344970703125,bound=1.,generator=gen,denominator=512)
            opt=torch.optim.SGD(params,lr=.01,momentum=.9,nesterov=True,weight_decay=5e-4)
            assign_gradient(model,release); opt.step()
            if not all(torch.isfinite(p).all() for p in params):
                raise RuntimeError('AUTO-S public optimizer failure')
            with torch.no_grad():
                for p,v in zip(params,saved):p.copy_(v)
            checks.append(dict(role=role,gamma=gamma,q=q,max_query_norm=float(u.norm(dim=-1).max())))
    result=dict(status='PASS',physical_batch=128,logical_batch=512,public_only=True,
        checks=checks,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name())
    write_json(output,result)
    return result


def calibrate(refs,routes,data,target):
    target.mkdir(parents=True,exist_ok=False)
    centers={}
    for row in refs:
        path=get_final(row,routes[row['run_id']],Path('calibration_inputs'))
        model,architecture=load_final(path,row); model.cuda().eval()
        with diagnostic_rng():
            public_gpu_gate(model,data,target/(row['config']['role']+'_gpu_gate.json'))
            sums=None; count=0
            for values,_,_ in probe_batches(model,data,microbatch=2,seed=20260921):
                role=row['config']['role']
                raw=values['g'] if role=='CleanDP' else values['g']+values['h']
                updates=[clip(raw,1.)]+[query(values,role,gamma) for gamma in GAMMAS]
                batch_sums=torch.stack([u.double().sum(0) for u in updates])
                sums=batch_sums if sums is None else sums+batch_sums
                count+=len(raw)
            norms=(sums/count).norm(dim=-1).cpu().tolist()
        if count!=512 or not all(0<v<float('inf') for v in norms):
            raise RuntimeError('invalid public update norms')
        for i,gamma in enumerate(GAMMAS,1):
            raw_center=.05*norms[0]/(norms[i]+1e-12)
            centers[f'{role}/{gamma:g}']=dict(center=min(.2,max(.0125,raw_center)),raw_center=raw_center,
                hard_mean_norm=norms[0],autos_mean_norm=norms[i],reference_run_id=row['run_id'],
                reference_checkpoint_sha256=row['checkpoint_sha256'],gamma=gamma)
        del model,sums,updates,values,batch_sums
        torch.cuda.empty_cache()
    result=dict(status='PASS',centers=centers,runs=make_rows(centers),examples=512,
        split_hash=data.split['split_hash'],probe_hash=data.split['hashes']['probe'],
        calibration_references_not_final_selection=True,hard_reference_C=1.,hard_reference_lr=.05,
        public_only=True,no_dp_noise=True,official_test_loaded=False)
    write_json(target/'calibration.json',result)
    return result


def run(args):
    from trustml.convnext_gtsrb.common import code_identity
    code_identity()  # Fail before any download, calibration or GPU work.
    configure_gpu()
    if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('exactly one A100 required')
    payload=json.loads(args.routes.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    archive=args.input_archive or Path('training_inputs.tar')
    if not args.input_archive:fetch(payload['inputs']['training_url'],archive,payload['inputs']['training_sha256'])
    materialize(archive,payload['inputs']['training_sha256'])
    augmentation=json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data=GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),Path('gtsrb_inputs/splits.json'),augmentation)
    if data.split['hashes']['dev']!=DEV_HASH:raise RuntimeError('locked split mismatch')
    store=TransactionStore(payload['routes'],args.output/'transactions')
    if args.calibrate:
        calibration=calibrate(payload['references'],payload['references_routes'],data,args.output/'calibration')
        store.commit('calibration',args.output/'calibration')
    else:
        calibration=json.loads(Path('autos_calibration.json').read_text())
        if digest(calibration)!=payload['calibration_hash'] or calibration['status']!='PASS':
            raise RuntimeError('calibration identity mismatch')
    rows=calibration['runs']
    chosen=DS_RUNS if args.calibrate or args.chain_training else [args.run_id]
    if not set(chosen)<={r['run_id'] for r in rows}:raise RuntimeError('unknown training run')
    selected_archive=Path('public_selected.tar.gz')
    fetch(payload['inputs']['selected_url'],selected_archive,payload['inputs']['selected_archive_sha256'])
    selected=Path('gtsrb_inputs/public_pretrain_selected.pt')
    with tarfile.open(selected_archive) as tf:
        m=tf.getmember('public_pretrain_selected.pt')
        if not m.isfile():raise RuntimeError('invalid selected checkpoint member')
        with tf.extractfile(m) as source,selected.open('xb') as out:shutil.copyfileobj(source,out)
    if sha256(selected)!=SOURCE_SHA:raise RuntimeError('public pretrain mismatch')
    p=protocol();p['campaign']='ssg_gtsrb_autos30_v1'
    p['clipping_map']='auto_s';p['execution']['mlm_cpus']=4;p['execution']['mlm_memory_gib']=32
    old_cal=json.loads(Path('configs/ssg_gtsrb_rect54/calibration.json').read_text())
    privacy=json.loads(Path('configs/ssg_gtsrb_rect54/accounting.json').read_text())
    write_json(args.output/'protocol.json',p)
    completed=[]
    for rid in chosen:
        row=next(r for r in rows if r['run_id']==rid)
        with store.request('GET',payload['routes'][rid]['receipt_get']) as response:
            if response.status_code!=404:raise RuntimeError('existing/uncertain run receipt; reconcile before training')
        with diagnostic_rng():
            model,_=load_selected(Path('gtsrb_inputs'),selected,SOURCE_SHA);model.cuda().eval()
            public_gpu_gate(model,data,args.output/(rid+'_public_gpu_gate.json'))
            del model
            torch.cuda.empty_cache()
        result=train(row,p,data,old_cal,privacy,Path('gtsrb_inputs'),selected,secrets.token_bytes(32),args.output/rid)
        verify_completed_run(args.output/rid,row,result)
        store.commit(rid,args.output/rid)
        if args.mlm:
            from scripts.platform.mlm.publish_job_artifacts import publish_job_artifacts
            publish_job_artifacts([store.root/(rid+'.tar.gz'),store.root/(rid+'.COMMITTED.json')])
        completed.append(result);write_json(args.output/'completed.json',completed)
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,required=True)
    p.add_argument('--input-archive',type=Path)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--chain-training',action='store_true')
    p.add_argument('--mlm',action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--output',type=Path,default=Path('outputs/autos30'))
    a=p.parse_args()
    if not a.calibrate and not a.chain_training and not a.run_id:p.error('--calibrate, --chain-training or --run-id required')
    run(a)

"""Additional public-dev attacks, unioned with immutable standard-AA masks.

No training or selection. Never relabel this custom diagnostic as standard AA.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.common import code_identity
from trustml.convnext_gtsrb.data import GTSRBTrainOnly, digest
from trustml.convnext_gtsrb.evaluation import autoattack_inputs
from .real_dev_aa import DEV_HASH
from .night_aa import get_final, load_final
from .rectangular import fetch
from .persistence import TransactionStore

PROTOCOL = dict(norm='Linf',epsilon=8/255,examples=4020,precision='FP32',model_mode='eval',
    apgd_iterations=100,apgd_restarts=5,square_queries=10000,square_seeds=[42,43,44],
    attack_seed=42,official_test_loaded=False,union_with_previous_standard_aa=True)


def evaluate(model, data, baseline, output, checkpoint_sha256):
    from autoattack import AutoAttack
    ids=np.asarray(data.split['indices']['dev'])
    if len(ids)!=4020 or data.split['hashes']['dev']!=DEV_HASH:raise RuntimeError('ordered dev mismatch')
    if not np.array_equal(ids,np.asarray(baseline['sample_ids'])):raise RuntimeError('baseline IDs differ')
    model.eval()
    # Use exactly the existing AA boundary: dataset NHWC->NCHW views are not
    # contiguous, and pinned APGD checks gradient.view(N,-1). Values unchanged.
    x,y=autoattack_inputs(*data.batch(ids), 'cpu')
    if x.dtype!=torch.float32 or any(m.training for m in model.modules()):raise RuntimeError('FP32/eval required')
    clean=[]
    for i in range(0,len(ids),64):
        with torch.no_grad():z=model(x[i:i+64].cuda())
        if z.shape[1]!=43 or not torch.isfinite(z).all():raise RuntimeError('invalid clean logits')
        clean.extend(z.argmax(1).cpu().eq(y[i:i+64]).tolist())
    clean=np.asarray(clean,dtype=bool)
    if not np.array_equal(clean,np.asarray(baseline['clean_correct'],dtype=bool)):
        raise RuntimeError('clean masks changed')
    robust=np.asarray(baseline['aa_correct'],dtype=bool)
    if robust.shape!=clean.shape or np.any(robust & ~clean):raise RuntimeError('invalid baseline AA mask')
    robust=robust.copy();output.mkdir(parents=True,exist_ok=False)
    arrays=dict(sample_ids=ids,clean_correct=clean,standard_aa_correct=robust.copy())
    stages=[];start=time.monotonic()
    configurations=[('apgd-ce',42),('apgd-dlr',42)]+[('square',s) for s in PROTOCOL['square_seeds']]
    for name,seed in configurations:
        tag=f'{name}_s{seed}';stage_start=time.monotonic();maximum=0.
        adversary=AutoAttack(model,norm='Linf',eps=8/255,version='standard',seed=seed,
            device='cuda',log_path=str(output/(tag+'.log')),verbose=False)
        adversary.apgd.n_restarts=5;adversary.apgd.n_iter=100
        adversary.apgd.loss='ce' if name=='apgd-ce' else 'dlr';adversary.apgd.seed=seed
        adversary.square.n_queries=10000;adversary.square.seed=seed
        previous=robust.copy();survived=np.flatnonzero(previous)
        print(json.dumps(dict(phase='enhanced_stage_start',attack=name,seed=seed,
            survivors=len(survived),input_layout='contiguous NCHW')),flush=True)
        for i in range(0,len(survived),64):
            subset=survived[i:i+64];xb=x[subset].cuda();yb=y[subset].cuda()
            adv=(adversary.square.perturb(xb,yb) if name=='square' else adversary.apgd.perturb(xb,yb))
            linf=float((adv-xb).abs().max());maximum=max(maximum,linf)
            if (not torch.isfinite(adv).all() or adv.dtype!=torch.float32
                    or linf>8/255+1e-6 or adv.min()<0 or adv.max()>1):
                raise RuntimeError('adversarial finite/FP32/Linf/bounds failure')
            with torch.no_grad():z=model(adv)
            if not torch.isfinite(z).all():raise RuntimeError('nonfinite attacked logits')
            robust[subset] &= z.argmax(1).eq(yb).cpu().numpy()
            print(json.dumps(dict(phase='enhanced_batch_complete',attack=name,seed=seed,
                processed=i+len(subset),total=len(survived),seconds=time.monotonic()-stage_start)),flush=True)
        if np.any(robust & ~previous):raise RuntimeError('nonmonotone enhanced attack')
        arrays['after_'+tag]=robust.copy()
        stages.append(dict(attack=name,seed=seed,survivors=int(robust.sum()),
            newly_broken=int(previous.sum()-robust.sum()),seconds=time.monotonic()-stage_start,max_linf=maximum))
        np.savez_compressed(output/'predictions.npz',**arrays,enhanced_correct=robust)
        write_json(output/'stages.json',stages)
        print(json.dumps(dict(phase='enhanced_stage_complete',**stages[-1])),flush=True)
    result=dict(status='PASS',protocol=PROTOCOL,standard_aa=float(100*arrays['standard_aa_correct'].mean()),
        enhanced_union_accuracy=float(100*robust.mean()),clean=float(100*clean.mean()),
        additional_drop_pp=float(100*(arrays['standard_aa_correct'].mean()-robust.mean())),
        checkpoint_sha256=checkpoint_sha256,ordered_dev_hash=DEV_HASH,stages=stages,
        seconds=time.monotonic()-start,official_test_loaded=False,optimizer_steps_added=0)
    write_json(output/'metrics.json',result)
    return result


def run(args):
    from scripts.platform.gtsrb_inputs import materialize
    code_identity();configure_gpu()
    if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('one A100 required')
    plan=json.loads(Path('enhanced_manifest.json').read_text());payload=json.loads(args.routes.read_text())
    if payload['manifest_hash']!=digest(plan):raise RuntimeError('manifest mismatch')
    rows=plan['chains'][args.chain]
    archive=args.input_archive or Path('training_inputs.tar')
    if not args.input_archive:fetch(payload['inputs']['training_url'],archive,payload['inputs']['training_sha256'])
    materialize(archive,payload['inputs']['training_sha256'])
    aug=json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())['augmentation']
    data=GTSRBTrainOnly(Path('gtsrb_inputs/GTSRB_Final_Training_Images.zip'),Path('gtsrb_inputs/splits.json'),aug)
    args.output.mkdir(parents=True,exist_ok=False);store=TransactionStore(payload['routes'],args.output/'transactions')
    done=[]
    print(json.dumps(dict(phase='enhanced_chain_start',chain=args.chain,gpu=torch.cuda.get_device_name())),flush=True)
    for row in rows:
        rid=row['run_id'];route=payload['routes'][rid]
        with store.request('GET',route['receipt_get']) as response:
            if response.status_code==200:
                receipt=response.json()
                if receipt.get('status')!='COMMITTED' or receipt.get('stage')!=rid:raise RuntimeError('invalid receipt')
                done.append(dict(run_id=rid,status='ALREADY_COMMITTED'));continue
            if response.status_code!=404:raise RuntimeError('uncertain receipt; do not repeat')
        path=get_final(row,route,Path('evaluation_inputs'))
        model,_=load_final(path,row);model.cuda().eval()
        print(json.dumps(dict(phase='enhanced_child_start',run_id=rid)),flush=True)
        result=evaluate(model,data,row['baseline'],args.output/rid,row['checkpoint_sha256'])
        write_json(args.output/rid/'provenance.json',dict(config_hash=digest(row['config']),
            manifest_hash=digest(plan),source_archive_sha256=row['source_archive_sha256']))
        store.commit(rid,args.output/rid)
        if args.mlm:
            from scripts.platform.mlm.publish_job_artifacts import publish_job_artifacts
            try:
                publish_job_artifacts([store.root/(rid+'.tar.gz'),store.root/(rid+'.COMMITTED.json')])
            except Exception as exc:
                write_json(store.root/(rid+'.mlm_mirror_pending.json'),dict(
                    status='MLM_MIRROR_PENDING_S3_COMMITTED',error_class=type(exc).__name__))
        done.append(dict(run_id=rid,metrics=result));write_json(args.output/'completed.json',done)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes',type=Path,required=True);p.add_argument('--chain',required=True)
    p.add_argument('--input-archive',type=Path);p.add_argument('--mlm',action='store_true')
    p.add_argument('--output',type=Path,default=Path('outputs/enhanced'))
    run(p.parse_args())

"""CPU artifact verification. No model forward and no official-test access."""
import json
import math
from pathlib import Path

import numpy as np

from .data import digest
from .model import sha256


def validate_lr(step, observed, config):
    from .mechanism import lr_at
    expected = lr_at(step,config['lr'],total=config['steps'],warmup=config['warmup_steps'])
    # Linux and macOS libm cosine can differ in the last double-precision bit.
    # Four ULPs of peak LR is ~2.2e-19, not a scheduler-policy tolerance.
    if not math.isclose(observed, expected, rel_tol=0., abs_tol=4*math.ulp(config['lr'])):
        raise ValueError('actual LR trajectory mismatch')


def validate_geometry(folder, split, *, dimension=None):
    import pyarrow.parquet as pq
    folder=Path(folder)
    aggregate=json.loads((folder/'gtsrb_geometry_aggregate.json').read_text())
    identity=json.loads((folder/'gtsrb_geometry_manifest.json').read_text())
    expected=[split['records'][i] for i in split['indices']['probe']]
    if aggregate['probe_records']!=expected or aggregate['probe_hash']!=split['hashes']['probe']:
        raise ValueError('geometry probe/order mismatch')
    per=pq.read_table(folder/'gtsrb_geometry_per_record.parquet').to_pylist()
    layers=pq.read_table(folder/'gtsrb_geometry_layers.parquet').to_pylist()
    if len(per)!=512*18 or len(layers)!=512*6 or len(aggregate['settings'])!=18 or len(aggregate['blocks'])!=6:
        raise ValueError('incomplete geometry tables/grid')
    for key in aggregate['settings']:
        rows=sorted((r for r in per if r['setting']==key),key=lambda r:r['probe_order'])
        if [{k:r[k] for k in ['image_id','track_id','label']} for r in rows]!=expected:
            raise ValueError('setting changed ordered identities')
        if any(r['projection_slack'] < -r['projection_tolerance'] for r in rows):
            raise ValueError('projection inequality violated')
    G=np.asarray(aggregate['global']['gram'])
    n=len(aggregate['vector_names'])
    if G.shape!=(n,n) or not np.isfinite(G).all() or not np.allclose(G,G.T,rtol=1e-10,atol=1e-10):
        raise ValueError('invalid global Gram matrix')
    summed=np.zeros_like(G)
    for block, entry in aggregate['blocks'].items():
        rows=sorted((r for r in layers if r['block']==block),key=lambda r:r['probe_order'])
        if [{k:r[k] for k in ['image_id','track_id','label']} for r in rows]!=expected:
            raise ValueError('block changed ordered identities')
        B=np.asarray(entry['gram'])
        if B.shape!=G.shape or not np.isfinite(B).all():
            raise ValueError('invalid block Gram matrix')
        summed+=B
    np.testing.assert_allclose(summed,G,rtol=1e-8,atol=1e-8)
    if aggregate['private_gradients_logged'] or aggregate['raw_parameter_gradients_persisted']:
        raise ValueError('forbidden raw gradient persistence')
    return {'status':'PASS','records':512,'settings':18,'blocks':6,'vector_count':n,'snapshot_hash':identity['snapshot_hash']}


def validate_evaluation(folder, split, *, final):
    folder=Path(folder)
    with np.load(folder/'public_dev_predictions.npz',allow_pickle=False) as a:
        if not np.array_equal(a['dev_ids'],split['indices']['dev']) or not np.array_equal(a['trajectory_ids'],split['indices']['trajectory']):
            raise ValueError('evaluation ordering changed')
        if a['clean_correct'].shape!=(len(split['indices']['dev']),) or a['pgd20_correct'].shape!=(1024,):
            raise ValueError('evaluation shape mismatch')
        if not np.isfinite(a['clean_ce']).all() or np.any(a['clean_ce']<0):
            raise ValueError('invalid public CE')
        if np.any(a['pgd20_correct'] & ~a['trajectory_clean_correct']):
            raise ValueError('PGD monotonicity failed')
        metrics={'dev_clean':float(100*a['clean_correct'].mean()),'dev_ce_loss':float(a['clean_ce'].mean()),
                 'pgd20':float(100*a['pgd20_correct'].mean())}
        if final:
            survived=a['trajectory_clean_correct'].copy()
            for name in ['apgd-ce','apgd-t']:
                survived &= a[name+'_correct']
                if not np.array_equal(survived,a['after_'+name]):
                    raise ValueError('APGD stage mask mismatch')
                metrics[name]=float(100*a[name+'_correct'].mean())
            if not np.array_equal(survived,a['apgd_union_correct']):
                raise ValueError('APGD union mismatch')
            metrics['apgd_union']=float(100*survived.mean())
    saved=json.loads((folder/'metrics.json').read_text())
    for k,v in metrics.items():
        if abs(v-saved[k])>1e-8:
            raise ValueError('saved evaluation metric mismatch: '+k)
    return metrics


def validate_run(folder, split, expected_hash):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    final=json.loads((folder/'summary.json').read_text())
    if digest(config)!=expected_hash or final['config_hash']!=expected_hash or final['status']!='PASS':
        raise ValueError('run config/summary mismatch')
    if final['optimizer_steps']!=config['steps'] or sha256(folder/'checkpoints/final.pt')!=final['checkpoint_sha256']:
        raise ValueError('final iterate mismatch')
    expected=config['steps'] if config['private'] else 0
    history=json.loads((folder/'accountant_history.json').read_text())
    if final['release_count']!=expected or history['release_count']!=expected or history['rdp']!=history['prv'] or sum(r[2] for r in history['rdp'])!=expected:
        raise ValueError('actual accountant release history mismatch')
    updates=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [r['step'] for r in updates]!=list(range(1,config['steps']+1)):
        raise ValueError('missing optimizer steps')
    for row in updates:
        validate_lr(row['step'],row['lr'],config)
    checks={}
    for epoch in config['checkpoint_epochs']:
        checks[str(epoch)]={'evaluation':validate_evaluation(folder/f'evaluation_epoch{epoch:03d}',split,final=epoch==50)}
        if epoch in config['geometry_epochs']:
            checks[str(epoch)]['geometry']=validate_geometry(folder/f'geometry_epoch{epoch:03d}',split)
    return {'status':'PASS','config_hash':expected_hash,'checkpoints':checks,'actual_optimizer_steps':len(updates)}

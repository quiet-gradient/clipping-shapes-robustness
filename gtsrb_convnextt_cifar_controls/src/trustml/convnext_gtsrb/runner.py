"""Exactly nine predeclared roles, 50 expected epochs, immediate public audits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import secrets
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from .accounting import bounds, horizon
from .attacks import attack, public_uniform
from .common import code_identity, diagnostic_rng, provenance, tensor_digest
from .data import GTSRBTrainOnly, digest
from .evaluation import evaluate
from .mechanism import gradients, contribution, assign_gradient, private_release, lr_at
from .model import build_model, sha256
from .probe import audit_geometry
from trustml.convnext_cifar100.preflight import configure_gpu, write_json


def validate_config(config):
    protocol = json.loads(Path('configs/convnext_gtsrb/protocol.json').read_text())
    for key, value in protocol.items():
        if key != 'expected_epochs' and config.get(key) != value:
            raise ValueError('locked pilot protocol changed: '+key)
    for key, value in horizon(config['N_private']).items():
        if config[key] != value:
            raise ValueError('GTSRB accounting horizon mismatch: '+key)
    if config['role'] not in ['Natural','RobustOnly','DPOnly','DPPGD','Joint','Componentwise']:
        raise ValueError('unrequested scientific role')
    if config['private'] != (config['role'] not in ['Natural','RobustOnly']):
        raise ValueError('incorrect privacy role')
    if config['private'] and (config['C'] <= 0 or config['sigma'] <= 0):
        raise ValueError('invalid private bound/noise')
    if config['role']=='Componentwise' and config['q'] not in [.1,.25]:
        raise ValueError('not a pilot allocation')
    if config['source_metadata']['source_kind'] != 'clean_imagenet_convstem':
        raise ValueError('adversarial source forbidden')


def save_checkpoint(path, model, optimizer, config, step, *, final=False):
    state = {'config':config,'config_hash':digest(config),'optimizer_steps':step,
             'release_count':step if config['private'] else 0,'source_checkpoint_hash':config['source_metadata']['sha256']}
    state['trainable'] = {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    if final:
        state['model'] = {n:p.detach().cpu() for n,p in model.state_dict().items()}
        state['optimizer'] = optimizer.state_dict()
    if path.exists():
        raise FileExistsError('never overwrite a checkpoint')
    torch.save(state,path)


def secret_streams(private):
    # A public fixed Gaussian seed would reveal the DP noise. Persist commitments,
    # not reconstructible sampler/noise states; this is not a CSPRNG certification.
    seeds = {k:(secrets.randbits(63) if private else 42+i) for i,k in enumerate(['sampler','attack','noise','augmentation'])}
    torch.manual_seed(seeds['augmentation'])
    random.seed(seeds['augmentation'])
    np.random.seed(seeds['augmentation'] % 2**32)
    generators = {k:torch.Generator(device='cpu' if k=='sampler' else 'cuda').manual_seed(seeds[k])
                  for k in ['sampler','attack','noise']}
    audit = {'public_model_seed':42,'stream_commitments':{k:hashlib.sha256(str(v).encode()).hexdigest() for k,v in seeds.items()},
             'private_rng_seeds_published':False,'noise_generator':'torch.Generator, not CSPRNG-certified',
             'exact_cross_arm_randomness_coupling':False,'diagnostics_restore_training_rng':True}
    if not private:
        audit['nonprivate_seeds']=seeds
    return generators,audit


def public_training_benchmark(model,data,config,output):
    """No optimizer updates: validate/timing at the actual physical batch size."""
    rows=data.split['indices']['probe'][:64]
    elapsed=0.
    checked=False
    for begin in range(0,len(rows),config['training_microbatch']):
        ids=rows[begin:begin+config['training_microbatch']]
        x,y=data.batch(ids)
        x,y=x.cuda(),y.cuda()
        torch.cuda.synchronize()
        started=time.monotonic()
        adv=None
        if config['role'] not in ['Natural','DPOnly']:
            uniform=public_uniform(x,[data.split['records'][i]['image_id'] for i in ids],20260921,'timing')
            adv,_=attack(model,x,y,objective='ce' if config['role']=='DPPGD' else 'kl',uniform=uniform)
        if config['private']:
            kwargs={'trades_adv':adv if config['role'] in ['Joint','Componentwise'] else None,
                    'ce_adv':adv if config['role']=='DPPGD' else None,'need_g':config['role']!='DPPGD'}
            values=gradients(model,x,y,**kwargs)
            u=contribution(values,config['role'],config['C'],config['q'])
            torch.cuda.synchronize()
            elapsed+=time.monotonic()-started
            if not checked:
                small={k:(v[:2] if isinstance(v,torch.Tensor) else v) for k,v in kwargs.items()}
                refs=gradients(model,x[:2],y[:2],**small)
                for k in values:
                    torch.testing.assert_close(values[k][:2],refs[k],rtol=5e-4,atol=1e-5)
                checked=True
            if not torch.isfinite(u).all() or (u.norm(dim=-1)>config['C']*(1+2e-5)).any():
                raise RuntimeError('public actual-batch bound validation failed')
        else:
            model.zero_grad(set_to_none=True)
            z=model(x)
            loss=F.cross_entropy(z,y,reduction='sum')
            if adv is not None:
                loss=loss+F.kl_div(model(adv).log_softmax(-1),z.softmax(-1),reduction='sum')
            (loss/512).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError('nonprivate public gradient check failed')
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed+=time.monotonic()-started
    report={'public_records':64,'optimizer_updates':0,'physical_batch':config['training_microbatch'],
            'gradient_equivalence_checked':checked,'elapsed_compute_seconds':elapsed,
            'estimated_training_hours_excluding_augmentation_and_optimizer':elapsed/64*512*config['steps']/3600,
            'note':'public timing proxy, not a training result or an end-to-end ETA'}
    write_json(output/'public_runtime_audit.json',report)
    print(json.dumps({'stage':'public_runtime_audit','run_id':config['run_id'],**report}),flush=True)


def train(config_path, assets_root, output_root):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    code = code_identity()
    device = configure_gpu()
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('A100 required')
    root = Path(assets_root)
    output = Path(output_root)/config['run_id']
    output.mkdir(parents=True,exist_ok=False)
    snapshots = output/'checkpoints'
    snapshots.mkdir()
    started = time.monotonic()
    write_json(output/'config.json',config)
    write_json(output/'code_identity.json',code)
    environment = {'packages':{k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','numpy','scipy','opacus','wandb','pyarrow','Pillow']},
                   'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'FP32':True,'TF32':False}
    write_json(output/'environment.json',environment)
    if sha256(root/'GTSRB_Final_Training_Images.zip') != config['data_sha256']:
        raise ValueError('dataset changed')
    data = GTSRBTrainOnly(root/'GTSRB_Final_Training_Images.zip',root/'splits.json',config['augmentation'])
    if data.split['split_hash']!=config['split_hash'] or data.split['hashes']['probe']!=config['probe_hash']:
        raise ValueError('split/probe changed')
    if len(data.split['indices']['private']) != config['N_private']:
        raise ValueError('private dataset cardinality changed')
    model,architecture = build_model(root/'convnext_tiny_cvst_clean.pt',config['source_metadata'],seed=config['seed'])
    if tensor_digest({n:p for n,p in model.named_parameters() if p.requires_grad}) != config['preflight_initial_trainable_digest']:
        raise ValueError('initial model differs from public calibration')
    model.to(device).eval()
    write_json(output/'architecture.json',architecture)
    privacy = bounds(config['sigma'],config['N_private']) if config['private'] else {'private':False,'status':'NOT_PRIVATE'}
    if config['private']:
        if privacy['status']!='PASS':
            raise RuntimeError('privacy validation failed')
        privacy.update(sensitivity_bound=config['C'],absolute_noise_std_before_averaging=config['sigma']*config['C'])
    write_json(output/'gtsrb_privacy_accounting.json',privacy)
    streams,rng_audit = secret_streams(config['private'])
    write_json(output/'randomness_audit.json',rng_audit)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=config['lr'],betas=tuple(config['adam_betas']),eps=config['adam_eps'],weight_decay=config['weight_decay'])
    from opacus.accountants import RDPAccountant,PRVAccountant
    rdp,prv = RDPAccountant(),PRVAccountant()
    import wandb
    wb = wandb.init(project='convnext-gtsrb-pilot',name=config['run_id'],mode='offline',dir=str(output),config=config,reinit=True)
    milestones = {int(s):int(e) for e,s in config['checkpoint_steps'].items()}
    curves,compute = [],[]

    def diagnose(step, lr):
        epoch = milestones[step]
        checkpoint = snapshots/('final.pt' if epoch==50 else f'epoch{epoch:03d}.pt')
        save_checkpoint(checkpoint,model,optimizer,config,step,final=epoch==50)
        actual_epoch=step*config['sample_rate']
        identity=provenance(config,data.split,architecture,checkpoint,step=step,epoch=actual_epoch,lr=lr,code=code)
        identity['nominal_expected_epoch']=epoch
        if config['private']:
            identity['epsilon_rdp_at_snapshot'] = rdp.get_epsilon(delta=config['delta'],alphas=privacy['orders']) if step else 0.
        with diagnostic_rng():
            t=time.monotonic()
            metrics=evaluate(model,data,output/f'evaluation_epoch{epoch:03d}',seed=config['attack_seed'],final=epoch==50)
            eval_attack={'epsilon':config['attack_epsilon'],'pgd_steps':20,'pgd_restarts':1,'seed':config['attack_seed'],
                         'final_attacks':config['final_attacks'] if epoch==50 else [],'subset_size':1024,'precision':'FP32'}
            write_json(output/f'evaluation_epoch{epoch:03d}'/'provenance.json',
                {**identity,'attack_seed':config['attack_seed'],'attack_config':eval_attack,'attack_hash':digest(eval_attack)})
            compute.append({'phase':'evaluation','epoch':epoch,'seconds':time.monotonic()-t})
            row={'epoch':epoch,'expected_epoch':actual_epoch,'step':step,'LR':lr,'checkpoint_sha256':sha256(checkpoint),**metrics}
            curves.append(row)
            write_json(output/'gtsrb_pilot_learning_curves.json',curves)
            wb.log({k:v for k,v in row.items() if isinstance(v,(int,float))},step=step)
            print(json.dumps({'stage':'evaluation','run_id':config['run_id'],**row}),flush=True)
            if epoch in config['geometry_epochs']:
                t=time.monotonic()
                audit_geometry(model,data,architecture,config['calibration'],config,output/f'geometry_epoch{epoch:03d}',identity,microbatch=config['microbatch'])
                compute.append({'phase':'geometry','epoch':epoch,'seconds':time.monotonic()-t})
        write_json(output/'gtsrb_compute_ledger.json',compute)
        torch.cuda.empty_cache()

    ids = np.asarray(data.split['indices']['private'])
    history = output/'history.jsonl'
    training_seconds=0.
    try:
        with diagnostic_rng():
            public_training_benchmark(model,data,config,output)
        diagnose(0,0.)
        for step in range(1,config['steps']+1):
            t=time.monotonic()
            selected=ids[torch.nonzero(torch.rand(config['N_private'],generator=streams['sampler'])<config['sample_rate']).flatten().numpy()]
            optimizer.zero_grad(set_to_none=True)
            summed=torch.zeros(architecture['trainable_parameter_count'],device=device) if config['private'] else None
            for begin in range(0,len(selected),config['training_microbatch']):
                x,y=data.batch(selected[begin:begin+config['training_microbatch']],augment=True)
                x,y=x.to(device),y.to(device)
                adv=None
                if config['role'] not in ['Natural','DPOnly']:
                    objective='ce' if config['role']=='DPPGD' else 'kl'
                    uniform=torch.rand(x.shape,device=device,generator=streams['attack'])
                    adv,_=attack(model,x,y,objective=objective,uniform=uniform)
                if config['private']:
                    values=gradients(model,x,y,trades_adv=adv if config['role'] in ['Joint','Componentwise'] else None,
                        ce_adv=adv if config['role']=='DPPGD' else None,beta=config['beta'],need_g=config['role']!='DPPGD')
                    u=contribution(values,config['role'],config['C'],config['q'])
                    if not torch.isfinite(u).all() or (u.norm(dim=-1)>config['C']*(1+2e-5)).any():
                        raise RuntimeError('scientific failure: contribution bound/nonfinite')
                    summed.add_(u.sum(0))
                    del values,u
                else:
                    z=model(x)
                    loss=F.cross_entropy(z,y,reduction='sum')
                    if config['role']=='RobustOnly':
                        loss=loss+config['beta']*F.kl_div(model(adv).log_softmax(-1),z.softmax(-1),reduction='sum')
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite nonprivate objective')
                    (loss/config['batch_denominator']).backward()
            if config['private']:
                update=private_release(summed,sigma=config['sigma'],bound=config['C'],generator=streams['noise'],denominator=config['batch_denominator'])
                assign_gradient(model,update)
            elif len(selected)==0:
                for p in params:
                    p.grad=torch.zeros_like(p)
            lr=lr_at(step,config['lr'],total=config['steps'],warmup=config['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr']=lr
            optimizer.step()
            if config['private']:
                rdp.step(noise_multiplier=config['sigma'],sample_rate=config['sample_rate'])
                prv.step(noise_multiplier=config['sigma'],sample_rate=config['sample_rate'])
            if not all(torch.isfinite(p).all() for p in params):
                raise RuntimeError('scientific failure: nonfinite model parameters')
            duration=time.monotonic()-t
            training_seconds+=duration
            row={'step':step,'lr':lr,'expected_epoch':step*config['sample_rate'],'step_seconds':duration,
                 'release_count':step if config['private'] else 0,'elapsed_seconds':time.monotonic()-started}
            with history.open('a') as f:
                f.write(json.dumps(row)+'\n')
            if step==1 or step%20==0:
                wb.log(row,step=step)
                print(json.dumps({'run_id':config['run_id'],'stage':'training',**row}),flush=True)
            if step in milestones:
                diagnose(step,lr)
        expected=config['steps'] if config['private'] else 0
        if sum(v[2] for v in rdp.history)!=expected or rdp.history!=prv.history:
            raise RuntimeError('accountant release count mismatch')
        if config['private']:
            privacy['achieved_rdp_epsilon_from_history']=rdp.get_epsilon(delta=config['delta'],alphas=privacy['orders'])
            privacy['achieved_prv_epsilon_from_history']=prv.get_epsilon(delta=config['delta'],eps_error=.1,delta_error=1e-8)
            if max(privacy['achieved_rdp_epsilon_from_history'],privacy['achieved_prv_epsilon_from_history'])>4:
                raise RuntimeError('achieved privacy exceeds epsilon=4')
            write_json(output/'gtsrb_privacy_accounting.json',privacy)
        write_json(output/'accountant_history.json',{'rdp':rdp.history,'prv':prv.history,'release_count':expected})
        compute.append({'phase':'training','seconds':training_seconds})
        write_json(output/'gtsrb_compute_ledger.json',compute)
        final={'status':'PASS','run_id':config['run_id'],'role':config['role'],'config_hash':digest(config),
            'optimizer_steps':config['steps'],'release_count':expected,'checkpoint_sha256':sha256(snapshots/'final.pt'),
            'privacy':privacy,'dev':curves[-1],'wandb_run_id':wb.id,'git_commit':code['git_commit'],
            'elapsed_seconds':time.monotonic()-started,'gpu':environment['gpu'],'official_test_loaded':False,
            'checkpoint_epochs':[r['epoch'] for r in curves],'geometry_epochs':config['geometry_epochs'],
            'final_iterate':'epoch50, no selection of intermediate checkpoint'}
        write_json(output/'summary.json',final)
    finally:
        wb.finish()
    return output


def archive_result(output):
    output=Path(output)
    final=json.loads((output/'summary.json').read_text())
    config=json.loads((output/'config.json').read_text())
    if final['status']!='PASS' or final['optimizer_steps']!=config['steps'] or final['checkpoint_epochs']!=config['checkpoint_epochs']:
        raise ValueError('incomplete pilot run')
    for epoch in config['geometry_epochs']:
        p=output/f'geometry_epoch{epoch:03d}'
        if not all((p/name).is_file() for name in ['gtsrb_geometry_per_record.parquet','gtsrb_geometry_layers.parquet','gtsrb_geometry_aggregate.json','gtsrb_geometry_manifest.json']):
            raise ValueError('missing geometry artifact')
    write_json(output/'artifact_manifest.json',{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    archive=output.parent/(output.name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:
        tar.add(output,arcname=output.name)
    receipt={'status':'COMMITTED','run_id':output.name,'config_hash':digest(config),'checkpoint_sha256':final['checkpoint_sha256'],
        'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,
        'persistence':'local completed archive; remote publish and download verification required'}
    write_json(output.parent/(output.name+'.COMMITTED.json'),receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--assets-root',default='gtsrb_inputs')
    p.add_argument('--output',default='outputs/gtsrb_pilot')
    a=p.parse_args()
    manifest=json.loads(Path('configs/convnext_gtsrb/gtsrb_pilot_manifest.json').read_text())
    rows=[r for r in manifest['runs'] if r['run_id']==a.run_id]
    if len(manifest['runs'])!=9 or len(rows)!=1 or manifest['full_sweep_authorized']:
        raise ValueError('only the nine declared pilot runs are authorized')
    row=rows[0]
    if digest(json.loads(Path(row['config_path']).read_text()))!=row['config_hash']:
        raise ValueError('config hash mismatch')
    output=train(row['config_path'],a.assets_root,a.output)
    print(json.dumps(archive_result(output)),flush=True)

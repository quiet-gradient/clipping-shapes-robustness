"""GTSRB fixed-norm N control with native full-KL gradients and exact resume."""
from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .io import sha,write_json
from .protected_state import private_root,write_private_snapshot,read_private_snapshot,assert_state_equal
from trustml.convnext_gtsrb.accounting import horizon
from trustml.convnext_gtsrb.attacks import attack,public_uniform
from trustml.convnext_gtsrb.common import diagnostic_rng
from trustml.convnext_gtsrb.data import GTSRBTrainOnly,digest
from trustml.convnext_gtsrb.mechanism import gradients,contribution as hard_query,private_release,assign_gradient,lr_at
from trustml.ssg_gtsrb.autos_map import query as soft_query
from trustml.ssg_gtsrb.private import matched_streams,stream_positions,training_microbatch
from .inputs import load_selected
from trustml.ssg_gtsrb.rectangular import SOURCE_SHA
from trustml.ssg_gtsrb.telemetry import optimizer_step

def validate_row(row,privacy,protocol):
    if row.get('intervention')!='N' or row.get('role')!='Componentwise' or row.get('outer_kl_target')!='full':raise ValueError('N requires native GTSRB full outer KL')
    soft=row.get('clipping_map')=='auto_s'
    if row.get('clipping_map') not in ('hard','auto_s') or (row['C'],row['beta'],row['optimizer'],row['lr'],row['q'])!=(1.,1,'SGD',.04304104621706269 if soft else .025,.25 if soft else 1/3):raise ValueError('GTSRB frozen anchor mismatch')
    if soft and row.get('gamma')!=.1:raise ValueError('AUTO-S gamma mismatch')
    if row['source_checkpoint_sha256']!=SOURCE_SHA or row['seed_label'] not in (45,46,47):raise ValueError('public initialization or seed differs')
    if privacy['status']!='PASS' or privacy['steps']!=3436 or privacy['noise_multiplier']!=1.22344970703125 or max(privacy['rdp_epsilon_upper'],privacy['prv_epsilon_upper'])>4:raise ValueError('native accounting differs')
    return training_microbatch(protocol,row)

def queries(values,row):
    return (soft_query(values,'Joint',row['gamma'],row['q']),soft_query(values,'Componentwise',row['gamma'],row['q'])) if row['clipping_map']=='auto_s' else (hard_query(values,'Joint',1.,row['q']),hard_query(values,'Componentwise',1.,row['q']))

def n_query(values,row):
    u=soft_query(values,'Componentwise',row['gamma'],row['q']) if row['clipping_map']=='auto_s' else hard_query(values,'Componentwise',1.,row['q'])
    norm=u.norm(dim=-1,keepdim=True)
    u.mul_(1/torch.where(norm>0,norm,torch.ones_like(norm)))
    if not torch.isfinite(u).all() or (u.norm(dim=-1)>1+2e-5).any():raise RuntimeError('N sensitivity/finite invariant failed')
    return u

def snapshot(model,optimizer,rdp,prv,streams,secret,step,identity,history):
    return dict(version='gtsrb_N_resume_v1',step=step,identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rdp=rdp.state_dict(),prv=prv.state_dict(),streams={k:g.get_state() for k,g in streams.items()},master_key=secret,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],history=history)

def restore(saved,model,optimizer,rdp,prv,streams,identity):
    if saved['version']!='gtsrb_N_resume_v1' or saved['identity']!=identity:raise RuntimeError('protected GTSRB state identity differs')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);rdp.load_state_dict(saved['rdp']);prv.load_state_dict(saved['prv'])
    if set(saved['streams'])!=set(streams):raise RuntimeError('protected GTSRB stream names differ')
    for k,g in streams.items():g.set_state(saved['streams'][k])
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])

def public_checkpoint(model,row,architecture,step,output):
    path=output/'checkpoints'/('final.pt' if step==3436 else f'step_{step:04d}.pt');path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},scientific_config=row,source_checkpoint_sha256=SOURCE_SHA,optimizer_steps=step,release_count=step,bounded_query_steps=step,trainable_mask_hash=architecture['trainable_mask_hash'],private_rng_in_artifact=False)
    if path.exists():assert_state_equal(torch.load(path,map_location='cpu',weights_only=False),state)
    else:
        temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)
    return path

def public_gate(model,data,row,output,microbatch):
    with diagnostic_rng():
        ids=data.split['indices']['probe'][:microbatch];x,y=data.batch(ids);x,y=x.cuda(),y.cuda()
        uniform=public_uniform(x,[data.split['records'][i]['image_id'] for i in ids],20260921,'trades')
        adv,_=attack(model,x,y,objective='kl',uniform=uniform)
        values=gradients(model,x,y,trades_adv=adv,beta=1.,outer_kl_target='full');n=n_query(values,row)
        one=gradients(model,x[:1],y[:1],trades_adv=adv[:1],beta=1.,outer_kl_target='full');torch.testing.assert_close(n[:1],n_query(one,row),rtol=5e-4,atol=1e-5)
        params=[p for p in model.parameters() if p.requires_grad]
        logits=model(x[:1]);log_adv=model(adv[:1]).log_softmax(-1)
        loss=F.cross_entropy(logits,y[:1])+F.kl_div(log_adv,logits.softmax(-1),reduction='sum')
        expected=torch.cat([g.flatten() for g in torch.autograd.grad(loss,params)])
        torch.testing.assert_close(one['g'][0]+one['h'][0],expected,rtol=5e-4,atol=2e-5)
    write_json(output/'public_startup.json',dict(status='PASS',public_only=True,private_updates=0,physical_batch=microbatch,outer_kl_target='full',gradient_equivalence='CE + full outer KL',fp32=True))

def public_probe(model,data,row,step,output):
    with diagnostic_rng():
        ids=data.split['indices']['probe'];sums={k:torch.zeros(15505195,device='cuda',dtype=torch.float64) for k in ('J','U','N','M','D','R')};norms={k:[] for k in sums};gn=[];hn=[]
        for begin in range(0,len(ids),2):
            selected=ids[begin:begin+2];x,y=data.batch(selected);x,y=x.cuda(),y.cuda()
            uniform=public_uniform(x,[data.split['records'][i]['image_id'] for i in selected],20260921,'trades')
            adv,_=attack(model,x,y,objective='kl',uniform=uniform)
            v=gradients(model,x,y,trades_adv=adv,beta=1.,outer_kl_target='full',stable_public_kl=True)
            j,u=queries(v,row);nj=j.norm(dim=-1,keepdim=True);nu=u.norm(dim=-1,keepdim=True)
            denom=v['g'].norm(dim=-1,keepdim=True)+v['h'].norm(dim=-1,keepdim=True)
            r=(v['g']+v['h'])/(denom+row['gamma']) if row['clipping_map']=='auto_s' else (v['g']+v['h'])*(1/denom.clamp_min(torch.finfo(denom.dtype).tiny)).clamp(max=1)
            controls=dict(J=j,U=u,N=u/torch.where(nu>0,nu,torch.ones_like(nu)),M=u*(nj/torch.where(nu>0,nu,torch.ones_like(nu))),D=j*(nu/torch.where(nj>0,nj,torch.ones_like(nj))),R=r)
            for k,value in controls.items():sums[k].add_(value.double().sum(0));norms[k].extend(value.norm(dim=-1).cpu().tolist())
            gn.extend(v['g'].norm(dim=-1).cpu().tolist());hn.extend(v['h'].norm(dim=-1).cpu().tolist())
        folder=output/'geometry';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'step_{step:04d}.npz',sample_ids=np.asarray(ids),g_norm=np.asarray(gn),h_norm=np.asarray(hn),**{k+'_norm':np.asarray(value) for k,value in norms.items()})
        report=dict(step=step,sample_count=len(ids),probe_hash=data.split['hashes']['probe'],split_hash=data.split['split_hash'],public_only=True,outer_kl_target='full',per_record_precision='FP32',diagnostic_sum_precision='FP64',aggregate_norms={k:float((v/len(ids)).norm()) for k,v in sums.items()},mean_individual_norms={k:float(np.mean(v)) for k,v in norms.items()})
        write_json(folder/f'step_{step:04d}.json',report)
    return report

def train(row,protocol,privacy,data,output,protected,*,stop_after=None,backup=None):
    microbatch=validate_row(row,privacy,protocol)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name():raise RuntimeError('one A100 required; no local fallback')
    output=Path(output);protected=Path(protected);output.mkdir(parents=True,exist_ok=True)
    h=horizon(len(data.split['indices']['private']))
    if h['steps']!=3436 or h['sample_rate']!=privacy['sample_rate']:raise RuntimeError('GTSRB horizon differs')
    model,architecture=load_selected(Path('gtsrb_inputs'),Path('gtsrb_inputs/public_pretrain_selected.pt'),SOURCE_SHA);model.cuda().float().eval()
    params=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=15505195:raise RuntimeError('GTSRB trainable mask differs')
    optimizer=torch.optim.SGD(params,lr=row['lr'],momentum=.9,nesterov=True,weight_decay=5e-4)
    from opacus.accountants import RDPAccountant,PRVAccountant
    rdp,prv=RDPAccountant(),PRVAccountant()
    secret=private_root(protected/'root.key');streams,rng=matched_streams(secret,row)
    identity=dict(scientific_fingerprint=digest(row),runtime_sha256=sha(__file__),source_checkpoint_sha256=SOURCE_SHA,split_hash=data.split['split_hash'],gpu=torch.cuda.get_device_name(),cuda_version=torch.version.cuda)
    history=[];step=0
    if (protected/'latest.json').exists():
        saved,_=read_private_snapshot(protected)
        if saved['master_key']!=secret:raise RuntimeError('resume secret changed')
        restore(saved,model,optimizer,rdp,prv,streams,identity);step=saved['step'];history=saved['history']
        assert_state_equal(saved,snapshot(model,optimizer,rdp,prv,streams,secret,step,identity,history))
        del saved
        if step==100 and not (output/'pilot.json').exists():
            public_checkpoint(model,row,architecture,100,output)
            write_json(output/'pilot.json',dict(status='PASS',prefix_updates=100,full_schedule_updates=3436,full_state_roundtrip='bitwise_equal',continued_same_run=True,outer_kl_target='full',physical_batch=microbatch,verified_on_resume_at_step=100,gpu_name=torch.cuda.get_device_name(),visible_gpu_count=1))
    else:
        if (output/'pilot.json').exists() or (output/'checkpoints').exists():raise RuntimeError('public progress exists without protected state')
        write_json(output/'scientific_config.json',row);write_json(output/'architecture.json',architecture);write_json(output/'planned_accounting.json',privacy)
        public_gate(model,data,row,output,microbatch)
        checkpoint=public_checkpoint(model,row,architecture,0,output)
        history.append(public_probe(model,data,row,0,output))
        write_private_snapshot(protected,snapshot(model,optimizer,rdp,prv,streams,secret,0,identity,history))
        if backup:
            with diagnostic_rng():backup(protected,0)
    limit=3436 if stop_after is None else stop_after
    if limit not in (100,3436) or step>limit:raise ValueError('only full-schedule prefix100 or continuation to3436')
    initial_step=step;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    private_ids=np.asarray(data.split['indices']['private']);milestones={v for k,v in h['checkpoint_steps'].items() if int(k) in (10,20,30,40,50)}
    for step in range(step+1,limit+1):
        selected=private_ids[torch.nonzero(torch.rand(h['N_private'],generator=streams['sampler'])<h['sample_rate']).flatten().numpy()]
        summed=torch.zeros(15505195,device='cuda');optimizer.zero_grad(set_to_none=True)
        for begin in range(0,len(selected),microbatch):
            x,y=data.batch(selected[begin:begin+microbatch],augment=True);x,y=x.cuda(),y.cuda()
            uniform=torch.rand(x.shape,device='cuda',generator=streams['attack']);adv,_=attack(model,x,y,objective='kl',uniform=uniform)
            values=gradients(model,x,y,trades_adv=adv,beta=1.,outer_kl_target='full');update=n_query(values,row);summed.add_(update.sum(0));del values,update
        sanitized=private_release(summed,sigma=privacy['noise_multiplier'],bound=1.,generator=streams['noise'],denominator=512);assign_gradient(model,sanitized)
        optimizer.param_groups[0]['lr']=lr_at(step,row['lr'],total=3436,warmup=h['warmup_steps'])
        scalars=optimizer_step(optimizer,params);rdp.step(noise_multiplier=privacy['noise_multiplier'],sample_rate=h['sample_rate']);prv.step(noise_multiplier=privacy['noise_multiplier'],sample_rate=h['sample_rate'])
        if any(not torch.isfinite(p).all() for p in params):raise RuntimeError('nonfinite trained parameter')
        with (output/'optimizer_telemetry.jsonl').open('a') as log:log.write(json.dumps(dict(step=step,gradient_provenance='gaussian_sanitized',**scalars))+'\n')
        if step in milestones:
            public_checkpoint(model,row,architecture,step,output);history.append(public_probe(model,data,row,step,output))
            from trustml.convnext_gtsrb.evaluation import evaluate
            with diagnostic_rng():evaluate(model,data,output/f'evaluation_step{step:04d}',final=False)
        save=(step%100==0 or step%344==0 or step==limit)
        if save:
            current=snapshot(model,optimizer,rdp,prv,streams,secret,step,identity,history);write_private_snapshot(protected,current)
            if step==100:
                saved,_=read_private_snapshot(protected);assert_state_equal(saved,current);restore(saved,model,optimizer,rdp,prv,streams,identity);assert_state_equal(saved,snapshot(model,optimizer,rdp,prv,streams,secret,step,identity,history));del saved
                public_checkpoint(model,row,architecture,step,output)
                write_json(output/'pilot.json',dict(status='PASS',prefix_updates=100,full_schedule_updates=3436,full_state_roundtrip='bitwise_equal',continued_same_run=True,outer_kl_target='full',physical_batch=microbatch,gpu_name=torch.cuda.get_device_name(),visible_gpu_count=1,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),prefix_wall_seconds=time.monotonic()-started))
            del current
            if backup and (step==100 or step%344==0 or step==limit):
                with diagnostic_rng():backup(protected,step)
            generations=sorted(protected.glob('state_*.pt'))
            for old in generations[:-2]:old.unlink()
    if limit==3436:
        if sum(v[2] for v in rdp.history)!=3436 or rdp.history!=prv.history:raise RuntimeError('accountant release count differs')
        eps=rdp.get_epsilon(delta=1e-5,alphas=privacy['orders']);eps_prv=prv.get_epsilon(delta=1e-5,eps_error=.1,delta_error=1e-8)
        if max(eps,eps_prv)>4:raise RuntimeError('GTSRB epsilon exceeded')
        write_json(output/'gtsrb_privacy_accounting.json',dict(privacy,achieved_rdp=eps,achieved_prv=eps_prv,release_count=3436))
        write_json(output/'run_summary.json',dict(status='TRAINED',steps=3436,source_checkpoint_sha256=SOURCE_SHA,checkpoint_sha256=sha(output/'checkpoints/final.pt'),outer_kl_target='full',scientific_fingerprint=digest(row),private_rng_in_artifact=False))
        write_json(output/'randomness_audit.json',dict(rng,exact_matched_J_U_coupling=False,coupling_design='New independent N root; historical J/U roots are not reused',final_stream_position_hashes=stream_positions(streams)))
    return dict(status='TRAINED' if limit==3436 else 'PREFIX_COMPLETE',steps=step,invocation_initial_step=initial_step,seconds=time.monotonic()-started)


def load_checkpoint(path,row,step):
    model,architecture=load_selected(Path('gtsrb_inputs'),Path('gtsrb_inputs/public_pretrain_selected.pt'),SOURCE_SHA)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['optimizer_steps']!=step or state['scientific_config']!=row or state['trainable_mask_hash']!=architecture['trainable_mask_hash']:raise RuntimeError('GTSRB checkpoint/config/step mismatch')
    model.load_state_dict(state['model'],strict=True)
    return model.cuda().float().eval()

def attack_smoke(model,data,output):
    from autoattack import AutoAttack
    from trustml.convnext_gtsrb.evaluation import autoattack_inputs
    ids=data.split['indices']['probe'][:128];x,y=autoattack_inputs(*data.batch(ids),'cuda')
    with torch.no_grad():selected=model(x).argmax(1).eq(y).nonzero().flatten()[:8]
    if len(selected)<2:raise RuntimeError('insufficient clean-correct public examples for attack smoke')
    x,y=x[selected],y[selected];records=[]
    for component in ('apgd-ce','apgd-t','fab-t','square'):
        aa=AutoAttack(model,norm='Linf',eps=8/255,version='standard',seed=42,device='cuda',verbose=False)
        aa.attacks_to_run=[component]
        owner={'apgd-ce':aa.apgd,'apgd-t':aa.apgd_targeted,'fab-t':aa.fab,'square':aa.square}[component]
        original=owner.perturb;calls=[]
        def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
        owner.perturb=counted;started=time.monotonic()
        adv,_=aa.run_standard_evaluation(x,y,bs=64,return_labels=True)
        delta=float((adv-x).abs().max())
        if not calls or not torch.isfinite(adv).all() or delta>8/255+1e-6 or adv.min()<0 or adv.max()>1:raise RuntimeError('GTSRB component smoke invalid')
        records.append(dict(component=component,actual_perturb_calls=len(calls),examples=len(selected),max_linf=delta,seconds=time.monotonic()-started))
    write_json(output,dict(status='PASS',technical_smoke_only=True,public_sample_indices=[int(ids[i]) for i in selected.cpu()],components=records,normalization='ImageNet normalization once inside native model',precision='FP32',model_mode='eval'))

def evaluate_test(model,row,checkpoint,output):
    from trustml.ssg_gtsrb.official_test import GTSRBOfficialTest,full_autoattack,EXPECTED_EXAMPLES
    output.mkdir(parents=True,exist_ok=True)
    data=GTSRBOfficialTest(Path('gtsrb_inputs/GTSRB_Final_Test_Images.zip'),Path('gtsrb_inputs/GTSRB_Final_Test_GT.zip'))
    clean=[]
    for start in range(0,EXPECTED_EXAMPLES,64):
        x,y=data.batch(list(range(start,min(start+64,EXPECTED_EXAMPLES))));x,y=x.cuda().contiguous(),y.cuda()
        with torch.no_grad():logits=model(x)
        if logits.shape!=(len(y),43) or not torch.isfinite(logits).all():raise RuntimeError('GTSRB official logits invalid')
        clean.extend(logits.argmax(1).eq(y).cpu().tolist())
    aa=full_autoattack(model,data,output/'aa_8_255',epsilon=8/255,clean_correct=np.asarray(clean,dtype=bool),batch_size=64)
    if aa['status']!='PASS' or aa['attack_time_posthoc_mismatches']!=0:raise RuntimeError('GTSRB AA numerical warning requires evaluation-only review')
    write_json(output/'evaluation.json',dict(status='PASS',split='test',sample_count=12630,epsilon=8/255,checkpoint_sha256=sha(checkpoint),scientific_fingerprint=digest(row),ordered_test_hash=data.ordered_hash,clean_accuracy=float(np.mean(clean)),aa_accuracy=aa['final_survivors']/12630,autoattack=aa,normalization='ImageNet mean/std once inside native ConvNeXt model',outer_kl_target='full'))

def main():
    import yaml
    parser=argparse.ArgumentParser(description="Frozen GTSRB fixed-norm N control; one A100.")
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--private-state-dir",type=Path)
    parser.add_argument("--protocol",type=Path,default=Path("configs/gradient_controls/gtsrb_training_protocol.json"))
    parser.add_argument("--accounting",type=Path,default=Path("configs/gradient_controls/gtsrb_accounting.json"))
    parser.add_argument("--evaluate-checkpoint",type=Path)
    args=parser.parse_args()
    from trustml.convnext_cifar100.preflight import configure_gpu
    configure_gpu()
    row=yaml.safe_load(args.config.read_text())
    if args.evaluate_checkpoint:
        model=load_checkpoint(args.evaluate_checkpoint,row,3436)
        evaluate_test(model,row,args.evaluate_checkpoint,args.output)
        return
    if args.private_state_dir is None:parser.error("training requires --private-state-dir")
    public=args.output.resolve();private=args.private_state_dir.resolve()
    if public==private or public in private.parents or private in public.parents:
        parser.error("public output and protected resume directories must be separate")
    protocol=json.loads(args.protocol.read_text());privacy=json.loads(args.accounting.read_text())
    augmentation=json.loads(Path("configs/convnext_gtsrb/protocol.json").read_text())["augmentation"]
    data=GTSRBTrainOnly(Path("gtsrb_inputs/GTSRB_Final_Training_Images.zip"),Path("gtsrb_inputs/splits.json"),augmentation)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"scientific_config.json",row)
    print(json.dumps(train(row,protocol,privacy,data,args.output,args.private_state_dir),sort_keys=True))

if __name__ == "__main__":
    main()

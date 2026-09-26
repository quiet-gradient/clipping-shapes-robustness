"""Public-only numerical gate before retrying the incomplete Natural LR=.1 run.

No optimizer updates; preserve the failed checkpoint and original diagnostics.
Training is a fresh attempt, not an exact resume: intermediate optimizer/RNG
state was never persisted by the original job.
"""
import json
from pathlib import Path
import shutil
import tarfile

import requests
import torch

from trustml.convnext_cifar100.preflight import write_json
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.mechanism import gradients
from trustml.convnext_gtsrb.model import sha256
from .private import load_selected

RUN = "rect57_natural_lr0100_s42"
SIBLING = "rect57_natural_lr0050_s42"


def check_checkpoint(state, row):
    if state.get("scientific_config") != row or state.get("optimizer_steps") != 1375:
        raise ValueError("unexpected recovery checkpoint/config")
    if state.get("release_count") != 0 or "optimizer" in state:
        raise ValueError("unexpected recovery state; reconcile resume capability")
    if not all(torch.isfinite(t).all() for t in state["trainable"].values()):
        raise ValueError("failed checkpoint contains nonfinite parameters")


def preflight(spec, group, data, selected, output):
    if [r["run_id"] for r in group["runs"]] != [RUN]:
        raise ValueError("recovery authorized for exactly one incomplete Natural run")
    row = group["runs"][0]
    if row["private"] or row["beta"] != 0 or row["lr"] != .1:
        raise ValueError("recovery changed scientific parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = Path("natural_failure_source.tar.gz")
    try:
        with requests.get(spec["download_url"], stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            with archive.open("xb") as handle:
                for chunk in response.iter_content(4*1024*1024):
                    handle.write(chunk)
    except requests.RequestException:
        raise RuntimeError("recovery download failed; signed URL suppressed") from None
    if archive.stat().st_size != spec["bytes"] or sha256(archive) != spec["sha256"]:
        raise ValueError("failed-job fallback SHA/size mismatch")
    wanted = {
        f"outputs/rect57/{RUN}/checkpoints/epoch020.pt": "epoch020.pt",
        f"outputs/rect57/{RUN}/scientific_config.json": "scientific_config.json",
        f"outputs/rect57/{RUN}/evaluation_epoch020/metrics.json": "original_epoch020_metrics.json",
        f"outputs/rect57/{SIBLING}/summary.json": "preserved_sibling_summary.json",
    }
    seen = set()
    with tarfile.open(archive,"r:gz") as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            if not member.isfile() or member.name in seen:
                raise ValueError("unsafe/duplicate recovery member")
            seen.add(member.name)
            with tar.extractfile(member) as src, (output/wanted[member.name]).open("xb") as dst:
                shutil.copyfileobj(src,dst)
    if seen != set(wanted):
        raise ValueError("required failed-job evidence missing")
    sibling=json.loads((output/'preserved_sibling_summary.json').read_text())
    if sibling['status']!='PASS' or sibling['checkpoint_sha256']!=spec['preserved_checkpoint_sha256']:
        raise ValueError("completed sibling identity mismatch")
    state=torch.load(output/'epoch020.pt',map_location='cpu',weights_only=True)
    check_checkpoint(state,row)
    if json.loads((output/'scientific_config.json').read_text()) != row:
        raise ValueError("original config mismatch")
    model, architecture=load_selected(Path('gtsrb_inputs'),selected,row['source_checkpoint_sha256'])
    if architecture['trainable_mask_hash']!=state['trainable_mask_hash']:
        raise ValueError("recovery trainable mask mismatch")
    params=dict(model.named_parameters())
    if set(state['trainable']) != {n for n,p in params.items() if p.requires_grad}:
        raise ValueError("recovery parameter set mismatch")
    with torch.no_grad():
        for name,value in state['trainable'].items():params[name].copy_(value)
    model.cuda().eval()
    ids=data.split['indices']['probe']
    if len(ids)!=512:raise ValueError('expected locked public probe512')
    failures=zeros=checked=0
    for begin in range(0,len(ids),2):
        selected_ids=ids[begin:begin+2]
        x,y=data.batch(selected_ids);x,y=x.cuda(),y.cuda()
        names=[data.split['records'][i]['image_id'] for i in selected_ids]
        adv,_=attack(model,x,y,objective='kl',uniform=public_uniform(x,names,20260921,'trades'))
        with torch.no_grad():
            z=model(x)
            if not torch.isfinite(z).all():raise RuntimeError('nonfinite checkpoint logits')
            zeros+=int((z.softmax(-1)==0).sum())
        try:
            gradients(model,x,y,adv,need_g=False)
        except RuntimeError as exc:
            if str(exc)!='nonfinite per-record gradient':raise
            failures+=1
        stable=gradients(model,x,y,adv,need_g=False,stable_public_kl=True)
        if not torch.isfinite(stable['h']).all():raise RuntimeError('stable public KL failed')
        checked+=len(x)
        if checked%64==0:
            print(json.dumps(dict(phase='natural_kl_recovery_probe',public_records=checked,
                                 original_nonfinite_batches=failures)),flush=True)
    if not failures or not zeros:
        raise RuntimeError('original underflow failure not reproduced; investigate before retraining')
    report=dict(status='PASS',source_job_id=spec['job_id'],run_id=RUN,
                source_archive_sha256=spec['sha256'],checkpoint_sha256=sha256(output/'epoch020.pt'),
                public_records=checked,original_nonfinite_batches=failures,softmax_zero_entries=zeros,
                stable_nonfinite_batches=0,training_updates=0,official_test_loaded=False,
                recovery_mode='fresh_same_config_retry_not_resume',
                reason='intermediate momentum and RNG states were not saved',
                training_KL_changed=False,public_geometry_KL='sum(exp(logp)*(logp-logq))',
                preserved_run=SIBLING,preserved_checkpoint_sha256=spec['preserved_checkpoint_sha256'])
    write_json(output/'preflight.json',report)
    print(json.dumps(dict(phase='natural_kl_recovery_preflight',**report)),flush=True)
    del model,state
    torch.cuda.empty_cache()

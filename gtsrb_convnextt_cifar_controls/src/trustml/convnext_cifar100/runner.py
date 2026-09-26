"""Portable fixed-final-iterate training; no test access or HPO-based epoch choice."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import secrets
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

from .accounting import SAMPLE_RATE, STEPS, bounds
from .data import TrainOnlyData, indices_hash
from .mechanism import assign_gradient, gradient_branches, lr_at, private_release, query
from .model import build_model, sha256
from .preflight import configure_gpu, write_json
from .attacks import trades_adversarial


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def validate_config(config):
    expected={"seed":42,"beta":1.,"steps":1760,"warmup_steps":176,"batch_denominator":512,
              "sample_rate":SAMPLE_RATE,"target_epsilon":4.,"delta":1e-5,"optimizer":"AdamW",
              "adam_betas":[.9,.95],"adam_eps":1e-8,"weight_decay":.005,
              "attack_steps":10,"attack_epsilon":8/255,"attack_step_size":2/255,
              "precision":"FP32","official_test_enabled":False}
    for key,value in expected.items():
        if config.get(key)!=value: raise ValueError(f"locked protocol mismatch: {key}")
    if config["arm"] not in {"J","U"}: raise ValueError("HPO contains only J/U")
    if config["arm"]=="U" and config["q"] not in [.1,.25,1/3,.5]: raise ValueError("q grid mismatch")
    if config["lr"] not in [1e-4,4.5e-4,2e-3]: raise ValueError("LR grid mismatch")
    if config["C"]<=0 or config["sigma"]<=0: raise ValueError("invalid bound/noise")


def save_state(path,model,optimizer,*,step,config,privacy,final=False):
    # No Gaussian/sampler RNG state, private losses, or private gradients are released.
    state={"config":config,"config_hash":canonical_hash(config),"optimizer_steps":step,
           "privacy_events":step,"privacy":privacy,"source_checkpoint_sha256":config["checkpoint_sha256"]}
    if final:
        state["model"]={k:v.detach().cpu() for k,v in model.state_dict().items()}
        state["optimizer"]=optimizer.state_dict()
    else:
        state["trainable"]={k:v.detach().cpu() for k,v in model.named_parameters() if v.requires_grad}
    temporary=path.with_suffix(".tmp")
    torch.save(state,temporary)
    temporary.replace(path)


def dev_pgd(model,data,output,*,batch_size=64):
    model.eval()
    ids=data.split["indices"]["dev"]
    generator=torch.Generator(device="cuda").manual_seed(42)
    clean_all=[]; robust_all=[]; loss_all=[]
    start=time.monotonic()
    for begin in range(0,len(ids),batch_size):
        x,y=data.batch(ids[begin:begin+batch_size])
        x,y=x.cuda(),y.cuda()
        with torch.no_grad():
            logits=model(x)
            clean=logits.argmax(-1).eq(y)
            losses=F.cross_entropy(logits,y,reduction="none")
        adv=(x+(torch.rand(x.shape,device=x.device,generator=generator)*2-1)*(8/255)).clamp(0,1)
        robust=clean.clone()
        for iteration in range(21):
            adv=adv.detach().requires_grad_(True)
            logits=model(adv)
            if not torch.isfinite(logits).all(): raise RuntimeError("nonfinite dev logits")
            robust &= logits.argmax(-1).eq(y).detach()
            if iteration==20: break
            gradient=torch.autograd.grad(F.cross_entropy(logits,y,reduction="sum"),adv)[0]
            adv=adv.detach()+(2/255)*gradient.sign()
            adv=(x+(adv-x).clamp(-8/255,8/255)).clamp(0,1)
        if float((adv-x).abs().max())>8/255+1e-6: raise RuntimeError("PGD Linf violation")
        clean_all.extend(clean.cpu().tolist());robust_all.extend(robust.cpu().tolist())
        loss_all.extend(losses.cpu().tolist())
        if begin%512==0: print(json.dumps({"stage":"dev_pgd","examples":min(begin+batch_size,5000)}),flush=True)
    if len(ids)!=5000 or len(set(ids))!=5000: raise RuntimeError("dev example count mismatch")
    np.savez_compressed(output/"dev_predictions.npz",sample_ids=np.asarray(ids),
        clean_correct=np.asarray(clean_all),pgd_correct=np.asarray(robust_all),clean_loss=np.asarray(loss_all))
    result=dict(dev_clean=100*float(np.mean(clean_all)),dev_pgd20=100*float(np.mean(robust_all)),
                examples=5000,split_hash=indices_hash(ids),epsilon=8/255,steps=20,restarts=1,
                attack_seed=42,precision="FP32",model_mode="eval",official_test_loaded=False,
                elapsed_seconds=time.monotonic()-start)
    write_json(output/"dev_metrics.json",result)
    return result


def train(config_path,checkpoint,data_root,output_root):
    config=json.loads(Path(config_path).read_text())
    validate_config(config)
    device=configure_gpu()
    output=Path(output_root)/config["run_id"]
    if output.exists(): raise FileExistsError("Never overwrite a previous attempt; validate/recover it first")
    output.mkdir(parents=True)
    start=time.monotonic()
    write_json(output/"config.json",config)
    write_json(output/"environment.json",{
        "packages":{name:importlib.metadata.version(name) for name in
            ["torch","torchvision","timm","numpy","scipy","opacus","wandb","Pillow"]},
        "gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,
        "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn":torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark":torch.backends.cudnn.benchmark})
    print(json.dumps({"run_id":config["run_id"],"stage":"initializing","gpu":torch.cuda.get_device_name()}),flush=True)
    privacy=bounds(config["sigma"])
    privacy["sensitivity_bound"]=config["C"]
    privacy["absolute_noise_std_before_averaging"]=config["sigma"]*config["C"]
    if privacy["status"]!="PASS": raise RuntimeError("privacy calibration failed; do not train")
    write_json(output/"accounting.json",privacy)
    if sha256(Path(data_root)/"train_only.npz")!=config["data_sha256"]: raise RuntimeError("dataset hash mismatch")
    data=TrainOnlyData(data_root)
    if data.split["hashes"]!=config["split_hashes"]: raise RuntimeError("split hashes mismatch")
    model,architecture=build_model(checkpoint,seed=42,expected_sha=config["checkpoint_sha256"])
    model.to(device)
    write_json(output/"architecture.json",architecture)
    # Separate secret, unlogged training streams. Public seed controls only model initialization.
    # Confirmation CRN is a different, locked phase; the HPO does not reuse Gaussian noise.
    sample_gen=torch.Generator().manual_seed(secrets.randbits(63))
    attack_gen=torch.Generator(device=device).manual_seed(secrets.randbits(63))
    noise_gen=torch.Generator(device=device).manual_seed(secrets.randbits(63))
    torch.manual_seed(secrets.randbits(63))
    random.seed(secrets.randbits(256))
    np.random.seed(secrets.randbits(32))
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config["lr"],betas=(.9,.95),eps=1e-8,weight_decay=.005)
    from opacus.accountants import RDPAccountant,PRVAccountant
    rdp,prv=RDPAccountant(),PRVAccountant()
    import wandb
    wandb_run=wandb.init(project="convnext-cifar100-dp-trades",name=config["run_id"],mode="offline",
                        dir=str(output),config=config,reinit=True)
    snapshots=output/"checkpoints"
    snapshots.mkdir()
    save_state(snapshots/"step0000.pt",model,optimizer,step=0,config=config,privacy=None)
    private_ids=np.asarray(data.split["indices"]["private"])
    dimension=sum(p.numel() for p in params)
    history=output/"history.jsonl"
    try:
        for step in range(1,1761):
            step_start=time.monotonic()
            indices=torch.nonzero(torch.rand(45000,generator=sample_gen)<SAMPLE_RATE).flatten().numpy()
            ids=private_ids[indices]
            summed=torch.zeros(dimension,device=device)
            for begin in range(0,len(ids),config["microbatch"]):
                x,y=data.batch(ids[begin:begin+config["microbatch"]],augment=True)
                x,y=x.to(device),y.to(device)
                with torch.no_grad(): probs=model(x).softmax(-1)
                adv=trades_adversarial(model,x,probs,epsilon=8/255,step_size=2/255,steps=10,generator=attack_gen)
                g,h=gradient_branches(model,x,y,adv,probs)
                contribution=query(g,h,bound=config["C"],q=config["q"],arm=config["arm"])
                if not torch.isfinite(contribution).all(): raise RuntimeError("scientific failure: nonfinite per-record contribution")
                if contribution.norm(dim=-1).max()>config["C"]*(1+2e-5): raise RuntimeError("scientific failure: sensitivity bound")
                summed.add_(contribution.sum(0))
                del g,h,contribution
            update=private_release(summed,sigma=config["sigma"],bound=config["C"],generator=noise_gen)
            optimizer.zero_grad(set_to_none=True)
            assign_gradient(model,update)
            learning_rate=lr_at(step,config["lr"])
            for group in optimizer.param_groups: group["lr"]=learning_rate
            optimizer.step()  # Adam moments and decoupled decay consume only privatized gradients.
            rdp.step(noise_multiplier=config["sigma"],sample_rate=SAMPLE_RATE)
            prv.step(noise_multiplier=config["sigma"],sample_rate=SAMPLE_RATE)
            if not all(torch.isfinite(p).all() for p in params): raise RuntimeError("scientific failure: nonfinite weights")
            row={"step":step,"lr":learning_rate,"elapsed_seconds":time.monotonic()-start,
                 "step_seconds":time.monotonic()-step_start,"privacy_events":step}
            with history.open("a") as handle: handle.write(json.dumps(row)+"\n")
            if step==1 or step%20==0:
                print(json.dumps(row),flush=True)
                wandb_run.log(row,step=step)
            if step in [440,880,1320]:
                save_state(snapshots/f"step{step:04d}.pt",model,optimizer,step=step,config=config,privacy=None)
        if sum(item[2] for item in rdp.history)!=1760 or rdp.history!=prv.history:
            raise RuntimeError("accountant event mismatch")
        save_state(snapshots/"final.pt",model,optimizer,step=1760,config=config,privacy=privacy,final=True)
        write_json(output/"accountant_history.json",{"rdp":rdp.history,"prv":prv.history,"events":1760})
        dev=dev_pgd(model,data,output)
        wandb_run.log(dev,step=1760)
        summary=dict(status="PASS",run_id=config["run_id"],config_hash=canonical_hash(config),
                     optimizer_steps=1760,accountant_events=1760,privacy=privacy,dev=dev,
                     elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),
                     wandb_run_id=wandb_run.id,checkpoint_sha256=sha256(snapshots/"final.pt"),
                     torch_version=torch.__version__,source_git=os.environ.get("TRUSTML_GIT_COMMIT"),
                     bundle_sha256=os.environ.get("TRUSTML_BUNDLE_HASH"),
                     randomness="independent internal streams; not cryptographic RNG certification")
        write_json(output/"summary.json",summary)
    finally:
        wandb_run.finish()
    return output


def commit_child(output,uploads=None):
    """Immutable transaction: archive -> checksum -> upload/verify -> COMMITTED."""
    output=Path(output)
    summary=json.loads((output/"summary.json").read_text())
    if summary["status"]!="PASS" or summary["optimizer_steps"]!=1760: raise RuntimeError("incomplete child")
    if sha256(output/"checkpoints/final.pt")!=summary["checkpoint_sha256"]: raise RuntimeError("checkpoint changed")
    file_hashes={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"artifact_manifest.json",file_hashes)
    archive=output.parent/f"{output.name}.tar.gz"
    with tarfile.open(archive,"x:gz") as tar: tar.add(output,arcname=output.name)
    digest=sha256(archive)
    receipt=dict(status="COMMITTED",run_id=output.name,config_hash=summary["config_hash"],
                 archive_sha256=digest,archive_bytes=archive.stat().st_size,
                 checkpoint_sha256=summary["checkpoint_sha256"],committed_unix=time.time())
    receipt_path=output.parent/f"{output.name}.COMMITTED.json"
    if uploads:
        for attempt in range(3):
            with archive.open("rb") as handle:
                response=requests.put(uploads["archive_put"],data=handle,timeout=(30,1800))
            if response.status_code in [200,201]: break
            if attempt==2: raise RuntimeError(f"artifact persistence HTTP {response.status_code}")
            time.sleep(5)
        # Verify bytes remotely, not just uploader-provided SHA metadata.
        verifier=hashlib.sha256(); count=0
        with requests.get(uploads["archive_get"],stream=True,timeout=(30,1800)) as response:
            if response.status_code!=200: raise RuntimeError(f"archive verification HTTP {response.status_code}")
            for chunk in response.iter_content(4*1024*1024): verifier.update(chunk);count+=len(chunk)
        if verifier.hexdigest()!=digest or count!=archive.stat().st_size:
            raise RuntimeError("persisted archive content mismatch")
        receipt["object_key"]=uploads["archive_key"]
        # Publish the commit only after the remote archive is verified.
        response=requests.put(uploads["receipt_put"],data=json.dumps(receipt).encode(),timeout=(30,120))
        if response.status_code not in [200,201]: raise RuntimeError("commit receipt upload failed")
    else:
        receipt["persistence"]="MLM output artifact; remote download verification still required"
    write_json(receipt_path,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--run-id",action="append",default=[])
    p.add_argument("--chain",choices=["A","B"])
    p.add_argument("--uploads")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())
    rows=[r for r in manifest["runs"] if r["run_id"] in a.run_id or (a.chain and r["backend"]=="datasphere" and r["chain"]==a.chain)]
    if not rows: raise ValueError("no matching runs")
    if a.chain and not a.uploads: raise ValueError("DataSphere chains require transactional persistence")
    uploads=json.loads(Path(a.uploads).read_text()) if a.uploads else {}
    ledger={"chain_start":time.time(),"completed":[],"failed":[],"runtime_killer":False}
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    try:
        for row in rows:
            if row["config_hash"]!=canonical_hash(json.loads(Path(row["config_path"]).read_text())):
                raise RuntimeError("immutable configuration hash mismatch")
            local_receipt=root/f"{row['run_id']}.COMMITTED.json"
            if local_receipt.exists():
                receipt=json.loads(local_receipt.read_text())
                if receipt["config_hash"]!=row["config_hash"]: raise RuntimeError("existing committed config mismatch")
                ledger["completed"].append(receipt)
                continue
            # Remote receipts are also checked on recovery before any new computation.
            routes=uploads.get(row["run_id"])
            if routes:
                remote=requests.get(routes["receipt_get"],timeout=(30,120))
                if remote.status_code==200:
                    receipt=remote.json()
                    if receipt["status"] not in {"COMMITTED","RESERVED"} or receipt["config_hash"]!=row["config_hash"]:
                        raise RuntimeError("remote receipt mismatch")
                    if receipt["status"]=="COMMITTED":
                        write_json(local_receipt,receipt);ledger["completed"].append(receipt);continue
                elif remote.status_code!=404: raise RuntimeError(f"cannot check recovery receipt: HTTP {remote.status_code}")
            output=train(row["config_path"],a.checkpoint,a.data,a.output)
            receipt=commit_child(output,routes)
            ledger["completed"].append(receipt)
            write_json(root/"chain_ledger.json",ledger)
    except BaseException as exc:
        ledger["failed"].append({"run_id":row["run_id"],"type":type(exc).__name__,"message":str(exc)[:500]})
        raise
    finally:
        ledger["chain_end"]=time.time()
        ledger["total_gpu_seconds"]=ledger["chain_end"]-ledger["chain_start"]
        write_json(root/"chain_ledger.json",ledger)


if __name__=="__main__": main()

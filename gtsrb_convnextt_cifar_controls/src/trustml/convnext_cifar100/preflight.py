"""Public-only GPU gradient audit and clipping calibration. No training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model, sha256
from .data import TrainOnlyData
from .mechanism import gradient_branches, query
from .attacks import trades_adversarial


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def configure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU computations must run on remote CUDA, not the workstation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return "cuda"


def run(checkpoint, data_root, output, microbatch=4, benchmark=False):
    device = configure_gpu()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .accounting import calibrate
    write_json(output/"privacy_calibration.json", calibrate())
    model, info = build_model(checkpoint)
    model.to(device)
    write_json(output/"architecture.json", info)
    data = TrainOnlyData(data_root)
    generator = torch.Generator(device=device).manual_seed(20260919)
    x = torch.rand((2,3,32,32), generator=generator, device=device)
    y = torch.tensor([1,71], device=device)
    with torch.no_grad():
        probs = model(x).softmax(-1)
    adv = trades_adversarial(model,x,probs,epsilon=8/255,step_size=2/255,steps=10,generator=generator)
    g,h = gradient_branches(model,x,y,adv,probs)
    params = [p for p in model.parameters() if p.requires_grad]
    ce = F.cross_entropy(model(x[:1]),y[:1])
    kl = F.kl_div(model(adv[:1]).log_softmax(-1), probs[:1], reduction="sum")
    g_ref = torch.cat([t.flatten() for t in torch.autograd.grad(ce,params)])
    h_ref = torch.cat([t.flatten() for t in torch.autograd.grad(kl,params)])
    torch.testing.assert_close(g[0],g_ref,rtol=3e-4,atol=5e-6)
    torch.testing.assert_close(h[0],h_ref,rtol=3e-4,atol=5e-6)
    composite = F.cross_entropy(model(x[:1]),y[:1]) + F.kl_div(model(adv[:1]).log_softmax(-1),probs[:1],reduction="sum")
    s_ref = torch.cat([t.flatten() for t in torch.autograd.grad(composite,params)])
    torch.testing.assert_close(g[0]+h[0],s_ref,rtol=3e-4,atol=1e-5)
    offset=0
    gradient_norms={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            n=p.numel()
            norm=float((g[:,offset:offset+n]+h[:,offset:offset+n]).norm())
            if not np.isfinite(norm) or norm<=0:
                raise RuntimeError(f"trainable gradient missing: {name}")
            gradient_norms[name]=norm
            offset+=n
        elif p.grad is not None:
            raise RuntimeError("frozen parameter received gradient")
    write_json(output/"gradient_audit.json",dict(status="PASS",gradient_norms=gradient_norms,
        ce_reference_max_abs=float((g[0]-g_ref).abs().max()),
        kl_reference_max_abs=float((h[0]-h_ref).abs().max()), global_dimension=g.shape[1],
        composite_reference_max_abs=float((g[0]+h[0]-s_ref).abs().max())))
    del g,h,g_ref,h_ref,s_ref
    ids=data.split["indices"]["probe"]
    records=[]
    start=time.monotonic()
    for start_idx in range(0,len(ids),microbatch):
        x,y=data.batch(ids[start_idx:start_idx+microbatch])
        x,y=x.to(device),y.to(device)
        with torch.no_grad(): probs=model(x).softmax(-1)
        adv=trades_adversarial(model,x,probs,epsilon=8/255,step_size=2/255,steps=10,generator=generator)
        g,h=gradient_branches(model,x,y,adv,probs)
        values=torch.stack([g.norm(dim=-1),h.norm(dim=-1),(g+h).norm(dim=-1)],dim=1)
        if not torch.isfinite(values).all(): raise RuntimeError("nonfinite probe gradients")
        records.extend(values.cpu().tolist())
        if start_idx%64==0: print(json.dumps({"probe_examples":min(start_idx+microbatch,512)}),flush=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    values=np.asarray(records)
    cref=float(np.median(values[:,2]))
    if not np.isfinite(cref) or cref<=0: raise RuntimeError("invalid clipping center")
    grid=[cref/4,cref,4*cref]
    fractions=[float((values[:,2]>bound).mean()) for bound in grid]
    if all(f>.95 for f in fractions) or all(f<.05 for f in fractions):
        raise RuntimeError("degenerate clipping grid")
    np.savez_compressed(output/"public_gradient_norms.npz", sample_ids=np.asarray(ids),norm_g=values[:,0],norm_h=values[:,1],norm_s=values[:,2])
    # Timing proxy includes attack + both per-record branch gradients at actual 32x32 resolution.
    result=dict(status="PASS",C_ref=cref,C_grid=grid,clip_fractions=fractions,
        public_probe_count=512,probe_hash=data.split["hashes"]["probe"],dev_hash=data.split["hashes"]["dev"],
        checkpoint_sha256=sha256(checkpoint),data_sha256=sha256(Path(data_root)/"train_only.npz"),
        probe_seconds=elapsed,estimated_training_seconds=elapsed*1760,
        estimated_run_seconds=elapsed*1760*1.15,
        timing_note="proxy: 512 public examples, adds 15% optimizer/eval overhead; not measured full training",
        microbatch=microbatch,gpu=torch.cuda.get_device_name(),architecture=info,
        official_test_loaded=False)
    if benchmark:
        # Public-only implementation timing, never a training/HPO result.
        # Keep the original 512-example clipping calibration unchanged.
        del g,h,values,adv,x,y
        torch.cuda.empty_cache()
        timings=[]
        for candidate in [8,16,32]:
            torch.cuda.reset_peak_memory_stats()
            try:
                # Warm up and check batch independence against the audited path.
                batch_ids=ids[:candidate]
                bx,by=data.batch(batch_ids)
                bx,by=bx.to(device),by.to(device)
                with torch.no_grad(): bp=model(bx).softmax(-1)
                ba=trades_adversarial(model,bx,bp,epsilon=8/255,step_size=2/255,steps=10,generator=generator)
                bg,bh=gradient_branches(model,bx,by,ba,bp)
                sg,sh=gradient_branches(model,bx[:2],by[:2],ba[:2],bp[:2])
                torch.testing.assert_close(bg[:2],sg,rtol=3e-4,atol=1e-5)
                torch.testing.assert_close(bh[:2],sh,rtol=3e-4,atol=1e-5)
                del bg,bh,sg,sh,ba,bx,by,bp
                torch.cuda.synchronize(); timer=time.monotonic()
                for begin in range(0,64,candidate):
                    bx,by=data.batch(ids[begin:begin+candidate],augment=True)
                    bx,by=bx.to(device),by.to(device)
                    with torch.no_grad(): bp=model(bx).softmax(-1)
                    ba=trades_adversarial(model,bx,bp,epsilon=8/255,step_size=2/255,steps=10,generator=generator)
                    bg,bh=gradient_branches(model,bx,by,ba,bp)
                    contribution=query(bg,bh,bound=cref,q=1/3,arm="U")
                    if not torch.isfinite(contribution).all(): raise RuntimeError("nonfinite public contribution")
                    del bg,bh,contribution,ba,bx,by,bp
                torch.cuda.synchronize()
                seconds=(time.monotonic()-timer)*8
                peak=torch.cuda.max_memory_allocated()
                if peak>.85*torch.cuda.get_device_properties(0).total_memory:
                    raise RuntimeError("insufficient GPU memory headroom")
                timings.append({"microbatch":candidate,"public_512_estimated_seconds":seconds,"peak_bytes":peak})
                print(json.dumps({"throughput_benchmark":timings[-1]}),flush=True)
            except torch.cuda.OutOfMemoryError:
                print(json.dumps({"microbatch":candidate,"benchmark":"does_not_fit"}),flush=True)
                torch.cuda.empty_cache()
                break
        if not timings: raise RuntimeError("no larger microbatch passed public benchmark")
        selected=min(timings,key=lambda item:item["public_512_estimated_seconds"])
        result.update(throughput_benchmarks=timings,microbatch=selected["microbatch"],
            estimated_training_seconds=selected["public_512_estimated_seconds"]*1760,
            estimated_run_seconds=selected["public_512_estimated_seconds"]*1760*1.20,
            timing_note="public 64-example warmed benchmark extrapolated to 512; includes augmentation/query plus 20% optimizer/eval reserve; no training")
    write_json(output/"calibration.json",result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--microbatch",type=int,default=4)
    p.add_argument("--benchmark",action="store_true")
    a=p.parse_args()
    run(a.checkpoint,a.data,a.output,a.microbatch,a.benchmark)


if __name__=="__main__": main()

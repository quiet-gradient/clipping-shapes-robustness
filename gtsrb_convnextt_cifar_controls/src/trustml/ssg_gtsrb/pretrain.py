"""Public-only TRADES pretrain. Explicit source/selection decisions and audit required.

No private fitting, no MLM, no official test. This entrypoint is not an
authorization to accept unresolved upstream provenance: the decision must be
supplied as a separate immutable input before a GPU job can execute it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.common import code_identity, diagnostic_rng
from trustml.convnext_gtsrb.data import digest
from trustml.convnext_gtsrb.evaluation import autoattack_inputs
from trustml.convnext_gtsrb.model import build_model, sha256
from trustml.convnext_cifar100.preflight import write_json


SAVE_EPOCHS = [40, 50, 60, 70, 80, 90, 100]


def public_lr(step, steps_per_epoch):
    if not 0 <= step < 100 * steps_per_epoch or steps_per_epoch < 1:
        raise ValueError("invalid public schedule position")
    warmup = 10 * steps_per_epoch
    if step < warmup:
        return .001 * (step + 1) / warmup
    phase = (step - warmup) / (100 * steps_per_epoch - warmup - 1)
    return 1e-6 + (.001 - 1e-6) * .5 * (1 + math.cos(math.pi * phase))


def validate_source_gate(audit, sources, decision):
    if audit["record_overlap_status"] != "PASS" or audit["exact_duplicate_count"] != 0:
        raise ValueError("public/real record-overlap gate failed")
    if audit["real_records"] != 39209 or audit["public_records"] != 43000:
        raise ValueError("incomplete source audit")
    if audit["synthetic_split"]["shared_train_val_scene_count"] or audit["synthetic_split"]["shared_train_val_pixel_hash_count"]:
        raise ValueError("synthetic train/validation leakage requires review")
    if audit["sources"] != sources:
        raise ValueError("source manifest changed since audit")
    if decision["renderers"] != ["cycles", "ogre"]:
        raise ValueError("both renderers are required: cycles and ogre")
    if decision.get("selection_rule") != "pareto_clean_within_1pp_max_aa":
        raise ValueError("Pareto/AA-1pp selection rule required")
    if not decision.get("decision_record"):
        raise ValueError("source/selection choices must have an immutable decision record")
    provenance = decision.get("generator_provenance")
    if provenance not in {"confirmed_independent", "explicitly_accepted_unresolved_with_limitation"}:
        raise ValueError("upstream generator provenance decision missing")
    if provenance == "confirmed_independent" and not decision.get("independence_evidence"):
        raise ValueError("independence cannot be asserted without evidence")


def checkpoint_frontier(rows):
    return [r for r in rows if not any(
        o["clean_correct_count"] >= r["clean_correct_count"] and
        o["aa_correct_count"] >= r["aa_correct_count"] and
        (o["clean_correct_count"] > r["clean_correct_count"] or
         o["aa_correct_count"] > r["aa_correct_count"]) for o in rows)]


def select_checkpoint(rows):
    if len(rows) != len(SAVE_EPOCHS) or sorted(r["epoch"] for r in rows) != SAVE_EPOCHS:
        raise ValueError("all seven saved epochs must have validated AA before selection")
    if len({r["examples"] for r in rows}) != 1 or len({r["ordered_ids_hash"] for r in rows}) != 1:
        raise ValueError("checkpoint comparison must use identical ordered validation data")
    if any(not 0 <= r["aa_correct_count"] <= r["clean_correct_count"] <= r["examples"] for r in rows):
        raise ValueError("invalid attack counts")
    best_count = max(r["aa_correct_count"] for r in rows)
    # Exact integer test: <= 1 percentage point, not a confidence interval.
    eligible = [r for r in checkpoint_frontier(rows)
                if 100 * (best_count - r["aa_correct_count"]) <= r["examples"]]
    return min(eligible, key=lambda r: (-r["clean_correct_count"], -r["aa_correct_count"], r["epoch"]))


def public_model(checkpoint, source_metadata):
    model, old = build_model(checkpoint, source_metadata, seed=42, drop_path_rate=.1)
    tensors = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("model.stages.2.", "model.stages.3.", "model.head.", "model.norm_pre.")))
        if parameter.requires_grad:
            tensors[name] = {"shape": list(parameter.shape), "numel": parameter.numel()}
    info = {key: value for key, value in old.items() if key not in {
        "trainable_tensors", "trainable_parameter_count", "trainable_mask_hash", "blocks",
        "ordered_trainable_tensor_names", "frozen_modules", "model_architecture_digest"}}
    info.update(trainable_tensors=tensors, ordered_trainable_tensor_names=list(tensors),
        trainable_parameter_count=sum(r["numel"] for r in tensors.values()), trainable_mask_hash=digest(tensors),
        model_architecture_digest=digest({"repr": str(model), "trainable": tensors}),
        frozen_modules=["model.stem", "model.stages.0", "model.stages.1"], drop_path_rate=.1)
    return model, info


class PublicSSG:
    """Only chosen pinned synthetic files enter this loader. No real images."""
    def __init__(self, sources, cache, renderers):
        import io
        from PIL import Image
        import pyarrow.parquet as pq
        from torchvision.transforms import RandomErasing
        from scripts.audit_ssg_gtsrb_overlap import fetch_checked
        self.rows, pixels = [], []
        for source in sources["files"]:
            renderer, name = source["path"].split("/")
            if renderer not in renderers:
                continue
            path = fetch_checked(source, sources, cache)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["image", "label", "sample_idx"]):
                for row in batch.to_pylist():
                    image_id = f"{renderer}/{int(row['label']):02d}/{int(row['sample_idx']):03d}"
                    image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
                    pixels.append(np.asarray(image, dtype=np.uint8))
                    self.rows.append({"image_id": image_id, "label": int(row["label"]), "split": Path(name).stem,
                                      "scene_group": f"{int(row['label']):02d}/{int(row['sample_idx']):03d}"})
        self.pixels = np.stack(pixels)
        self.ids = {split: [i for i, row in enumerate(self.rows) if row["split"] == split] for split in ["train", "validation"]}
        if len(self.ids["train"]) != 17200 * len(renderers) or len(self.ids["validation"]) != 4300 * len(renderers):
            raise ValueError("unexpected official synthetic split counts")
        train_groups = {self.rows[i]["scene_group"] for i in self.ids["train"]}
        val_groups = {self.rows[i]["scene_group"] for i in self.ids["validation"]}
        if train_groups & val_groups:
            raise ValueError("synthetic scenes cross train/validation")
        self.erase = RandomErasing(p=.25, value=0)
        self.manifest = {"source_revision": sources["revision"], "renderers": renderers, "rows": self.rows,
            "indices": self.ids, "hashes": {k: digest([self.rows[i] for i in v]) for k, v in self.ids.items()},
            "resize": "RGB bilinear 96x96", "augmentations": ["random_erasing_p0.25_value0"], "horizontal_flip": False}

    def batch(self, indices, *, augment=False):
        x = torch.from_numpy(self.pixels[np.asarray(indices)].copy()).permute(0, 3, 1, 2).float().div_(255)
        if augment:
            x = torch.stack([self.erase(image) for image in x])
        return x.contiguous(), torch.tensor([self.rows[i]["label"] for i in indices], dtype=torch.long)


def evaluate_synthetic(model, data, output, epoch, *, full_aa=False, batch_size=64):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    ids = data.ids["validation"]
    clean, losses, robust = [], [], []
    started = time.monotonic()
    for begin in range(0, len(ids), batch_size):
        batch_ids = ids[begin:begin + batch_size]
        x, y = data.batch(batch_ids)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite synthetic-val logits")
            clean.extend(logits.argmax(-1).eq(y).cpu().tolist())
            losses.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
        if not full_aa:
            uniform = public_uniform(x, [data.rows[i]["image_id"] for i in batch_ids], 42, "ssg_pgd10")
            _, mask = attack(model, x, y, objective="ce", steps=10, uniform=uniform, track_best=True)
            robust.extend(mask.cpu().tolist())
    arrays = {"sample_ids": np.array([data.rows[i]["image_id"] for i in ids]),
              "clean_correct": np.asarray(clean), "clean_ce": np.asarray(losses)}
    result = {"epoch": epoch, "examples": len(ids), "ordered_ids_hash": data.manifest["hashes"]["validation"],
              "clean_correct_count": int(sum(clean)), "clean_accuracy": float(100 * np.mean(clean)),
              "clean_ce": float(np.mean(losses)), "precision": "FP32", "model_mode": "eval",
              "epsilon": 8/255, "attack_seed": 42, "dataset": "synthetic_public_validation", "official_test_loaded": False}
    if full_aa:
        from autoattack import AutoAttack
        from autoattack.state import EvaluationState
        x, y = data.batch(ids)
        x, y = autoattack_inputs(x, y, device)
        aa = AutoAttack(model, norm="Linf", eps=8/255, version="standard", seed=42,
                        device=str(device), log_path=str(output / "autoattack.log"))
        state_path = output / "autoattack_state.json"
        adversarial = aa.run_standard_evaluation(x, y, bs=batch_size, state_path=state_path)
        if (not torch.isfinite(adversarial).all() or (adversarial - x).abs().max() > 8/255 + 1e-6
                or adversarial.min() < 0 or adversarial.max() > 1):
            raise RuntimeError("invalid AutoAttack perturbation")
        posthoc = []
        for begin in range(0, len(ids), batch_size):
            with torch.no_grad():
                logits = model(adversarial[begin:begin + batch_size])
                if not torch.isfinite(logits).all():
                    raise RuntimeError("nonfinite AA posthoc logits")
                posthoc.extend(logits.argmax(-1).eq(y[begin:begin + batch_size]).cpu().tolist())
        attack_time = EvaluationState.from_disk(state_path).robust_flags.cpu().numpy()
        posthoc = np.asarray(posthoc) & arrays["clean_correct"]
        arrays.update(aa_attack_time_correct=attack_time, aa_posthoc_correct=posthoc)
        result["attack_time_posthoc_mismatches"] = int(np.count_nonzero(attack_time != posthoc))
        robust = attack_time & posthoc
        result.update(aa_correct_count=int(robust.sum()), aa_accuracy=float(100 * robust.mean()),
                      autoattack_version="0.1 / a39220048b3c9f2cca9a4d3a54604793c68eca7e")
        metric = "aa"
    else:
        metric = "pgd10"
        result.update(pgd10_correct_count=int(sum(robust)), pgd10_accuracy=float(100 * np.mean(robust)))
    arrays[metric + "_correct"] = np.asarray(robust)
    if np.any(arrays[metric + "_correct"] & ~arrays["clean_correct"]):
        raise RuntimeError("synthetic validation attack monotonicity violation")
    result["seconds"] = time.monotonic() - started
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_json(output / "metrics.json", result)
    if result.get("attack_time_posthoc_mismatches", 0):
        raise RuntimeError("AA numerical disagreement saved; diagnose before selecting a checkpoint")
    return result


def run(args, store=None):
    if not torch.cuda.is_available():
        raise RuntimeError("public training must run on DataSphere CUDA, never locally")
    if "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("A100 required for the new public pretrain")
    audit = json.loads(args.audit.read_text())
    sources = json.loads(args.sources.read_text())
    decision = json.loads(args.decision.read_text())
    validate_source_gate(audit, sources, decision)
    identity = code_identity()
    if args.output.exists():
        raise FileExistsError("immutable public run already exists; no overwrite or implicit retrain")
    args.output.mkdir(parents=True)
    args.output.joinpath("checkpoints").mkdir()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata = json.loads((args.assets_root / "assets.json").read_text())["convnext_tiny_cvst_clean.pt"]
    model, architecture = public_model(args.assets_root / "convnext_tiny_cvst_clean.pt", metadata)
    model = model.cuda()
    data = PublicSSG(sources, args.cache, decision["renderers"])
    manifest = {"campaign": "ssg_gtsrb_v1", "stage": "public_robust_pretrain", "decision": decision,
        "audit_sha256": sha256(args.audit), "sources_manifest_sha256": sha256(args.sources),
        "source_checkpoint_sha256": metadata["sha256"], "architecture": architecture, **identity,
        "optimizer": {"family": "AdamW", "betas": [.9, .95], "eps": 1e-8, "weight_decay": .01},
        "epochs": 100, "effective_batch_size": 256, "microbatch_size": args.microbatch,
        "drop_last": False, "last_batch_size": len(data.ids["train"]) % 256,
        "steps_per_epoch": math.ceil(len(data.ids["train"]) / 256),
        "last_batch_loss_denominator": "actual_batch_size",
        "beta": 3, "label_smoothing": .1, "drop_path": .1, "train_real_records": 0,
        "trainable_tensor_names": architecture["ordered_trainable_tensor_names"], "official_test_loaded": False}
    write_json(args.output / "ssg_convnext_trades_pretrain_manifest.json", manifest)
    write_json(args.output / "synthetic_split_manifest.json", data.manifest)
    import importlib.metadata
    write_json(args.output / "environment.json", {
        "packages": {k: importlib.metadata.version(k) for k in ["torch", "torchvision", "timm", "numpy", "scipy", "wandb", "pyarrow", "Pillow"]},
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "FP32": True, "TF32": False})
    import wandb
    wb = wandb.init(project="dp-robustness", name="ssg-gtsrb-public-trades-s42", mode="offline", dir=str(args.output), config=manifest)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001,
                                 betas=(.9, .95), eps=1e-8, weight_decay=.01)
    steps_per_epoch = math.ceil(len(data.ids["train"]) / 256)
    step = 0
    curves = []
    started = time.monotonic()
    for epoch in range(1, 101):
        # Public epoch shuffle independent of any evaluation or attack RNG usage.
        indices = np.random.default_rng(42000 + epoch).permutation(data.ids["train"]).tolist()
        epoch_started = time.monotonic()
        train_ce, train_kl, observed = 0., 0., 0
        for begin in range(0, len(indices), 256):
            logical = indices[begin:begin + 256]
            rate = public_lr(step, steps_per_epoch)
            optimizer.param_groups[0]["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(logical), args.microbatch):
                chunk = logical[offset:offset + args.microbatch]
                x, y = data.batch(chunk, augment=True)
                x, y = x.cuda(), y.cuda()
                model.eval()
                x_adv, _ = attack(model, x, objective="kl", steps=10)
                model.train()
                model.model.stem.eval()
                model.model.stages[0].eval()
                model.model.stages[1].eval()
                logits, logits_adv = model(x), model(x_adv.detach())
                task = F.cross_entropy(logits, y, label_smoothing=.1, reduction="sum")
                robust = F.kl_div(logits_adv.log_softmax(-1), logits.softmax(-1), reduction="sum")
                loss = (task + 3 * robust) / len(logical)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite public TRADES loss")
                loss.backward()
                train_ce += task.item()
                train_kl += robust.item()
                observed += len(chunk)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("nonfinite public parameter gradient")
            optimizer.step()
            step += 1
            if step == 1 or step % 20 == 0:
                print(json.dumps({"phase": "public_training_step", "epoch": epoch, "optimizer_step": step,
                    "total_optimizer_steps": 100 * steps_per_epoch, "learning_rate": rate,
                    "elapsed_seconds": time.monotonic() - started}), flush=True)
        row = {"epoch": epoch, "optimizer_step": step, "learning_rate": rate, "train_ce": train_ce / observed,
               "train_kl": train_kl / observed, "train_examples": observed,
               "epoch_seconds": time.monotonic() - epoch_started, "elapsed_seconds": time.monotonic() - started}
        # Every ten epochs: immutable recovery checkpoint, only 40..100 eligible.
        if epoch % 10 == 0:
            checkpoint = args.output / "checkpoints" / f"epoch{epoch:03d}.pt"
            model.eval()
            state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "epoch": epoch, "optimizer_step": step,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "public_manifest": manifest, "source_checkpoint_sha256": metadata["sha256"],
                "validation_order_hash": data.manifest["hashes"]["validation"]}
            torch.save(state, checkpoint)
            row["checkpoint_sha256"] = sha256(checkpoint)
        if epoch in SAVE_EPOCHS:
            with diagnostic_rng():
                row.update(evaluate_synthetic(model, data, args.output / f"validation_epoch{epoch:03d}", epoch))
        curves.append(row)
        write_json(args.output / "public_training_curves.json", curves)
        wb.log(row, step=step)
        print(json.dumps({"phase": "public_pretrain", **row}), flush=True)
        if store is not None and epoch % 10 == 0:
            paths = ["ssg_convnext_trades_pretrain_manifest.json", "synthetic_split_manifest.json",
                     "environment.json", "public_training_curves.json", f"checkpoints/epoch{epoch:03d}.pt"]
            paths += [str(p.relative_to(args.output)) for p in (args.output / f"validation_epoch{epoch:03d}").rglob("*") if p.is_file()]
            store.commit(f"public_epoch{epoch:03d}", args.output, paths)
    screened = [r for r in curves if r["epoch"] in SAVE_EPOCHS]
    aa_rows = []
    for candidate in screened:
        epoch = candidate["epoch"]
        path = args.output / "checkpoints" / f"epoch{epoch:03d}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
        with diagnostic_rng():
            row = evaluate_synthetic(model, data, args.output / f"autoattack_epoch{epoch:03d}", epoch, full_aa=True)
        row["checkpoint_sha256"] = sha256(path)
        row["pgd10_accuracy"] = candidate["pgd10_accuracy"]
        aa_rows.append(row)
        write_json(args.output / "public_autoattack_candidates.json", aa_rows)
        if store is not None:
            store.commit(f"public_aa{epoch:03d}", args.output / f"autoattack_epoch{epoch:03d}")
    selected = select_checkpoint(aa_rows)
    if selected["aa_correct_count"] == 0 or selected["clean_accuracy"] <= 2 * 100 / 43:
        raise RuntimeError("public pretrain is unhealthy: no private training permitted")
    selected_path = args.output / "checkpoints" / f"epoch{selected['epoch']:03d}.pt"
    import shutil
    shutil.copy2(selected_path, args.output / "public_pretrain_selected.pt")
    write_json(args.output / "public_pretrain_selection.json", {
        "selected": selected, "all_candidates": aa_rows, "decision": decision,
        "pareto_epochs": [r["epoch"] for r in checkpoint_frontier(aa_rows)],
        "selection_aa_tolerance_pp": 1.0, "threshold_is_confidence_interval": False,
        "selection_uses_real_private_or_dev": False, "status": "PUBLIC_CHECKPOINT_LOCKED",
        "selected_checkpoint_sha256": sha256(args.output / "public_pretrain_selected.pt")})
    wb.finish()
    from .reports import public_reports
    public_reports(args.output, selected, curves, aa_rows)
    write_json(args.output / "artifact_manifest.json", {str(p.relative_to(args.output)): sha256(p)
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})
    if store is not None:
        store.commit("public_selected", args.output, ["public_pretrain_selected.pt", "public_pretrain_selection.json",
                     "ssg_convnext_trades_pretrain_manifest.json", "synthetic_split_manifest.json", "environment.json",
                     "public_training_curves.json", "public_autoattack_candidates.json", "artifact_manifest.json",
                     "ssg_convnext_trades_pretrain_curves.md", "ssg_convnext_trades_pretrain_selection.md"])
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path("configs/ssg_gtsrb_v1/sources.json"))
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=Path("gtsrb_inputs"))
    parser.add_argument("--cache", type=Path, default=Path("ssg_cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ssg_public_pretrain"))
    parser.add_argument("--microbatch", type=int, default=64)
    args = parser.parse_args()
    if args.microbatch < 1 or args.microbatch > 256:
        parser.error("microbatch must lie in 1..256; logical batch remains 256")
    run(args)

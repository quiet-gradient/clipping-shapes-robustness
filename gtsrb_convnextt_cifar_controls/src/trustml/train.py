from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf

from trustml.core.config import load_config
from trustml.core.seed import seed_everything
from trustml.data.registry import build_datamodule
from trustml.defenses.registry import build_defenses
from trustml.logging.composite_logger import build_logger
from trustml.models.registry import build_model
from trustml.training.trainer import Trainer


def run(cfg):
    if cfg.get("intervention", None) is not None:
        raise ValueError("Intervention configs require trustml.cifar_interventions.runtime; legacy defense dispatch is forbidden")
    seed_everything(
        int(cfg.experiment.get("seed", 42)),
        deterministic=bool(cfg.experiment.get("deterministic", False)),
    )
    _restore_wandb_identity_for_resume(cfg)
    logger = build_logger(cfg)
    result = {}
    try:
        logger.log_config(cfg)
        data = build_datamodule(cfg.data, audit_cfg=cfg.get("audit", None))
        audit_pack = data.pop("audit_pack", None)
        model = build_model(cfg.model, root_cfg=cfg)
        defenses = build_defenses(cfg.get("defenses", []))

        trainer = Trainer(
            cfg=cfg,
            model=model,
            data=data,
            defenses=defenses,
            logger=logger,
        )
        audit_state = None
        if audit_pack is not None:
            # One-run epsilon audit (Steinke et al., NeurIPS'23): canary pack and
            # w0 scores are persisted before training so the bound can be
            # recomputed offline from final.pt even if the in-run audit fails.
            from trustml.audit.runner import finalize_audit, prepare_audit

            audit_state = prepare_audit(cfg, trainer, data, audit_pack)
        train_metrics = trainer.fit()
        # Weights go to disk before the eval phase: no attack eval may cost us
        # the trained model (a numpy incompat in MIA once killed finished runs).
        final_path = trainer.save_checkpoint("final.pt", metrics=train_metrics)
        try:
            eval_metrics = trainer.evaluate()
        except Exception:
            print(f"evaluation failed:\n{traceback.format_exc()}", file=sys.stderr)
            eval_metrics = {"eval/failed": 1.0}
        if audit_state is not None:
            try:
                audit_metrics = finalize_audit(cfg, trainer, data, audit_state)
                logger.log_metrics(audit_metrics, step=trainer.global_step)
                eval_metrics = {**eval_metrics, **audit_metrics}
            except Exception:
                print(f"audit failed:\n{traceback.format_exc()}", file=sys.stderr)
                eval_metrics = {**eval_metrics, "audit/failed": 1.0}
        if "eval/failed" not in eval_metrics:
            final_path = trainer.save_checkpoint("final.pt", metrics={**train_metrics, **eval_metrics})
        logger.log_artifact(final_path, artifact_path="final-checkpoint")
        result = {**train_metrics, **eval_metrics}
        return result
    finally:
        logger.finish(result)


def _restore_wandb_identity_for_resume(cfg) -> None:
    resume_from = cfg.training.get("resume_from", None)
    if resume_from is None or str(resume_from).strip().lower() in {"", "none", "null"}:
        return
    path = Path(str(resume_from))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_run_id = checkpoint.get("wandb_run_id")
    if not saved_run_id:
        return
    configured = cfg.logging.wandb.get("run_id", None)
    if configured is not None and str(configured).strip().lower() not in {"", "none", "null"}:
        if str(configured) != str(saved_run_id):
            raise RuntimeError(
                f"resume W&B run ID mismatch: {configured!r} != {saved_run_id!r}"
            )
        return
    OmegaConf.update(cfg, "logging.wandb.run_id", str(saved_run_id), force_add=True)


def main(argv: Sequence[str] | None = None):
    cfg = load_config(argv=argv)
    return run(cfg)


if __name__ == "__main__":
    main()

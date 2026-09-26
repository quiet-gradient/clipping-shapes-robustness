from __future__ import annotations

from typing import Sequence

from trustml.core.config import load_config
from trustml.core.seed import seed_everything
from trustml.data.cifar10 import build_cifar10_datamodule
from trustml.defenses.dp_sgd import DPSGDDefense
from trustml.defenses.trades import TRADESDefense
from trustml.logging.local_logger import LocalLogger
from trustml.models.resnet18 import build_resnet18
from trustml.training.trainer import Trainer


def _build_defenses(cfg) -> list:
    builders = {"dp_sgd": DPSGDDefense, "trades": TRADESDefense}
    defenses = []
    for item in cfg or []:
        if not bool(item.get("enabled", True)):
            continue
        name = str(item.name).lower()
        if name not in builders:
            raise ValueError(f"Unsupported defense in the reproduction package: {item.name}")
        defenses.append(builders[name](item))
    return defenses


def run(cfg):
    seed_everything(
        int(cfg.experiment.get("seed", 42)),
        deterministic=bool(cfg.experiment.get("deterministic", False)),
    )
    if str(cfg.data.name).lower() != "cifar10":
        raise ValueError("The reproduction package supports only CIFAR-10 fine-tuning.")
    logger = LocalLogger(cfg)
    result = {}
    try:
        logger.log_config(cfg)
        data = build_cifar10_datamodule(cfg.data)
        model = build_resnet18(cfg.model)
        defenses = _build_defenses(cfg.get("defenses", []))

        trainer = Trainer(
            cfg=cfg,
            model=model,
            data=data,
            defenses=defenses,
            logger=logger,
        )
        train_metrics = trainer.fit()
        eval_metrics = trainer.evaluate()
        final_path = trainer.save_checkpoint(
            "final.pt", metrics={**train_metrics, **eval_metrics}
        )
        logger.log_artifact(final_path, artifact_path="final-checkpoint")
        result = {**train_metrics, **eval_metrics}
        return result
    finally:
        logger.finish(result)
def main(argv: Sequence[str] | None = None):
    cfg = load_config(argv=argv)
    return run(cfg)


if __name__ == "__main__":
    main()

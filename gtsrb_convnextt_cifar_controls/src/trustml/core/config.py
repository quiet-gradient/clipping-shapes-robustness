from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig, OmegaConf


def parse_cli(argv: Sequence[str] | None = None) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(description="Run a trustml experiment.")
    parser.add_argument("--config", required=True, help="Path to an OmegaConf YAML config.")
    args, overrides = parser.parse_known_args(argv)
    return Path(args.config), list(overrides)


def load_config(
    config_path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    argv: Sequence[str] | None = None,
) -> DictConfig:
    if config_path is None:
        config_path, cli_overrides = parse_cli(argv)
    else:
        config_path = Path(config_path)
        cli_overrides = list(overrides or [])

    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        _apply_dotlist_overrides(cfg, cli_overrides)

    OmegaConf.resolve(cfg)
    return cfg


def config_to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path), resolve=True)


def _apply_dotlist_overrides(cfg: DictConfig, overrides: Iterable[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form, got: {override}")
        key, raw_value = override.split("=", 1)
        parsed = OmegaConf.from_dotlist([f"value={raw_value}"])
        OmegaConf.update(cfg, key, parsed.value, merge=True)

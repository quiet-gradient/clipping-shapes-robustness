from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def capture_rng_state(
    dataloader_generator: torch.Generator,
    trades_generator: torch.Generator,
    pgd_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader": dataloader_generator.get_state(),
        "trades": trades_generator.get_state(),
        "pgd": pgd_generator.get_state(),
    }


def restore_rng_state(
    state: dict[str, Any],
    dataloader_generator: torch.Generator,
    trades_generator: torch.Generator,
    pgd_generator: torch.Generator,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    dataloader_generator.set_state(state["dataloader"])
    trades_generator.set_state(state["trades"])
    pgd_generator.set_state(state["pgd"])


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    device_type = torch.device(device).type
    generator = torch.Generator(device=device_type)
    generator.manual_seed(int(seed))
    return generator

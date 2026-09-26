from __future__ import annotations

import torch

from trustml.defenses.base import Defense
class DPSGDDefense(Defense):
    name = "dp_sgd"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.epsilon = float(cfg.get("epsilon", 8.0))
        self.delta = float(cfg.get("delta", 1e-5))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.noise_multiplier = cfg.get("noise_multiplier", None)
        self.accountant = str(cfg.get("accountant", "rdp"))
        self.secure_mode = bool(cfg.get("secure_mode", False))
        self.convert_batchnorm = bool(cfg.get("convert_batchnorm", True))
        self.poisson_sampling = bool(cfg.get("poisson_sampling", False))
        self.c_task_budget_fraction = float(cfg.get("c_task_budget_fraction", 0.6))
        self.c_regularizer_clip_norm = float(cfg.get("c_regularizer_clip_norm", 1.0))
        # Naive per-channel calibration composes below the target epsilon (the
        # joint RDP accountant is tighter than eps_task + eps_reg), so strategy C
        # historically under-spent the budget. When enabled, both sigmas are
        # scaled by a common factor so the *joint* accountant hits the target.
        self.c_joint_epsilon_calibration = bool(cfg.get("c_joint_epsilon_calibration", True))
        self.componentwise_one_release = bool(cfg.get("componentwise_one_release", False))
        self.q_regularizer = float(
            cfg.get("q_regularizer", cfg.get("componentwise_q_r", 0.5))
        )
        self.componentwise_q_r = self.q_regularizer
        self.sensitivity_bound = float(cfg.get("sensitivity_bound", self.max_grad_norm))
        self.auto_s = bool(cfg.get("auto_s", False))
        self.auto_s_gamma = float(cfg.get("auto_s_gamma", 0.01))
        self.auto_s_componentwise = False
        self.aggregation_denominator = str(
            cfg.get("aggregation_denominator", "opacus_expected")
        ).lower()
        self.dp_accounting_steps_per_batch = 1
        self.sample_rate = 0.0
        self.regularizer_name = None
        self.dp_strategy = None
        self.d_noise_multiplier = None
        self.active_noise_multiplier = None
        self._gradient_probe_noise_cache = None
        self.privacy_engine = None

    def setup(self, model, optimizer, train_loader, cfg):
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
        except ImportError as exc:
            raise RuntimeError("DP-SGD requires opacus. Install it or disable dp_sgd.") from exc

        device = next(model.parameters()).device
        _disable_inplace_relu(model)
        if self.convert_batchnorm:
            replaced_batchnorm = _replace_batchnorm_with_groupnorm(model)
            if replaced_batchnorm:
                optimizer = _rebuild_optimizer_like(optimizer, _trainable_params(model))
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model).to(device)
            _disable_inplace_relu(model)
            optimizer = _rebuild_optimizer_like(optimizer, _trainable_params(model))

        # BatchNorm->GroupNorm replacement builds fresh GroupNorm modules on CPU
        # and only copies the old weights in; if the model is already Opacus-valid
        # after that, the ModuleValidator.fix(...).to(device) branch above is
        # skipped and those GroupNorm params stay on CPU while the rest is on GPU
        # (device-mismatch crash on the first forward). Re-pin the whole module.
        model = model.to(device)

        privacy_engine = PrivacyEngine(accountant=self.accountant, secure_mode=self.secure_mode)
        self.sample_rate = 1.0 / max(1, len(train_loader))
        if self.noise_multiplier is None:
            raise ValueError("released configurations must pin noise_multiplier")
        separate = _separate_dp_regularizer(cfg)
        self.regularizer_name, self.dp_strategy = separate if separate is not None else (None, None)
        use_separate_dp_channels = self.regularizer_name is not None

        if self.auto_s:
            if self.auto_s_gamma <= 0.0:
                raise ValueError("auto_s_gamma must be positive")
            # v / (||v|| + gamma) has strict L2 norm below one.
            self.max_grad_norm = 1.0
            self.sensitivity_bound = 1.0

        if self.componentwise_one_release:
            if self.regularizer_name != "trades" or self.dp_strategy != "C":
                raise ValueError(
                    "componentwise_one_release requires enabled TRADES with dp_strategy=C"
                )
            beta = _trades_beta(cfg)
            if beta <= 0.0:
                raise ValueError("componentwise_one_release requires beta > 0")
            if self.sensitivity_bound <= 0.0:
                raise ValueError("sensitivity_bound must be positive")
            if not 0.0 < self.componentwise_q_r < 1.0:
                raise ValueError("componentwise_q_r must be between 0 and 1")
            self.auto_s_componentwise = self.auto_s
            if self.auto_s_componentwise:
                if abs(self.sensitivity_bound - 1.0) > 1e-12:
                    raise ValueError("AUTO-S componentwise sensitivity_bound must equal one")
                expected_task_cap = 1.0 - self.componentwise_q_r
                expected_robust_cap = self.componentwise_q_r
                configured_task_cap = float(
                    self.cfg.get("task_sensitivity_cap", expected_task_cap)
                )
                configured_robust_cap = float(
                    self.cfg.get("robust_sensitivity_cap", expected_robust_cap)
                )
                if abs(configured_task_cap - expected_task_cap) > 1e-12:
                    raise ValueError("task_sensitivity_cap must equal 1 - q_regularizer")
                if abs(configured_robust_cap - expected_robust_cap) > 1e-12:
                    raise ValueError("robust_sensitivity_cap must equal q_regularizer")
                # Opacus' one release is calibrated at the total unit bound;
                # branch caps are applied manually before their sums are joined.
                self.max_grad_norm = 1.0
            else:
                self.max_grad_norm, self.c_regularizer_clip_norm = componentwise_clip_bounds(
                    beta=beta,
                    sensitivity_bound=self.sensitivity_bound,
                    q_r=self.componentwise_q_r,
                )

        if use_separate_dp_channels:
            if not self.componentwise_one_release:
                raise ValueError(
                    "TRADES dp_strategy=C requires componentwise_one_release in the released matrix"
                )
            if self.c_regularizer_clip_norm <= 0.0:
                raise ValueError("c_regularizer_clip_norm must be positive.")

        if use_separate_dp_channels and self.componentwise_one_release:
            # Componentwise task and robustness contributions are clipped
            # separately and privatized by one Gaussian release per batch.
            self.d_noise_multiplier = float(self.noise_multiplier)
            self.dp_accounting_steps_per_batch = 1
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=float(self.d_noise_multiplier),
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=self.poisson_sampling,
            )

        else:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=float(self.noise_multiplier),
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=self.poisson_sampling,
            )
        if self.aggregation_denominator == "logical_batch_size":
            optimizer.expected_batch_size = int(cfg.data.batch_size)
        elif self.aggregation_denominator != "opacus_expected":
            raise ValueError(
                "aggregation_denominator must be 'logical_batch_size' or 'opacus_expected'"
            )

        self.privacy_engine = privacy_engine
        self.active_noise_multiplier = float(getattr(optimizer, "noise_multiplier", 0.0))
        self.setup_metrics = {
            "dp/enabled": 1.0,
            "dp/target_epsilon": self.epsilon,
            "dp/target_delta": self.delta,
            "dp/max_grad_norm": self.max_grad_norm,
            "dp/noise_multiplier": float(getattr(optimizer, "noise_multiplier", 0.0)),
            "dp/sensitivity_bound": float(self.sensitivity_bound),
            "dp/absolute_noise_std_before_averaging": float(
                getattr(optimizer, "noise_multiplier", 0.0)
            )
            * float(self.sensitivity_bound),
            "dp/poisson_sampling": 1.0 if self.poisson_sampling else 0.0,
            "DP target epsilon": self.epsilon,
            "DP target delta": self.delta,
            "DP clip norm": self.max_grad_norm,
            "DP sample rate": self.sample_rate,
            "DP main noise multiplier": float(getattr(optimizer, "noise_multiplier", 0.0)),
            "DP accounting steps per batch": float(self.dp_accounting_steps_per_batch),
            "DP aggregation denominator": float(optimizer.expected_batch_size),
            "DP aggregation uses logical batch size": (
                1.0 if self.aggregation_denominator == "logical_batch_size" else 0.0
            ),
        }
        if use_separate_dp_channels and self.componentwise_one_release:
            if self.auto_s_componentwise:
                self.setup_metrics.update(
                    {
                        # Privacy-safe aliases survive the private-training
                        # metric filter, while the human-readable S-C keys are
                        # retained for local diagnostics and compatibility.
                        "dp/sc_one_release": 1.0,
                        "dp/sc_one_optimizer_step": 1.0,
                        "dp/sc_q_regularizer": self.componentwise_q_r,
                        "dp/sc_task_sensitivity_cap": 1.0 - self.componentwise_q_r,
                        "dp/sc_robust_sensitivity_cap": self.componentwise_q_r,
                        "dp/sc_sensitivity_bound": 1.0,
                        "dp/sc_noise_multiplier": float(self.d_noise_multiplier),
                        "dp/sc_absolute_noise_std_before_averaging": float(
                            self.d_noise_multiplier
                        ),
                        "dp/sc_gamma": self.auto_s_gamma,
                        "S-C one release": 1.0,
                        "S-C one optimizer step": 1.0,
                        "S-C q_regularizer": self.componentwise_q_r,
                        "S-C task sensitivity cap": 1.0 - self.componentwise_q_r,
                        "S-C robust sensitivity cap": self.componentwise_q_r,
                        "S-C sensitivity bound": 1.0,
                        "S-C noise multiplier": float(self.d_noise_multiplier),
                        "S-C absolute noise std": float(self.d_noise_multiplier),
                        "S-C gamma": self.auto_s_gamma,
                    }
                )
            else:
                self.setup_metrics.update(
                    {
                        "H-C one release": 1.0,
                        "H-C noise multiplier": float(self.d_noise_multiplier),
                        "H-C task clip norm": self.max_grad_norm,
                        "H-C TRADES clip norm": self.c_regularizer_clip_norm,
                        "H-C q_r": self.componentwise_q_r,
                        "H-C sensitivity bound": self.sensitivity_bound,
                        "H-C absolute noise std": float(self.d_noise_multiplier)
                        * self.sensitivity_bound,
                    }
                )

        return model, optimizer, train_loader

    def get_epsilon(self) -> float | None:
        if self.privacy_engine is None:
            return None
        return float(self.privacy_engine.get_epsilon(delta=self.delta))

    def gradient_probe_noise_multipliers(self, cfg) -> dict[str, float]:
        """Return matched A/C/D noise multipliers without mutating accounting.

        Gradient-probe diagnostics compare counterfactual clipping mechanisms on
        one *public* batch and one model state.  To make their analytic noise
        terms comparable, they need the multipliers each mechanism would use at
        this run's sample rate, epoch budget, accountant and target ``(eps,
        delta)``.  This helper reuses the active mechanism's exact scalar and
        calibrates only missing counterfactual values with fresh accountant
        objects.  It caches the result, never touches the training accountant
        and never samples a noise tensor.
        """
        if self.privacy_engine is None:
            raise RuntimeError("gradient-probe noise scales require DP-SGD setup first")
        if self._gradient_probe_noise_cache is not None:
            return dict(self._gradient_probe_noise_cache)

        try:
            from opacus.accountants.utils import get_noise_multiplier
        except ImportError as exc:
            raise RuntimeError("Gradient-probe noise calibration requires opacus.") from exc

        mechanism = self.privacy_engine.accountant.mechanism()
        epochs = max(1, int(cfg.training.get("epochs", 1)))
        if self.componentwise_one_release and self.d_noise_multiplier is not None:
            one_channel_sigma = float(self.d_noise_multiplier)
        elif self.dp_strategy != "C" and self.active_noise_multiplier is not None:
            one_channel_sigma = float(self.active_noise_multiplier)
        else:
            one_channel_sigma = float(
                get_noise_multiplier(
                    target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    sample_rate=self.sample_rate,
                    epochs=epochs,
                    accountant=mechanism,
                )
            )

        task_epsilon = self.epsilon * self.c_task_budget_fraction
        regularizer_epsilon = self.epsilon - task_epsilon
        task_delta = self.delta * self.c_task_budget_fraction
        regularizer_delta = self.delta - task_delta
        task_sigma = float(
            get_noise_multiplier(
                target_epsilon=task_epsilon,
                target_delta=task_delta,
                sample_rate=self.sample_rate,
                epochs=epochs,
                accountant=mechanism,
            )
        )
        regularizer_sigma = float(
            get_noise_multiplier(
                target_epsilon=regularizer_epsilon,
                target_delta=regularizer_delta,
                sample_rate=self.sample_rate,
                epochs=epochs,
                accountant=mechanism,
            )
        )
        if self.c_joint_epsilon_calibration:
            steps_per_channel = epochs * max(1, round(1.0 / max(self.sample_rate, 1e-12)))
            scale, _ = _calibrate_joint_sigma_scale(
                sigma_task=task_sigma,
                sigma_regularizer=regularizer_sigma,
                sample_rate=self.sample_rate,
                steps_per_channel=steps_per_channel,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                mechanism=mechanism,
            )
            task_sigma *= float(scale)
            regularizer_sigma *= float(scale)

        result = {
            "A": one_channel_sigma,
            "C_task": task_sigma,
            "C_regularizer": regularizer_sigma,
            # The hard joint and one-release counterfactuals share one multiplier;
            # their sensitivities are accounted for separately by the probe.
            "D": one_channel_sigma,
            "matched_target_budget": 1.0 if self.c_joint_epsilon_calibration else 0.0,
        }
        self._gradient_probe_noise_cache = result
        return dict(result)

    def accountant_metrics(
        self,
        manual_task_passes: int | None = None,
        manual_regularizer_passes: int | None = None,
        manual_d_passes: int | None = None,
    ) -> dict[str, float]:
        if self.privacy_engine is None:
            return {}
        history = getattr(self.privacy_engine.accountant, "history", [])
        total_steps = sum(int(entry[2]) for entry in history if len(entry) == 3)
        metrics = {
            "dp/accountant_history_entries": float(len(history)),
            "dp/accountant_total_steps": float(total_steps),
            "dp/accountant_sample_rate": float(self.sample_rate),
            "dp/sensitivity_bound": float(self.sensitivity_bound),
            "dp/absolute_noise_std": float(self.active_noise_multiplier or 0.0)
            * float(self.sensitivity_bound),
        }
        if self.regularizer_name is None:
            return metrics
        if not self.componentwise_one_release:
            raise RuntimeError("unsupported non-componentwise accounted channel")
        prefix = "S-C" if self.auto_s_componentwise else "H-C"
        metrics.update(
            {
                f"{prefix} accountant history entries": float(len(history)),
                f"{prefix} accountant total steps": float(total_steps),
                f"{prefix} manual DP passes": float(manual_d_passes or 0),
                f"{prefix} noise multiplier": float(self.d_noise_multiplier),
            }
        )
        return metrics


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _separate_dp_regularizer(cfg) -> tuple[str, str] | None:
    """Return the componentwise TRADES channel declared by a released config."""
    for item in cfg.get("defenses", []):
        if not bool(item.get("enabled", True)):
            continue
        name = str(item.get("name", "")).lower()
        if name == "trades" and float(item.get("beta", 0.0)) > 0.0:
            strategy = str(item.get("dp_strategy", "A")).upper()
            if strategy not in {"A", "C"}:
                raise ValueError(f"released TRADES dp_strategy must be A or C, got: {strategy}")
            if strategy == "C":
                return "trades", strategy
    return None


def _trades_beta(cfg) -> float:
    for item in cfg.get("defenses", []):
        if bool(item.get("enabled", True)) and str(item.get("name", "")).lower() == "trades":
            return float(item.get("beta", 0.0))
    return 0.0


def componentwise_clip_bounds(
    *, beta: float, sensitivity_bound: float, q_r: float
) -> tuple[float, float]:
    beta = float(beta)
    sensitivity_bound = float(sensitivity_bound)
    q_r = float(q_r)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if sensitivity_bound <= 0.0:
        raise ValueError("sensitivity_bound must be positive")
    if not 0.0 < q_r < 1.0:
        raise ValueError("q_r must be between 0 and 1")
    return (1.0 - q_r) * sensitivity_bound, q_r * sensitivity_bound / beta


def _calibrate_joint_sigma_scale(
    sigma_task: float,
    sigma_regularizer: float,
    sample_rate: float,
    steps_per_channel: int,
    target_epsilon: float,
    target_delta: float,
    mechanism: str,
    tolerance: float = 1e-3,
    min_scale: float = 0.05,
) -> tuple[float, float]:
    """Find a common factor for both sigmas so the joint accountant hits target epsilon.

    The naive per-channel calibration keeps the sigma ratio between channels; the
    joint composition of both histories is tighter than eps_task + eps_reg, so the
    unscaled pair lands below target. Epsilon is monotone decreasing in the scale,
    so bisection on the scale is safe.
    """
    from opacus.accountants import create_accountant

    def joint_epsilon(scale: float) -> float:
        accountant = create_accountant(mechanism=mechanism)
        accountant.history = [
            (float(sigma_task * scale), float(sample_rate), int(steps_per_channel)),
            (float(sigma_regularizer * scale), float(sample_rate), int(steps_per_channel)),
        ]
        return float(accountant.get_epsilon(delta=target_delta))

    high = 1.0
    while joint_epsilon(high) > target_epsilon:
        high *= 1.5
        if high > 20.0:
            raise RuntimeError("Joint epsilon calibration failed: epsilon above target even at large sigma scale.")
    low = high
    while joint_epsilon(low) < target_epsilon:
        low *= 0.8
        if low < min_scale:
            break
    for _ in range(50):
        mid = 0.5 * (low + high)
        epsilon_mid = joint_epsilon(mid)
        if abs(epsilon_mid - target_epsilon) <= tolerance * target_epsilon:
            return mid, epsilon_mid
        if epsilon_mid > target_epsilon:
            low = mid
        else:
            high = mid
    scale = 0.5 * (low + high)
    return scale, joint_epsilon(scale)


def _disable_inplace_relu(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False


def _replace_batchnorm_with_groupnorm(module: torch.nn.Module) -> bool:
    replaced_any = False
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            groups = _group_count(child.num_features)
            replacement = torch.nn.GroupNorm(num_groups=groups, num_channels=child.num_features)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.detach())
                    replacement.bias.copy_(child.bias.detach())
                replacement.weight.requires_grad = child.weight.requires_grad
                replacement.bias.requires_grad = child.bias.requires_grad
            setattr(module, name, replacement)
            replaced_any = True
        else:
            replaced_any = _replace_batchnorm_with_groupnorm(child) or replaced_any
    return replaced_any


def _group_count(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _rebuild_optimizer_like(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    if not optimizer.param_groups:
        raise ValueError("Cannot rebuild optimizer without param groups.")

    group = optimizer.param_groups[0]
    if isinstance(optimizer, torch.optim.SGD):
        return torch.optim.SGD(
            params,
            lr=group["lr"],
            momentum=group.get("momentum", 0.0),
            dampening=group.get("dampening", 0.0),
            weight_decay=group.get("weight_decay", 0.0),
            nesterov=group.get("nesterov", False),
        )
    raise ValueError("The released experiment matrix requires SGD")

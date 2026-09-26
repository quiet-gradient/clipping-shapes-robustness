from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from omegaconf import OmegaConf
from opacus import PrivacyEngine
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trustml.defenses.dp_sgd import componentwise_clip_bounds
from trustml.training.dp_diagnostics import (
    auto_s_componentwise_transform,
    auto_s_transform,
)
from trustml.training.gradient_probe import (
    auto_s_analytic_jvp,
    validate_componentwise_geometry_identities,
)
from trustml.training.trainer import Trainer


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _Logger:
    def log_metrics(self, metrics, step=None):
        del metrics, step


class _FixedTrades:
    beta = 3.0

    @staticmethod
    def _generate_adversarial(model, value):
        del model
        return value + 0.05

    @staticmethod
    def kl_loss(model, adversarial, clean_probs):
        return F.kl_div(
            F.log_softmax(model(adversarial), dim=1),
            clean_probs,
            reduction="batchmean",
        )


def _private_trainer(root: Path):
    x = torch.tensor([[1.0, 2.0, -1.0], [0.5, -2.0, 1.0], [1.5, 0.0, 0.5]])
    y = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
    model = _TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, _ = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        poisson_sampling=False,
    )
    cfg = OmegaConf.create(
        {
            "training": {"device": "cpu"},
            "data": {"batch_size": 3},
            "logging": {"local": {"dir": str(root)}},
        }
    )
    trainer = Trainer(cfg, model, {}, [], _Logger())
    trainer.optimizer = optimizer
    return trainer, engine, x, y


class MechanismTests(unittest.TestCase):
    def test_hard_component_bounds(self) -> None:
        task, robust = componentwise_clip_bounds(
            beta=3.0, sensitivity_bound=0.7, q_r=0.25
        )
        self.assertAlmostEqual(task + 3.0 * robust, 0.7)
        self.assertAlmostEqual(task, 0.525)
        self.assertAlmostEqual(robust, 0.175 / 3.0)

    def test_auto_s_strict_bound_and_component_caps(self) -> None:
        vectors = torch.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, -2.0]])
        transformed = auto_s_transform(vectors, gamma=0.01)
        self.assertTrue(torch.all(transformed.norm(dim=1) < 1.0))
        task = torch.tensor([[2.0, 0.0], [0.0, 0.2]])
        robust = torch.tensor([[0.0, 1.0], [4.0, 0.0]])
        actual = auto_s_componentwise_transform(
            task,
            robust,
            beta=3.0,
            q_regularizer=0.25,
            gamma=0.01,
        )
        individually = torch.cat(
            [
                auto_s_componentwise_transform(
                    task[index : index + 1],
                    robust[index : index + 1],
                    beta=3.0,
                    q_regularizer=0.25,
                    gamma=0.01,
                )
                for index in range(len(task))
            ]
        )
        self.assertTrue(torch.allclose(actual, individually))
        self.assertTrue(torch.all(actual.norm(dim=1) <= 1.0 + 1e-6))

    def test_fp64_geometry_and_jvp(self) -> None:
        generator = torch.Generator().manual_seed(20260827)
        task = 3.0 * torch.randn(64, 17, generator=generator, dtype=torch.float64)
        robust = 2.0 * torch.randn(64, 17, generator=generator, dtype=torch.float64)
        task[0].zero_()
        robust[1].zero_()
        robust[2] = -task[2]
        audit = validate_componentwise_geometry_identities(
            task, robust, q=1.0 / 3.0, sensitivity_bound=1.0
        )
        self.assertLess(audit["projection_max_negative_slack"], 2e-13)
        self.assertLess(audit["rotation_residual_max"], 2e-12)
        x = torch.randn(31, generator=generator, dtype=torch.float64, requires_grad=True)
        direction = torch.randn(31, generator=generator, dtype=torch.float64)
        for gamma in (0.01, 0.1, 1.0):
            analytic = auto_s_analytic_jvp(x.detach(), direction, gamma=gamma)
            _, automatic = torch.autograd.functional.jvp(
                lambda value: value / (value.norm() + gamma),
                x,
                direction,
                strict=True,
            )
            self.assertTrue(torch.allclose(analytic, automatic, atol=2e-13, rtol=2e-13))

    def test_s_joint_and_s_componentwise_release_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(43)
            trainer, engine, x, y = _private_trainer(Path(directory))
            trainer.optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(trainer.model(x), y).backward()
            metrics = trainer._apply_auto_s_update(SimpleNamespace(auto_s_gamma=0.01))
            self.assertLess(metrics["output_norm_max"], 1.0)
            self.assertEqual(sum(int(entry[2]) for entry in engine.accountant.history), 1)

            torch.manual_seed(44)
            trainer, engine, x, y = _private_trainer(Path(directory))
            _, _, metrics = trainer._auto_s_componentwise_trades_step(
                x,
                y,
                SimpleNamespace(
                    componentwise_q_r=0.25,
                    auto_s_gamma=0.01,
                    d_noise_multiplier=0.0,
                ),
                _FixedTrades(),
            )
            self.assertLess(metrics["S-C task output norm max"], 0.75)
            self.assertLess(metrics["S-C robust output norm max"], 0.25)
            self.assertEqual(metrics["S-C accountant steps this batch"], 1.0)
            self.assertEqual(metrics["S-C optimizer steps this batch"], 1.0)
            self.assertEqual(sum(int(entry[2]) for entry in engine.accountant.history), 1)

    def test_hard_componentwise_release_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(45)
            trainer, engine, x, y = _private_trainer(Path(directory))
            _, _, metrics = trainer._componentwise_trades_step(
                x,
                y,
                SimpleNamespace(
                    max_grad_norm=0.525,
                    c_regularizer_clip_norm=0.175 / 3.0,
                    d_noise_multiplier=0.0,
                    sensitivity_bound=0.7,
                ),
                _FixedTrades(),
            )
            self.assertAlmostEqual(metrics["H-C sensitivity"], 0.7)
            self.assertEqual(sum(int(entry[2]) for entry in engine.accountant.history), 1)


if __name__ == "__main__":
    unittest.main()

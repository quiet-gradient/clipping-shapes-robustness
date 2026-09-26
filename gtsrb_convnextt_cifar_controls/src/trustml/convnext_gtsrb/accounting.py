"""Exact GTSRB image-Poisson rate/horizon, independently checked per-run DP."""
import math
from importlib.metadata import version

ORDERS = [1.1 + i / 10 for i in range(99)] + list(range(12, 64)) + [96, 128, 256, 512, 1024]


def horizon(n_private):
    if n_private <= 512:
        raise ValueError("insufficient private data")
    rate = 512 / n_private
    steps = math.floor(50 * n_private / 512 + .5)
    return {"N_private": n_private, "sample_rate": rate, "steps": steps,
            "expected_epochs": steps * rate, "warmup_steps": math.floor(.1 * steps + .5),
            "checkpoint_steps": {str(e): (steps if e == 50 else math.floor(e * n_private / 512 + .5))
                                 for e in [0, 5, 10, 20, 30, 40, 50]}}


def bounds(sigma, n_private, *, with_prv=True):
    from opacus.accountants.analysis import rdp
    from opacus.accountants import PRVAccountant
    h = horizon(n_private)
    values = rdp.compute_rdp(q=h["sample_rate"], noise_multiplier=sigma, steps=h["steps"], orders=ORDERS)
    epsilon, order = rdp.get_privacy_spent(orders=ORDERS, rdp=values, delta=1e-5)
    result = {**h, "noise_multiplier": sigma, "rdp_epsilon_upper": float(epsilon), "rdp_optimal_order": float(order),
              "orders": ORDERS, "delta": 1e-5, "target_epsilon": 4., "adjacency": "add_remove", "privacy_unit": "image_not_track",
              "aggregation_denominator": 512, "releases_per_step": 1, "opacus_version": version("opacus"),
              "guarantee_scope": "one fixed-hyperparameter training run; NOT whole pilot/HPO/selection"}
    if with_prv:
        prv = PRVAccountant()
        prv.history = [(sigma, h["sample_rate"], h["steps"])]
        result.update(prv_epsilon_upper=float(prv.get_epsilon(delta=1e-5, eps_error=.1, delta_error=1e-8)),
                      prv_eps_error=.1, prv_delta_error=1e-8)
    result["status"] = "PASS" if max(result["rdp_epsilon_upper"], result.get("prv_epsilon_upper", 0)) <= 4 else "FAIL"
    return result


def calibrate(n_private):
    from opacus.accountants.utils import get_noise_multiplier
    h = horizon(n_private)
    sigma = get_noise_multiplier(target_epsilon=3.999, target_delta=1e-5, sample_rate=h["sample_rate"],
                                steps=h["steps"], accountant="rdp", epsilon_tolerance=.001, alphas=ORDERS)
    result = bounds(float(sigma), n_private)
    if result["status"] != "PASS":
        raise RuntimeError("independent RDP/PRV check failed; explicit accounting review required")
    return result

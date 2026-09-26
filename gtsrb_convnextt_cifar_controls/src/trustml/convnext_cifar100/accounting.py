"""A conservative calibration shared by all 45 fixed-hyperparameter runs."""
from __future__ import annotations

import importlib.metadata

SAMPLE_RATE = 512 / 45000
STEPS = 1760
ORDERS = [1.1 + i/10 for i in range(99)] + list(range(12,64)) + [96,128,256,512,1024]


def bounds(sigma, *, steps=STEPS, with_prv=True):
    from opacus.accountants.analysis import rdp
    from opacus.accountants import PRVAccountant
    values = rdp.compute_rdp(q=SAMPLE_RATE, noise_multiplier=sigma, steps=steps, orders=ORDERS)
    eps, order = rdp.get_privacy_spent(orders=ORDERS, rdp=values, delta=1e-5)
    result = dict(noise_multiplier=sigma, sampling_probability=SAMPLE_RATE, steps=steps,
                  delta=1e-5, target_epsilon=4., rdp_epsilon_upper=float(eps),
                  rdp_optimal_order=float(order), orders=ORDERS, adjacency="add_remove",
                  aggregation_denominator=512, releases_per_step=1,
                  opacus_version=importlib.metadata.version("opacus"),
                  guarantee_scope="one fixed-hyperparameter run, NOT whole HPO or correlated pairs")
    if with_prv:
        prv = PRVAccountant()
        prv.history = [(sigma, SAMPLE_RATE, steps)]
        # 0.01 and 0.05 trigger Opacus' discrete/continuous mean consistency
        # guard at this sample rate/sigma. Keep the guard; use the certified
        # upper endpoint with an explicitly recorded wider numerical interval.
        result.update(prv_epsilon_upper=float(prv.get_epsilon(delta=1e-5, eps_error=.1, delta_error=1e-8)),
                      prv_eps_error=.1, prv_delta_error=1e-8)
        result["status"] = "PASS" if max(eps,result["prv_epsilon_upper"]) <=4 else "FAIL"
    return result


def calibrate():
    from opacus.accountants.utils import get_noise_multiplier
    sigma = get_noise_multiplier(target_epsilon=3.999, target_delta=1e-5,
        sample_rate=SAMPLE_RATE, steps=STEPS, accountant="rdp", epsilon_tolerance=.001,
        alphas=ORDERS)
    result = bounds(float(sigma))
    while result["status"] != "PASS":
        sigma *= 1.001
        result = bounds(float(sigma))
    return result

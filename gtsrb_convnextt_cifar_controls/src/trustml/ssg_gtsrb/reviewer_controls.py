"""Frozen reviewer controls: uniform cJ and zero-noise J/U, one model per job."""
from pathlib import Path

from .confirmation import run as run_confirmation
from .private import is_private_run, uses_bounded_query
from .rectangular import SOURCE_SHA


CAMPAIGN = "gtsrb_reviewer_controls_20260925"
SEEDS = (45, 46, 47)
UNIFORM_JOINT_SCALE = 0.8140694935
ANCHOR = {"C": 1.0, "lr": 0.025, "q": 1 / 3, "beta": 1}


def _base(seed):
    return dict(
        optimizer="SGD",
        C=ANCHOR["C"],
        lr=ANCHOR["lr"],
        beta=ANCHOR["beta"],
        source_checkpoint_sha256=SOURCE_SHA,
        seed_label=seed,
        outer_kl_target="full",
        clipping_map="hard",
        rng_coupling="matched_secret_group",
        coupling_group=f"hard_dmr_s{seed}",
        joint_release_dp_claim=False,
    )


def manifest():
    groups = []
    for seed in SEEDS:
        common = _base(seed)
        rows = [
            dict(
                common,
                run_id=f"review_uniform_cj_s{seed}",
                role="Joint",
                q=None,
                private=True,
                control_kind="uniform_attenuation",
                uniform_joint_scale=UNIFORM_JOINT_SCALE,
                matched_existing_runs=[f"confirm_hard_j_s{seed}", f"confirm_hard_r_s{seed}"],
            ),
            dict(
                common,
                run_id=f"review_sigma0_j_s{seed}",
                role="Joint",
                q=None,
                private=False,
                control_kind="zero_noise",
                zero_noise_control=True,
            ),
            dict(
                common,
                run_id=f"review_sigma0_u_s{seed}",
                role="Componentwise",
                q=ANCHOR["q"],
                private=False,
                control_kind="zero_noise",
                zero_noise_control=True,
                matched_joint_run_id=f"review_sigma0_j_s{seed}",
            ),
        ]
        for row in rows:
            if not uses_bounded_query(row):
                raise RuntimeError("reviewer control must use a bounded per-record query")
            expected_private = row["control_kind"] == "uniform_attenuation"
            if is_private_run(row) is not expected_private:
                raise RuntimeError("reviewer control privacy label mismatch")
            groups.append(dict(group=row["run_id"], runs=[row], coupled=False))
    return dict(
        campaign=CAMPAIGN,
        training_runs=9,
        groups=groups,
        seeds=list(SEEDS),
        source_checkpoint_sha256=SOURCE_SHA,
        anchor=dict(ANCHOR),
        uniform_joint_scale=UNIFORM_JOINT_SCALE,
        uniform_scale_calibration="sum_i ||R_i|| / sum_i ||J_i|| on fixed 512-record public initialization",
        full_dev_aa_after_each_training=True,
        official_test_allowed=False,
        confirmation_reselection_allowed=False,
        one_model_per_parent=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--input-archive", type=Path)
    parser.add_argument("--rng-input", type=Path, required=True)
    parser.add_argument("--mlm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/reviewer_control"))
    args = parser.parse_args()
    args.layout = "single_controls"
    run_confirmation(args, plan=manifest())


if __name__ == "__main__":
    main()

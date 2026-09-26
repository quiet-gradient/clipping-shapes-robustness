"""Revised LR grid: 54 DP trainings plus three unclipped, non-DP CE controls.

The original rect54 module/manifests remain historical execution evidence.
This revision is prepared only; backend migration requires reconciliation.
"""
import argparse
import json
from pathlib import Path

from . import rectangular as previous

CAMPAIGN = "ssg_gtsrb_rect57_v2"
LRS = [("lr0025", .025), ("lr0050", .05), ("lr0100", .1)]


def manifest():
    groups = []
    # Baselines first. No C sweep for an unclipped optimizer.
    for tag, lr in LRS:
        row = dict(run_id=f"rect57_natural_{tag}_s42", role="Natural", optimizer="SGD",
                   C=None, lr=lr, q=None, beta=0, private=False,
                   source_checkpoint_sha256=previous.SOURCE_SHA, seed_label=42)
        groups.append(dict(group=f"natural_{tag}", backend="unassigned", runs=[row]))
    for ctag, c in previous.CS:
        for ltag, lr in LRS:
            key = f"{ctag}_{ltag}"
            joint = f"rect57_joint_{key}_s42"
            rows = []
            for arm, qtag, q in [("CleanDP", "", None), ("Joint", "", None)] + [
                    ("Componentwise", tag, q) for tag, q in previous.QS]:
                row = dict(run_id=f"rect57_{arm.lower()}_{key}" + (f"_{qtag}" if qtag else "") + "_s42",
                           role=arm, optimizer="SGD", C=c, lr=lr, q=q,
                           beta=0 if arm == "CleanDP" else 1, private=True,
                           source_checkpoint_sha256=previous.SOURCE_SHA, seed_label=42)
                if arm == "Componentwise":
                    row["matched_joint_run_id"] = joint
                rows.append(row)
            groups.append(dict(group=key, backend="unassigned", runs=rows))
    return dict(campaign=CAMPAIGN, revision=2, status="PREPARED_NOT_SUBMITTED",
                supersedes="ssg_gtsrb_rect54", source_epoch=50,
                source_sha256=previous.SOURCE_SHA, total_training_runs=57,
                private_training_runs=54, nonprivate_training_runs=3,
                gpu_per_job=1, groups=groups, official_test_allowed=False)


def protocol():
    p = previous.protocol()
    p.update(campaign=CAMPAIGN, status="PROTOCOL_SPECIFICATION")
    p["sgd"]["learning_rates"] = [lr for _, lr in LRS]
    p["grid_per_optimizer"].update(Natural=3)
    p["execution"].update(max_parent_hours=72, backend_assignment="PENDING_RECONCILIATION")
    from trustml.convnext_gtsrb.model import sha256
    audit_path = Path("configs/ssg_gtsrb_rect57_v2/public_microbatch_audit.json")
    if sha256(audit_path) != "27048c2922970e869b834b95670fc91a4c0cff8c02e922201f5bed72a933e936":
        raise ValueError("public microbatch audit changed")
    p["execution"].update(microbatch_audit=json.loads(audit_path.read_text()),
                          microbatch_policy="fastest_common_safe",
                          microbatch_audit_job_id="redacted-public-audit",
                          nonprivate_microbatch="same as validated CleanDP; public preflight required")
    p["nonprivate"] = dict(objective="CE", clipping=False, gaussian_noise=False,
                          accountant=False, epsilon=None, delta=None,
                          sampler="same_poisson_as_DP", aggregation_denominator=512,
                          optimizer_steps=3436, learning_rates=[lr for _, lr in LRS],
                          selection="maximum final-iterate public-dev clean; tie: lower LR",
                          privacy_guarantee="NONE; training uses the real-GTSRB training split")
    return p


def transport_group(name):
    if name == "natural_all":
        return dict(group=name, backend="unassigned",
                    runs=[g["runs"][0] for g in manifest()["groups"][:3]])
    return next(g for g in manifest()["groups"] if g["group"] == name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--input-archive", type=Path)
    parser.add_argument("--selected-archive", type=Path)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--output", type=Path, default=Path("outputs/rect57"))
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(manifest(), indent=2))
    elif args.group and args.routes:
        import sys
        previous.run(args, plan=sys.modules[__name__])
    else:
        parser.error("use --dry-run or provide --group and --routes")

"""Deterministic downstream grids and anchors. No source-dependent C guessed."""
import math

from trustml.convnext_gtsrb.data import digest


ALLOCATIONS = [("q010", .1), ("q025", .25), ("q1of3", 1/3), ("q050", .5)]
LEARNING_RATES = {"SGD": [.01, .03, .1], "AdamW": [.0001, .00045, .002]}


def initial_grid(optimizer, calibration, source_hash):
    if optimizer not in LEARNING_RATES or not source_hash:
        raise ValueError("locked optimizer and selected public source required")
    if calibration.get("status") != "PASS" or calibration.get("records") != 512:
        raise ValueError("complete new public calibration required")
    rows = []
    for arm in ["CleanDP", "Joint"]:
        bounds = ([calibration["C_CE"] / 2, calibration["C_CE"]] if arm == "CleanDP"
                  else [calibration["C_low"], calibration["C_mid"]])
        if any(not math.isfinite(c) or c <= 0 for c in bounds):
            raise ValueError("invalid calibrated bounds")
        for c_name, bound in zip(["low", "mid"], bounds):
            for i, lr in enumerate(LEARNING_RATES[optimizer]):
                rows.append({"run_id": f"{optimizer.lower()}_{arm.lower()}_{c_name}_lr{i}_s42",
                             "optimizer": optimizer, "role": arm, "C": bound, "lr": lr, "q": None,
                             "source_checkpoint_sha256": source_hash, "seed_label": 42})
    return rows


def pareto(rows):
    for row in rows:
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 100 for k in ["dev_clean", "pgd20"]):
            raise ValueError("invalid development accuracy")
    return [r for r in rows if not any(
        o["dev_clean"] >= r["dev_clean"] and o["pgd20"] >= r["pgd20"] and
        (o["dev_clean"] > r["dev_clean"] or o["pgd20"] > r["pgd20"]) for o in rows)]


def anchors(joint_results):
    if len(joint_results) != 6 or len({r["run_id"] for r in joint_results}) != 6:
        raise ValueError("six complete distinct Joint results required before anchor selection")
    if any(r["role"] != "Joint" or r["status"] != "PASS" for r in joint_results):
        raise ValueError("anchor selection cannot use incomplete/non-Joint results")
    if len({r["optimizer"] for r in joint_results}) != 1:
        raise ValueError("do not pool optimizers during anchor selection")
    frontier = pareto(joint_results)
    first = min(frontier, key=lambda r: (-r["dev_clean"], -r["pgd20"], r["run_id"]))
    second = min(frontier, key=lambda r: (-r["pgd20"], -r["dev_clean"], r["run_id"]))
    return [first] if first["run_id"] == second["run_id"] else [first, second]


def component_grid(selected_anchors):
    rows = []
    for anchor in selected_anchors:
        for label, q in ALLOCATIONS:
            # Exact C/LR/init/optimizer inherited, no independent q calibration.
            config = dict(anchor["scientific_config"])
            config.update(run_id=anchor["run_id"].replace("joint", "componentwise") + "_" + label,
                          role="Componentwise", q=q, matched_joint_run_id=anchor["run_id"])
            rows.append(config)
    return rows


def coupling_key(row):
    """Ignore q/arm for TRADES so each J/U pair receives identical streams."""
    return digest({"optimizer": row["optimizer"], "C": row["C"], "lr": row["lr"],
                   "source": row["source_checkpoint_sha256"], "seed_label": row["seed_label"],
                   "objective": "nonprivate_CE" if row["role"] == "Natural" else
                                "CE" if row["role"] == "CleanDP" else "TRADES_beta1"})

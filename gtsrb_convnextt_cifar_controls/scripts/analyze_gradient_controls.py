"""Recompute the statistics in Appendix E."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "reports/gradient_controls"
ACCURACY_KEYS = ("dataset", "clipping", "method", "epsilon_DP")
TRAJECTORY_KEYS = ("dataset", "clipping", "trajectory_method", "step", "epoch")


def read_json(directory, name):
    return json.loads((directory / name).read_text())


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def summarize(rows, keys, metrics, include_range=False):
    result = []
    groups = group_rows(rows, keys)
    for values, group in sorted(groups.items(), key=lambda item: str(item[0])):
        seeds = sorted(row["seed"] for row in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed in group {values}")
        row = dict(zip(keys, values))
        row.update(n=len(group), seeds=",".join(map(str, seeds)))
        for metric in metrics:
            samples = [item[metric] for item in group]
            row[f"{metric}_mean"] = statistics.mean(samples)
            row[f"{metric}_sd"] = statistics.stdev(samples)
            if include_range:
                row[f"{metric}_min"] = min(samples)
                row[f"{metric}_max"] = max(samples)
        result.append(row)
    return result


def accuracy_differences(rows):
    differences = []
    keys = ("dataset", "clipping", "epsilon_DP")
    for values, group in group_rows(rows, keys).items():
        methods = group_rows(group, ("method",))
        for left, right in (("M", "N"), ("M", "U"), ("cM", "U")):
            if (left,) not in methods or (right,) not in methods:
                continue
            a = sorted(methods[(left,)], key=lambda row: row["seed"])
            b = sorted(methods[(right,)], key=lambda row: row["seed"])
            if [row["seed"] for row in a] != [row["seed"] for row in b]:
                raise ValueError(f"Different seed sets for {values}: {left} and {right}")
            for x, y in zip(a, b):
                differences.append({
                    **dict(zip(keys, values)),
                    "comparison": f"{left}-{right}",
                    "seed": x["seed"],
                    "clean": x["clean"] - y["clean"],
                    "aa": x["aa"] - y["aa"],
                })
    return summarize(differences, (*keys, "comparison"), ("clean", "aa"))


def pareto_frontier(accuracy):
    rows = [row for row in accuracy if row["epsilon_DP"] == 4
            and row["method"] in ("J", "U", "D", "M", "R", "N")]
    result = []
    for group in group_rows(rows, ("dataset", "clipping")).values():
        for row in group:
            dominated = any(
                other["clean_mean"] >= row["clean_mean"]
                and other["aa_mean"] >= row["aa_mean"]
                and (other["clean_mean"] > row["clean_mean"]
                     or other["aa_mean"] > row["aa_mean"])
                for other in group
            )
            result.append({**row, "on_frontier": not dominated})
    return result


def calculate(data_dir):
    accuracy = read_json(data_dir, "accuracy.json")
    rows = accuracy["new_results"] + accuracy["original_matched_results"]
    test_rows = [row for row in rows if row["split"] == "test"]
    test_summary = summarize(
        test_rows + accuracy["historical_selected_baselines"],
        ACCURACY_KEYS, ("clean", "aa"),
    )
    norms = read_json(data_dir, "public_norm_trajectories.json")
    scaled = read_json(data_dir, "scaled_M_geometry.json")["rows"]
    radial = read_json(data_dir, "uniform_J_geometry.json")["rows"]
    return {
        "accuracy": test_summary,
        "differences": accuracy_differences(test_rows),
        "pareto": pareto_frontier(test_summary),
        "nonprivate_search": [
            {key: row[key] for key in ("seed", "lr", "clean", "aa")}
            for row in sorted(rows, key=lambda row: row.get("lr", 0))
            if row["split"] == "dev" and row["method"] == "nonprivate"
        ],
        "own_norm_trajectories": summarize(
            norms["own_trajectory_ratios"], TRAJECTORY_KEYS, ("actual_over_trained_J",),
        ),
        "same_state_norm_ratios": summarize(
            norms["same_state"], TRAJECTORY_KEYS, ("U_over_J", "M_over_J", "M_over_U"),
        ),
        "scaled_M": summarize(
            scaled, ("step", "epoch"),
            ("r_same_theta", "r_actual_trained_cM_vs_historical_U", "angle_U_M_degrees",
             "equal_norm_relative_residual", "L2_optimal_relative_residual"),
            include_range=True,
        ),
        "uniform_J": summarize(
            radial, ("step", "epoch"),
            ("cJ_to_R_norm_ratio", "angle_J_R_degrees", "cJ_R_relative_vector_residual"),
        ),
    }


def write_tables(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/gradient_controls")
    args = parser.parse_args()
    tables = calculate(args.data_dir)
    write_tables(tables, args.output_dir)
    print(f"Wrote {len(tables)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Reports from validated public diagnostics and sanitized optimizer telemetry."""
import html
import json
from pathlib import Path

import numpy as np

from trustml.convnext_cifar100.preflight import write_json
from .stages import pareto, LEARNING_RATES


def public_reports(output, selected, curves, aa_rows):
    lines = ["# Public SSG TRADES pretraining", "", "100 epochs; both renderers; 34,400 train / 8,600 validation.", "",
             "| Epoch | Step | Train CE | Train KL | Clean % | PGD10 % |", "|---|---:|---:|---:|---:|---:|"]
    for row in curves:
        lines.append(f"| {row['epoch']} | {row['optimizer_step']} | {row['train_ce']:.6f} | {row['train_kl']:.6f} | "
                     f"{row.get('clean_accuracy', '—')} | {row.get('pgd10_accuracy', '—')} |")
    (output / "ssg_convnext_trades_pretrain_curves.md").write_text("\n".join(lines) + "\n")
    lines = ["# Public checkpoint selection", "", "Highest clean on clean/AA Pareto frontier within 1 pp of maximum AA.",
             "The 1 pp allowance is a decision rule, NOT a confidence interval. No real-dev/private/test selection.", "",
             "| Epoch | Clean % | PGD10 % | AA % | Selected |", "|---|---:|---:|---:|---|"]
    for row in aa_rows:
        lines.append(f"| {row['epoch']} | {row['clean_accuracy']:.4f} | {row['pgd10_accuracy']:.4f} | {row['aa_accuracy']:.4f} | {row['epoch'] == selected['epoch']} |")
    lines += ["", "Upstream generator training-record independence remains unresolved; user accepted this limitation."]
    (output / "ssg_convnext_trades_pretrain_selection.md").write_text("\n".join(lines) + "\n")


def plot_frontier(folder, optimizer, rows):
    # Repo-native vector figure; no external plotting dependency or raster edits.
    colors = {"CleanDP": "#67748e", "Joint": "#1967d2", "Componentwise": "#c54436"}
    point = lambda r: (60 + 6 * r["dev_clean"], 460 - 4 * r["pgd20"])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="740" height="530" viewBox="0 0 740 530">',
           '<rect width="740" height="530" fill="white"/>',
           '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#888"/></marker></defs>',
           f'<text x="60" y="25" font-size="18">{optimizer}: fixed final private iterates</text>']
    for value in range(0, 101, 20):
        x, y = 60 + 6 * value, 460 - 4 * value
        svg += [f'<path d="M{x},60 V460 M60,{y} H660" stroke="#eee"/>',
                f'<text x="{x}" y="480" font-size="12">{value}</text>',
                f'<text x="30" y="{y}" font-size="12">{value}</text>']
    svg += ['<text x="290" y="515">Public-dev clean (%)</text>',
            '<text x="10" y="48">PGD20 (%)</text>']
    frontier = sorted(pareto(rows), key=lambda r: r["dev_clean"])
    points = " ".join(f"{point(r)[0]},{point(r)[1]}" for r in frontier)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#222" stroke-width="2"/>')
    for row in rows:
        x, y = point(row)
        if row["role"] == "Componentwise":
            joint = next(r for r in rows if r["run_id"] == row["matched_joint_run_id"])
            jx, jy = point(joint)
            svg.append(f'<path d="M{x},{y} L{jx},{jy}" fill="none" stroke="#aaa" marker-end="url(#arrow)"/>')
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[row["role"]]}"><title>{html.escape(row["run_id"])}</title></circle>')
    for i, (role, color) in enumerate(colors.items()):
        svg.append(f'<text x="{60+200*i}" y="48" fill="{color}" font-size="12">{role}</text>')
    svg.append('</svg>')
    (folder / f"gtsrb_{optimizer.lower()}_pareto.svg").write_text("\n".join(svg))


def aggregate_campaign(root, completed, protocol):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = root / "reports"
    geometry, telemetry, curves, compute = [], [], [], []
    writers = {}
    try:
        for row in completed:
            folder = root / "private" / row["run_id"]
            curves.append({"run_id": row["run_id"], "optimizer": row["optimizer"],
                           "curves": json.loads((folder / "gtsrb_learning_curves.json").read_text())})
            phases = json.loads((folder / "gtsrb_compute_ledger.json").read_text())
            compute.append({"run_id": row["run_id"], "phases": phases, "elapsed_seconds": row["elapsed_seconds"]})
            table = pq.read_table(folder / "gtsrb_optimizer_telemetry.parquet")
            telemetry.append({"run_id": row["run_id"], "optimizer": row["optimizer"],
                              "mean_sanitized_gradient_norm": float(np.mean(table["sanitized_gradient_norm"].to_numpy())),
                              "mean_parameter_update_norm": float(np.mean(table["parameter_update_norm"].to_numpy()))})
            for epoch in protocol["private"]["geometry_epochs"]:
                snapshot = folder / f"geometry_epoch{epoch:03d}"
                aggregate = json.loads((snapshot / "gtsrb_geometry_aggregate.json").read_text())
                geometry.append({"run_id": row["run_id"], "optimizer": row["optimizer"], "epoch": epoch, **aggregate})
                for suffix in ["per_record", "layers"]:
                    name = f"gtsrb_geometry_{suffix}.parquet"
                    values = pq.read_table(snapshot / name)
                    for col, value in [("run_id", row["run_id"]), ("optimizer", row["optimizer"]), ("epoch", epoch)]:
                        values = values.append_column(col, pa.array([value] * len(values)))
                    if name not in writers:
                        writers[name] = pq.ParquetWriter(out / name, values.schema, compression="zstd")
                    writers[name].write_table(values)
    finally:
        for writer in writers.values():
            writer.close()
    write_json(out / "gtsrb_geometry_aggregate.json", geometry)
    write_json(out / "gtsrb_geometry_manifest.json", {"scope": "fixed_public_512_only", "snapshots": len(geometry),
        "private_gradients_persisted": False, "D_M_R_training": False, "epochs": protocol["private"]["geometry_epochs"]})
    write_json(out / "gtsrb_learning_curves.json", curves)
    write_json(out / "gtsrb_compute_ledger.json", compute)
    lines = ["# Optimizer-step diagnostics", "", "Scalars from sanitized gradients only; no private per-record vectors.", "",
             "| Run | Optimizer | Mean sanitized gradient norm | Mean actual update norm |", "|---|---|---:|---:|"]
    lines += [f"| {r['run_id']} | {r['optimizer']} | {r['mean_sanitized_gradient_norm']:.6g} | {r['mean_parameter_update_norm']:.6g} |" for r in telemetry]
    (out / "gtsrb_optimizer_comparison.md").write_text("\n".join(lines) + "\n")
    lines = ["# Public-probe geometry", "", "Values below are public counterfactuals at C_mid and q=1/3, not private train statistics.", "",
             "| Run | Epoch | Mean kappa | Median beta_eff/beta | Mean U/C | Aggregate ||J||/||U|| |", "|---|---:|---:|---:|---:|---:|"]
    for item in geometry:
        values = item["settings"]["C_mid/q0.333333333333"]
        stats = values["per_record_summary"]
        lines.append(f"| {item['run_id']} | {item['epoch']} | {stats['kappa']['mean']} | {stats['beta_eff_over_beta']['p50']} | {stats['radius_U']['mean']} | {values['aggregate_norm_ratio']} |")
    (out / "gtsrb_geometry_summary.md").write_text("\n".join(lines) + "\n")
    write_json(out / "boundary_flags.json", {"automatic_extension": False, "runs": [
        {"run_id": r["run_id"], "lr_boundary_active": r["lr"] in [min(LEARNING_RATES[r["optimizer"]]), max(LEARNING_RATES[r["optimizer"]])],
         "allocation_boundary_active": r["role"] == "Componentwise" and r["q"] in [.1, .5]}
        for r in completed]})

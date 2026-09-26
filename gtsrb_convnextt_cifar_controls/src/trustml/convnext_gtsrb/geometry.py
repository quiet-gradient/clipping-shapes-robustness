"""FP64 sufficient statistics from one g/h/a calculation per public record.

Zero-angle convention: undefined directions are NaN in Parquet and null in
JSON, with explicit valid counts. D=0 when J=0 and M=0 when U=0. No raw
per-record parameter gradients leave the streaming accumulator.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .data import digest
from trustml.convnext_cifar100.preflight import write_json

QUANTILES = {"p01": .01, "p10": .10, "p25": .25, "p50": .50, "p75": .75, "p90": .90, "p99": .99}


def summary(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    result = {"count": int(a.size), "valid_count": int(valid.size), "undefined_count": int(a.size-valid.size)}
    if not valid.size:
        return {**result, **{k: None for k in ["mean", "std", "min", *QUANTILES, "max"]}}
    return {**result, "mean": float(valid.mean()), "std": float(valid.std()), "min": float(valid.min()), "max": float(valid.max()),
            **{name: float(np.quantile(valid, q)) for name, q in QUANTILES.items()}}


def divide(a, b):
    return torch.where(b > 0, a / b.clamp_min(torch.finfo(a.dtype).tiny), torch.full_like(a, float("nan")))


def vector_norm(coeff, gram):
    return torch.einsum("bi,bij,bj->b", coeff, gram, coeff).clamp_min(0).sqrt()


def cosine(left, right, gram):
    dot = torch.einsum("bi,bij,bj->b", left, gram, right)
    return divide(dot, vector_norm(left, gram)*vector_norm(right, gram)).clamp(-1, 1)


def angles(value):
    return torch.rad2deg(torch.acos(value.clamp(-1, 1)))


def coefficients(gram, bounds, q_grid):
    n = len(gram)
    device = gram.device
    basis = torch.eye(3, dtype=torch.float64, device=device)
    g, h, a = [basis[i].expand(n, -1) for i in range(3)]
    s = g+h
    vectors = {"g": g, "h": h, "s": s, "a": a}
    ng, nh, ns = (vector_norm(v, gram) for v in [g, h, s])
    rows = []
    tiny = torch.finfo(torch.float64).tiny
    for c_name, C in bounds.items():
        aj = (C/ns.clamp_min(tiny)).clamp(max=1)
        J = aj[:, None]*s
        R = (C/(ng+nh).clamp_min(tiny)).clamp(max=1)[:, None]*s
        vectors[f"{c_name}/J"] = J
        vectors[f"{c_name}/R"] = R
        nj = vector_norm(J, gram)
        for q in q_grid:
            key = f"{c_name}/q{q:.12g}"
            ag = ((1-q)*C/ng.clamp_min(tiny)).clamp(max=1)
            ah = (q*C/nh.clamp_min(tiny)).clamp(max=1)
            U = ag[:, None]*g + ah[:, None]*h
            nu = vector_norm(U, gram)
            D = torch.where((nj > 0)[:, None], J * (nu/nj.clamp_min(tiny))[:, None], torch.zeros_like(J))
            M = torch.where((nu > 0)[:, None], U * (nj/nu.clamp_min(tiny))[:, None], torch.zeros_like(U))
            dt, dr = (ag-aj)[:, None]*g, (ah-aj)[:, None]*h
            for name, v in [("U", U), ("D", D), ("M", M), ("Delta_task", dt), ("Delta_rob", dr)]:
                vectors[key+"/"+name] = v
            eu = vector_norm(s-U, gram).square()
            ej = vector_norm(s-J, gram).square()
            comm = vector_norm(U-J, gram)
            slack = eu-ej-comm.square()
            tolerance = 1e-8 * (1 + eu + ej + comm.square())
            if (slack < -tolerance).any():
                raise RuntimeError("scientific failure: projection inequality")
            if (nu > C*(1+1e-7)).any():
                raise RuntimeError("scientific failure: componentwise sensitivity")
            effective = divide(ah, ag)
            vals = {"alpha_J": aj, "alpha_g": ag, "alpha_h": ah,
                    "qstar": divide(nh, ng+nh), "abs_q_minus_qstar": (q-divide(nh, ng+nh)).abs(),
                    "beta_eff_over_beta": effective, "log_beta_eff_over_beta": effective.log(),
                    "regime": (ag < 1).long() + 2*(ah < 1).long(),
                    "cos_U_s": cosine(U, s, gram), "angle_U_s": angles(cosine(U, s, gram)),
                    "angle_U_J": angles(cosine(U, J, gram)),
                    "radius_J": nj/C, "radius_U": nu/C, "D_comm": comm, "E_J": ej, "E_U": eu,
                    "projection_slack": slack, "projection_tolerance": tolerance}
            for name, v in [("J", J), ("U", U), ("D", D), ("M", M), ("R", R)]:
                vals[name+"_norm"] = vector_norm(v, gram)
                for reference, r in [("s", s), ("g", g), ("h", h)]:
                    vals[f"{name}_angle_{reference}"] = angles(cosine(v, r, gram))
                vals[name+"_distance_J"] = vector_norm(v-J, gram)
                vals[name+"_distance_U"] = vector_norm(v-U, gram)
            rows.append((key, C, q, vals))
    return vectors, rows


def gram_metrics(gram):
    gram = np.asarray(gram, dtype=np.float64)
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    denom = norms[:, None]*norms[None, :]
    cos = np.divide(gram, denom, out=np.full_like(gram, np.nan), where=denom > 0).clip(-1, 1)
    distances = np.sqrt(np.maximum(norms[:, None]**2 + norms[None, :]**2 - 2*gram, 0))
    def json_array(a):
        return np.where(np.isfinite(a), a, None).tolist()
    return {"gram": gram.tolist(), "norms": norms.tolist(), "cosines": json_array(cos),
            "angles_degrees": json_array(np.degrees(np.arccos(cos))), "distances": distances.tolist()}


class GeometryAccumulator:
    def __init__(self, bounds, q_grid, blocks, dimension, *, device, provenance, actual=None):
        self.bounds, self.q_grid, self.blocks = bounds, q_grid, blocks
        self.dimension, self.device, self.provenance = dimension, device, provenance
        self.actual = actual
        self.sums = None
        self.names = None
        self.records, self.layers, self.ids = [], [], []
        self.count = 0

    def update(self, g, h, a, identities, outputs):
        if not all(torch.isfinite(v).all() for v in [g, h, a]):
            raise RuntimeError("nonfinite geometry gradients")
        batch = len(g)
        # One batch of three vectors, not 512 x model dimension retained.
        raw = torch.stack([g, h, a], dim=1).double()
        gram = torch.bmm(raw, raw.transpose(1, 2))
        vectors, settings = coefficients(gram, self.bounds, self.q_grid)
        soft_metrics = {}
        if self.actual:
            role, C = self.actual["role"], self.actual["C"]
            tiny = torch.finfo(torch.float64).tiny
            if self.actual.get('clipping_map') == 'auto_s':
                gamma = self.actual['gamma']
                ng, nh, ns = [vector_norm(vectors[v], gram) for v in ['g', 'h', 's']]
                vectors['SoftJoint'] = vectors['s']/(ns+gamma)[:, None]
                for q in [.25, 1/3, .5, .75]:
                    vectors[f'SoftU/q{q:.12g}'] = ((1-q)*vectors['g']/(ng+(1-q)*gamma)[:,None]
                                                +q*vectors['h']/(nh+q*gamma)[:,None])
                if role == 'DPOnly':
                    vectors['Actual'] = vectors['g']/(ng+gamma)[:, None]
                elif role == 'Joint':
                    vectors['Actual'] = vectors['SoftJoint']
                elif role == 'Componentwise':
                    q = self.actual['q']
                    vectors['Actual'] = vectors[f'SoftU/q{q:.12g}']
                    ag, ah = (1-q)/(ng+(1-q)*gamma), q/(nh+q*gamma)
                    soft_metrics.update(soft_alpha_g=ag, soft_alpha_h=ah, soft_beta_eff_over_beta=ah/ag)
                else:
                    raise ValueError('invalid soft geometry role')
                base_norm = ng if role == 'DPOnly' else ns
                soft_metrics.update(soft_actual_norm=vector_norm(vectors['Actual'],gram),
                    soft_joint_eta_effective=self.actual['lr']/(base_norm+gamma),
                    soft_joint_radial_response=gamma/(base_norm+gamma),
                    soft_h_parallel_s_fraction=cosine(vectors['h'],vectors['s'],gram).square(),
                    soft_actual_angle_s=angles(cosine(vectors['Actual'],vectors['s'],gram)))
                if (soft_metrics['soft_actual_norm'] > 1+1e-7).any():
                    raise RuntimeError('soft public query bound violated')
            elif role in {"DPOnly", "DPPGD"}:
                vector = vectors["g" if role == "DPOnly" else "a"]
                vectors["Actual"] = vector * (C/vector_norm(vector, gram).clamp_min(tiny)).clamp(max=1)[:, None]
            elif role == "Joint":
                vectors["Actual"] = vectors["s"] * (C/vector_norm(vectors["s"], gram).clamp_min(tiny)).clamp(max=1)[:, None]
            elif role == "Componentwise":
                q = self.actual["q"]
                vectors["Actual"] = sum(vectors[v] * (c/vector_norm(vectors[v], gram).clamp_min(tiny)).clamp(max=1)[:, None]
                                        for v, c in [("g", (1-q)*C), ("h", q*C)])
            if 'uniform_joint_scale' in self.actual:
                scale = self.actual['uniform_joint_scale']
                if role != 'Joint' or not 0 < scale <= 1:
                    raise ValueError('invalid uniform Joint scale in geometry')
                vectors['UniformScaledJ'] = vectors['Actual'] * scale
                vectors['Actual'] = vectors['UniformScaledJ']
                soft_metrics.update(actual_query_norm=vector_norm(vectors['Actual'], gram),
                    actual_query_angle_s=angles(cosine(vectors['Actual'], vectors['s'], gram)))
            if 'intervention' in self.actual:
                from trustml.ssg_gtsrb.confirmation import intervention_coefficients
                matched = intervention_coefficients(gram, self.actual)
                vectors.update({f'Matched/{k}': v for k, v in matched.items()})
                vectors['Actual'] = matched[self.actual['intervention']]
                soft_metrics.update(actual_query_norm=vector_norm(vectors['Actual'], gram),
                    actual_query_angle_s=angles(cosine(vectors['Actual'], vectors['s'], gram)))
                if self.actual.get('clipping_map') == 'auto_s':
                    soft_metrics['soft_actual_norm'] = soft_metrics['actual_query_norm']
                    soft_metrics['soft_actual_angle_s'] = soft_metrics['actual_query_angle_s']
        names = list(vectors)
        if self.names is None:
            self.names = names
            self.sums = torch.zeros((len(names), self.dimension), dtype=torch.float64, device=self.device)
        if self.names != names:
            raise ValueError("geometry vector order changed")
        coeff = torch.stack(list(vectors.values()), dim=1)
        # Matrix products accumulate all counterfactual vectors without backwards per setting.
        for branch in range(3):
            self.sums.addmm_(coeff[:, :, branch].T, raw[:, branch, :])
        ng, nh, ns, na = (vector_norm(vectors[name], gram) for name in ["g", "h", "s", "a"])
        base = {"norm_g": ng, "norm_h": nh, "norm_s": ns, "norm_a": na, "dot_g_h": gram[:, 0, 1],
                "cos_g_h": cosine(vectors["g"], vectors["h"], gram), "kappa": divide(ns, ng+nh),
                **{f"cos_a_{name}": cosine(vectors["a"], vectors[name], gram) for name in ["g", "h", "s"]}}
        totals = {"g": ng.square(), "h": nh.square(), "s": ns.square()}
        block_values = {}
        for name, spans in self.blocks.items():
            bg = sum(torch.bmm(raw[:, :, start:end], raw[:, :, start:end].transpose(1, 2)) for start, end in spans)
            vals = {f"norm_{v}": vector_norm(vectors[v], bg) for v in ["g", "h", "s", "a"]}
            vals.update(dot_g_h=bg[:, 0, 1], cos_g_h=cosine(vectors["g"], vectors["h"], bg))
            vals.update({f"fraction_{v}_squared_norm": divide(vals[f"norm_{v}"].square(), totals[v]) for v in totals})
            block_values[name] = {k: v.cpu().tolist() for k, v in vals.items()}
        base = {k: v.cpu().tolist() for k, v in {**base, **outputs, **soft_metrics}.items()}
        for v in ["g", "h", "s"]:
            f = block_values["classifier"][f"fraction_{v}_squared_norm"]
            base[f"classifier_fraction_{v}"] = f
            base[f"ConvNeXt_backbone_fraction_{v}"] = [1-x for x in f]
            base[f"feature_stage_fraction_{v}"] = [sum(block_values[b][f"fraction_{v}_squared_norm"][i]
                for b in self.blocks if b.startswith("stage3_")) for i in range(batch)]
            base[f"head_norm_fraction_{v}"] = block_values["head_norm"][f"fraction_{v}_squared_norm"]
        for name, vals in block_values.items():
            for i, identity in enumerate(identities):
                self.layers.append({**identity, "probe_order": self.count+i, "block": name, **{k: v[i] for k, v in vals.items()}})
        for key, C, q, vals in settings:
            scalars = {k: v.cpu().tolist() for k, v in vals.items()}
            for i, identity in enumerate(identities):
                self.records.append({**identity, "probe_order": self.count+i, "setting": key, "C": C, "q": q,
                                     **{k: v[i] for k, v in base.items()}, **{k: v[i] for k, v in scalars.items()}})
        self.ids.extend(identities)
        self.count += batch

    def finalize(self, output, expected_ids, *, sigma, bound_for_snr=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if self.ids != expected_ids or self.count != 512:
            raise ValueError("public geometry must cover the exact ordered 512 records")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        means = self.sums / self.count
        global_gram = (means @ means.T).cpu().numpy()
        aggregate = {"vector_names": self.names, "global": gram_metrics(global_gram), "blocks": {}, "settings": {},
                     "provenance": self.provenance, "probe_records": self.ids, "probe_hash": digest(self.ids),
                     "private_gradients_logged": False, "raw_parameter_gradients_persisted": False}
        aggregate["layer_summaries"] = {}
        for name, spans in self.blocks.items():
            value = sum(means[:, start:end] @ means[:, start:end].T for start, end in spans)
            aggregate["blocks"][name] = gram_metrics(value.cpu().numpy())
            rows = [r for r in self.layers if r["block"] == name]
            aggregate["layer_summaries"][name] = {k: summary([r[k] for r in rows]) for k in rows[0]
                if k.startswith(("norm_", "fraction_", "cos_", "dot_"))}
        numeric = [k for k, v in self.records[0].items() if isinstance(v, (int, float, bool)) and k not in {"label", "probe_order"}]
        idmap = {name: i for i, name in enumerate(self.names)}
        global_metrics = aggregate["global"]
        from scipy.special import gammaln
        chi_mean = math.exp(.5*math.log(2)+gammaln((self.dimension+1)/2)-gammaln(self.dimension/2))
        if self.actual:
            actual_norm = global_metrics["norms"][idmap["Actual"]]
            noise_norm = sigma*self.actual["C"]/512*chi_mean
            aggregate["actual_mechanism_snr"] = {"label": "PUBLIC-PROBE proxy, not true training SNR",
                "role": self.actual["role"], "C": self.actual["C"], "sigma": sigma, "dimension": self.dimension,
                "query_mean_norm": actual_norm, "expected_noise_l2_after_averaging": noise_norm,
                "ratio": actual_norm/noise_norm if noise_norm > 0 else None}
            aggregate['actual_mechanism'] = self.actual
        for key in dict.fromkeys(r["setting"] for r in self.records):
            rows = [r for r in self.records if r["setting"] == key]
            c_name = key.split("/")[0]
            selected = ["g", "h", "s", "a", f"{c_name}/J", key+"/U", key+"/D", key+"/M", f"{c_name}/R", key+"/Delta_task", key+"/Delta_rob"]
            ids = [idmap[n] for n in selected]
            G = global_gram[np.ix_(ids, ids)]
            residual = means[ids[5]]-means[ids[4]]-means[ids[9]]-means[ids[10]]
            if residual.norm() > 1e-7*(1+means[ids[5]].norm()+means[ids[4]].norm()):
                raise RuntimeError("aggregate decomposition invariant")
            stats = {k: summary([r[k] for r in rows]) for k in numeric}
            eff = np.asarray([r["beta_eff_over_beta"] for r in rows])
            rotation = np.asarray([r["angle_U_s"] for r in rows])
            reg = np.asarray([r["regime"] for r in rows])
            N = np.sqrt(np.maximum(np.diag(G), 0))
            C = rows[0]["C"]
            # E||N(0,I_d)||, computed stably; averaged update noise uses denominator 512.
            expected_noise_norm = sigma*C/512*chi_mean if sigma else 0
            safe_ratio = lambda x, y: float(x/y) if y > 0 else None
            d = {"vector_names": selected, **gram_metrics(G), "per_record_summary": stats,
                 "beta_eff_fractions": {"lt_025": float(np.mean(eff < .25)), "lt_05": float(np.mean(eff < .5)),
                    "in_05_2": float(np.mean((eff >= .5) & (eff <= 2))), "gt_2": float(np.mean(eff > 2)), "gt_4": float(np.mean(eff > 4))},
                 "rotation_fractions_gt": {str(t): float(np.mean(rotation > t)) for t in [5, 15, 30, 60, 90]},
                 "direction_reversal_fraction": float(np.mean(rotation > 90)),
                 "saturation_fractions": {name: float(np.mean(reg == i)) for i, name in enumerate(["unsaturated", "task_only", "robustness_only", "double_saturated"])},
                 "projection_max_violation": max(0., -min(r["projection_slack"] for r in rows)),
                 "per_record_norm_ratio": safe_ratio(stats["J_norm"]["mean"], stats["U_norm"]["mean"]),
                 "aggregate_norm_ratio": safe_ratio(N[4], N[5]), "aggregate_cancellation": safe_ratio(N[2], N[0]+N[1]),
                 "decomposition_residual_norm": float(residual.norm()),
                 "snr_label": "PUBLIC-PROBE proxy, not true training SNR", "noise_expected_l2_after_averaging": expected_noise_norm,
                 "snr_noise_convention": "counterfactual query uses its diagnostic C; actual training C is reported separately",
                 "public_probe_snr": {name: safe_ratio(N[i], expected_noise_norm) for i, name in enumerate(["g", "h", "s", "a", "J", "U", "D", "M", "R", "Delta_task", "Delta_rob"])}}
            # All desired Delta projections follow from the complete Gram matrix.
            d["distortion_projections"] = {f"{v}_on_{r}": safe_ratio(G[i, j], N[j])
                for i, v in [(9, "Delta_task"), (10, "Delta_rob")] for j, r in [(0, "g"), (1, "h"), (2, "s")]}
            aggregate["settings"][key] = d
        snapshot = {"snapshot_hash": digest(self.provenance), **self.provenance}
        for rows in [self.records, self.layers]:
            for row in rows:
                row["snapshot_hash"] = snapshot["snapshot_hash"]
        pq.write_table(pa.Table.from_pylist(self.records), output/"gtsrb_geometry_per_record.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(self.layers), output/"gtsrb_geometry_layers.parquet", compression="zstd")
        write_json(output/"gtsrb_geometry_aggregate.json", aggregate)
        write_json(output/"gtsrb_geometry_manifest.json", snapshot)
        return aggregate

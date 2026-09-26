"""Validate and aggregate downloaded full AutoAttack result artifacts.

The DataSphere launcher writes one append-only JSONL file and one official
AutoAttack text log per checkpoint.  A result is accepted only when those two
artifacts agree with the exact local checkpoint and with the publication
protocol (full CIFAR-10 test set, Linf 8/255, standard AutoAttack).

Single result::

    python -m trustml.autoattack_results \
      --jsonl extracted/aa_results/ftA_b30_s42.jsonl \
      --log extracted/aa_results/ftA_b30_s42.log \
      --checkpoint results/scoring_bundle/targets/ftA_b30_s42.pt

Batch discovery pairs files by checkpoint stem::

    python -m trustml.autoattack_results \
      --results-dir results/downloaded_jobs \
      --checkpoint-dir results/scoring_bundle/targets \
      --expected-count 14 \
      --json-output results/autoattack_validation.json \
      --markdown-output results/autoattack_validation.md

This command only reads result artifacts and checkpoints.  It never launches
jobs, downloads files, loads model weights, or mutates checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence

from trustml.robustness_eval import AUTOATTACK_COMMIT


STANDARD_ATTACKS = ("apgd-ce", "apgd-t", "fab-t", "square")
DEFAULT_ATTACK_EPSILON = 8.0 / 255.0
CORE_CHECKS = (
    "exactly_one_complete",
    "full_test_examples",
    "checkpoint_name",
    "checkpoint_sha256",
    "finite_values",
    "attack_epsilon",
    "pixel_wrapper_diagnostics",
    "clean_accuracy_match",
    "adversarial_not_above_clean",
    "standard_attack_metadata",
    "standard_attack_log",
    "official_exact_counts",
    "final_robust_log",
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ATTACK_LINE_RE = re.compile(
    r"^\s*robust accuracy after\s+([a-z0-9-]+):\s*"
    r"([0-9]+(?:\.[0-9]+)?)%\s*\(total time\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_FINAL_LINE_RE = re.compile(
    r"^\s*robust accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_INITIAL_LINE_RE = re.compile(
    r"^\s*initial accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class CaseSpec:
    jsonl: Path
    log: Path
    checkpoint: Path


@dataclass(frozen=True)
class Check:
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "detail": self.detail}


@dataclass
class CaseReport:
    model: str
    jsonl: str | None
    log: str | None
    checkpoint: str
    checks: dict[str, Check]
    summary: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "accepted": self.accepted,
            "inputs": {
                "jsonl": self.jsonl,
                "log": self.log,
                "checkpoint": self.checkpoint,
            },
            "checks": {name: check.as_dict() for name, check in self.checks.items()},
            "summary": self.summary,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_value(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if not _is_number(value):
        return None
    return float(value)


def _exact_percent_count(percent_text: str, n_examples: int) -> int | None:
    """Convert an upstream ``.2%`` log value to an exact integer count."""

    try:
        percent = Decimal(percent_text)
    except InvalidOperation:
        return None
    # Official AutoAttack formats percentages with exactly two fractional
    # digits.  On the required 10,000-example test set one displayed unit is
    # exactly one example, so no rounding ambiguity remains.
    if "." not in percent_text or len(percent_text.rsplit(".", 1)[1]) != 2:
        return None
    scaled = percent * Decimal(n_examples) / Decimal(100)
    integral = scaled.to_integral_value()
    if scaled != integral:
        return None
    count = int(integral)
    return count if 0 <= count <= n_examples else None


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    bad: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            bad.extend(_nonfinite_paths(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            bad.extend(_nonfinite_paths(child, f"{path}[{index}]"))
    return bad


def _failed_report(
    spec: CaseSpec,
    check_name: str,
    detail: str,
) -> CaseReport:
    checks = {
        name: Check(False, detail if name == check_name else f"not evaluated: {detail}")
        for name in CORE_CHECKS
    }
    if check_name not in checks:
        checks = {check_name: Check(False, detail), **checks}
    return CaseReport(
        model=spec.checkpoint.stem,
        jsonl=str(spec.jsonl),
        log=str(spec.log),
        checkpoint=str(spec.checkpoint),
        checks=checks,
        summary={},
    )


def _unavailable_report(
    checkpoint: Path,
    detail: str,
    jsonl: Path | None = None,
    log: Path | None = None,
) -> CaseReport:
    checks = {"input_discovery": Check(False, detail)}
    checks.update({name: Check(False, f"not evaluated: {detail}") for name in CORE_CHECKS})
    return CaseReport(
        model=checkpoint.stem,
        jsonl=None if jsonl is None else str(jsonl),
        log=None if log is None else str(log),
        checkpoint=str(checkpoint),
        checks=checks,
        summary={},
    )


def validate_case(
    spec: CaseSpec,
    *,
    expected_examples: int = 10_000,
    expected_epsilon: float = DEFAULT_ATTACK_EPSILON,
    expected_commit: str = AUTOATTACK_COMMIT,
    accuracy_tolerance: float | None = None,
) -> CaseReport:
    """Validate one JSONL/log/checkpoint triplet without loading model weights."""

    if accuracy_tolerance is None:
        # The standalone normalized loader and AutoAttack's pixel wrapper can
        # disagree on one borderline prediction due to floating-point
        # normalization.  Use the same count-aware guard as robustness_eval.
        accuracy_tolerance = 1.0 / expected_examples + 1e-8

    if not spec.jsonl.is_file():
        return _failed_report(spec, "exactly_one_complete", f"JSONL does not exist: {spec.jsonl}")
    try:
        records = _read_jsonl(spec.jsonl)
    except (OSError, ValueError) as exc:
        return _failed_report(spec, "exactly_one_complete", str(exc))

    complete = [record for record in records if record.get("status") == "complete"]
    exactly_one = len(records) == 1 and len(complete) == 1
    checks: dict[str, Check] = {
        "exactly_one_complete": Check(
            exactly_one,
            f"records={len(records)}, complete={len(complete)}; expected one complete record and no extras",
        )
    }
    if len(complete) != 1:
        report = _failed_report(
            spec,
            "exactly_one_complete",
            f"records={len(records)}, complete={len(complete)}; no unambiguous complete record",
        )
        report.checks["exactly_one_complete"] = checks["exactly_one_complete"]
        return report

    record = complete[0]
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    top_n = record.get("n_examples")
    metric_n = _numeric_value(metrics, "attack/autoattack_n_examples")
    full_test = (
        record.get("dataset") == "cifar10"
        and _is_number(top_n)
        and float(top_n) == float(expected_examples)
        and metric_n is not None
        and metric_n == float(expected_examples)
    )
    checks["full_test_examples"] = Check(
        full_test,
        f"dataset={record.get('dataset')!r}, record_n={top_n!r}, metric_n={metric_n!r}, "
        f"expected={expected_examples}",
    )

    remote_checkpoint = record.get("checkpoint")
    remote_name = Path(remote_checkpoint).name if isinstance(remote_checkpoint, str) else None
    experiment = record.get("experiment")
    name_matches = remote_name == spec.checkpoint.name and experiment == spec.checkpoint.stem
    checks["checkpoint_name"] = Check(
        name_matches,
        f"record checkpoint={remote_name!r}, experiment={experiment!r}, "
        f"expected={spec.checkpoint.name!r}/{spec.checkpoint.stem!r}",
    )

    record_sha = record.get("checkpoint_sha256")
    if not spec.checkpoint.is_file():
        checks["checkpoint_sha256"] = Check(False, f"expected checkpoint does not exist: {spec.checkpoint}")
        local_sha = None
    else:
        try:
            local_sha = _sha256(spec.checkpoint)
            sha_matches = isinstance(record_sha, str) and record_sha.lower() == local_sha
            checks["checkpoint_sha256"] = Check(
                sha_matches,
                f"record={record_sha!r}, local={local_sha}",
            )
        except OSError as exc:
            local_sha = None
            checks["checkpoint_sha256"] = Check(False, f"cannot hash checkpoint: {exc}")

    required_metrics = (
        "test/clean_acc",
        "attack/autoattack_clean_acc",
        "attack/autoattack_adv_acc",
        "attack/autoattack_asr",
        "attack/autoattack_n_examples",
        "attack/autoattack_max_perturbation",
        "AUTOATTACK_CLEAN_ACC",
        "AUTOATTACK_ADV_ACC",
        "AUTOATTACK_ASR",
    )
    non_numeric_metrics = [key for key in required_metrics if _numeric_value(metrics, key) is None]
    nonfinite = _nonfinite_paths(record)
    finite = not non_numeric_metrics and not nonfinite
    checks["finite_values"] = Check(
        finite,
        f"non_numeric_required={non_numeric_metrics}, non_finite={nonfinite}",
    )

    aa_meta = record.get("autoattack")
    if not isinstance(aa_meta, dict):
        aa_meta = {}
    actual_epsilon = _numeric_value(aa_meta, "epsilon")
    max_perturbation = _numeric_value(metrics, "attack/autoattack_max_perturbation")
    budget_tolerance = max(1e-6, abs(expected_epsilon) * 1e-4)
    epsilon_ok = (
        actual_epsilon is not None
        and math.isfinite(actual_epsilon)
        and math.isclose(actual_epsilon, expected_epsilon, rel_tol=0.0, abs_tol=1e-12)
        and aa_meta.get("norm") == "Linf"
        and max_perturbation is not None
        and math.isfinite(max_perturbation)
        and 0.0 <= max_perturbation <= expected_epsilon + budget_tolerance
    )
    checks["attack_epsilon"] = Check(
        epsilon_ok,
        f"norm={aa_meta.get('norm')!r}, epsilon={actual_epsilon!r}, "
        f"max_perturbation={max_perturbation!r}, expected={expected_epsilon:.17g}",
    )

    clean_values = {
        key: _numeric_value(metrics, key)
        for key in ("test/clean_acc", "attack/autoattack_clean_acc", "AUTOATTACK_CLEAN_ACC")
    }
    clean_present = all(value is not None and math.isfinite(value) for value in clean_values.values())
    clean_range = clean_present and all(0.0 <= float(value) <= 1.0 for value in clean_values.values())
    wrapper_diagnostics = record.get("pixel_wrapper_diagnostics")
    schema_version = record.get("schema_version")
    if schema_version == 2 and isinstance(wrapper_diagnostics, dict):
        direct_count = wrapper_diagnostics.get("direct_clean_correct")
        wrapper_count = wrapper_diagnostics.get("wrapper_clean_correct")
        prediction_disagreements = wrapper_diagnostics.get("prediction_disagreement_count")
        disagreement_limit = wrapper_diagnostics.get("max_prediction_disagreements")
        lattice_error = _numeric_value(wrapper_diagnostics, "input_lattice_max_abs_error")
        lattice_tolerance = _numeric_value(wrapper_diagnostics, "pixel_lattice_tolerance")
        roundtrip_error = _numeric_value(wrapper_diagnostics, "roundtrip_max_abs_error")
        roundtrip_tolerance = _numeric_value(wrapper_diagnostics, "roundtrip_tolerance")
        integer_counts = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (direct_count, wrapper_count, prediction_disagreements, disagreement_limit)
        )
        diagnostics_ok = (
            wrapper_diagnostics.get("validation_passed") is True
            and wrapper_diagnostics.get("input_validation_passed") is True
            and wrapper_diagnostics.get("n_examples") == expected_examples
            and wrapper_diagnostics.get("pixel_quantization_levels") == 256
            and integer_counts
            and 0 <= direct_count <= expected_examples
            and 0 <= wrapper_count <= expected_examples
            and 0 <= prediction_disagreements <= disagreement_limit <= 16
            and lattice_error is not None
            and lattice_tolerance is not None
            and 0.0 <= lattice_error <= lattice_tolerance <= 1e-6
            and roundtrip_error is not None
            and roundtrip_tolerance is not None
            and 0.0 <= roundtrip_error <= roundtrip_tolerance <= 1e-6
        )
        clean_match = (
            clean_range
            and diagnostics_ok
            and abs(float(clean_values["test/clean_acc"]) - direct_count / expected_examples)
            <= accuracy_tolerance
            and abs(float(clean_values["attack/autoattack_clean_acc"]) - wrapper_count / expected_examples)
            <= accuracy_tolerance
            and abs(float(clean_values["AUTOATTACK_CLEAN_ACC"]) - wrapper_count / expected_examples)
            <= accuracy_tolerance
        )
        wrapper_detail = (
            f"schema=2, direct={direct_count}, wrapper={wrapper_count}, "
            f"prediction_disagreements={prediction_disagreements}/{disagreement_limit}, "
            f"lattice={lattice_error}/{lattice_tolerance}, "
            f"roundtrip={roundtrip_error}/{roundtrip_tolerance}"
        )
    else:
        # Schema-1 jobs predate explicit pixel-wrapper evidence.  Preserve the
        # original one-example compatibility guard for already completed runs.
        diagnostics_ok = schema_version == 1
        clean_match = (
            clean_range
            and max(float(value) for value in clean_values.values())
            - min(float(value) for value in clean_values.values())
            <= accuracy_tolerance
        )
        wrapper_detail = f"legacy schema={schema_version!r}"
    checks["pixel_wrapper_diagnostics"] = Check(
        diagnostics_ok,
        wrapper_detail,
    )
    checks["clean_accuracy_match"] = Check(
        clean_match,
        f"values={clean_values}, tolerance={accuracy_tolerance:g}, {wrapper_detail}",
    )

    adv_values = {
        key: _numeric_value(metrics, key)
        for key in ("attack/autoattack_adv_acc", "AUTOATTACK_ADV_ACC")
    }
    adv_present = all(value is not None and math.isfinite(value) for value in adv_values.values())
    adv_alias_match = (
        adv_present
        and max(float(value) for value in adv_values.values())
        - min(float(value) for value in adv_values.values())
        <= accuracy_tolerance
    )
    clean_reference = clean_values["AUTOATTACK_CLEAN_ACC"]
    adv_reference = adv_values["AUTOATTACK_ADV_ACC"]
    adv_ok = (
        adv_alias_match
        and clean_reference is not None
        and math.isfinite(clean_reference)
        and adv_reference is not None
        and 0.0 <= adv_reference <= 1.0
        and adv_reference <= clean_reference + accuracy_tolerance
    )
    checks["adversarial_not_above_clean"] = Check(
        adv_ok,
        f"adv={adv_values}, clean={clean_reference!r}, tolerance={accuracy_tolerance:g}",
    )

    attacks = aa_meta.get("attacks")
    attack_names = tuple(str(name).lower() for name in attacks) if isinstance(attacks, list) else ()
    metadata_ok = (
        aa_meta.get("version") == "standard"
        and attack_names == STANDARD_ATTACKS
        and aa_meta.get("upstream_commit") == expected_commit
    )
    checks["standard_attack_metadata"] = Check(
        metadata_ok,
        f"version={aa_meta.get('version')!r}, attacks={list(attack_names)!r}, "
        f"commit={aa_meta.get('upstream_commit')!r}, expected_commit={expected_commit}",
    )

    log_text = ""
    log_error: str | None = None
    if not spec.log.is_file():
        log_error = f"AutoAttack log does not exist: {spec.log}"
    else:
        try:
            log_text = _ANSI_RE.sub("", spec.log.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            log_error = f"cannot read AutoAttack log: {exc}"

    attack_lines = [(name.lower(), float(percent) / 100.0) for name, percent in _ATTACK_LINE_RE.findall(log_text)]
    initial_values = [float(percent) / 100.0 for percent in _INITIAL_LINE_RE.findall(log_text)]
    logged_names = tuple(name for name, _ in attack_lines)
    log_values = [value for _, value in attack_lines]
    monotonic = all(
        later <= earlier + 0.000051
        for earlier, later in zip(log_values, log_values[1:])
    )
    # Upstream stops the loop once no robust examples remain.  That is a
    # legitimate completion (notably for the undefended baseline), provided
    # metadata configured all four attacks, the executed attacks are the exact
    # standard prefix, and the last logged robust accuracy is zero.
    full_sequence = logged_names == STANDARD_ATTACKS
    zero_robust_early_stop = (
        bool(logged_names)
        and logged_names == STANDARD_ATTACKS[: len(logged_names)]
        and len(logged_names) < len(STANDARD_ATTACKS)
        and math.isclose(log_values[-1], 0.0, rel_tol=0.0, abs_tol=0.000051)
    )
    standard_log_ok = (
        log_error is None
        and monotonic
        and len(initial_values) == 1
        and (full_sequence or zero_robust_early_stop)
    )
    execution_kind = "full sequence" if full_sequence else "zero-robust upstream early stop"
    checks["standard_attack_log"] = Check(
        standard_log_ok,
        log_error
        or f"initial={initial_values!r}, logged={list(logged_names)!r}, values={log_values!r}, monotonic={monotonic}, "
        f"completion={execution_kind if standard_log_ok else 'invalid'}",
    )

    initial_percent_texts = _INITIAL_LINE_RE.findall(log_text)
    final_percent_texts = _FINAL_LINE_RE.findall(log_text)
    attack_percent_texts = _ATTACK_LINE_RE.findall(log_text)
    official_initial_count = (
        _exact_percent_count(initial_percent_texts[0], expected_examples)
        if len(initial_percent_texts) == 1
        else None
    )
    official_final_count = (
        _exact_percent_count(final_percent_texts[0], expected_examples)
        if len(final_percent_texts) == 1
        else None
    )
    official_last_attack_count = (
        _exact_percent_count(attack_percent_texts[-1][1], expected_examples)
        if attack_percent_texts
        else None
    )
    exact_counts_ok = (
        expected_examples == 10_000
        and official_initial_count is not None
        and official_final_count is not None
        and official_last_attack_count is not None
        and official_final_count == official_last_attack_count
        and official_final_count <= official_initial_count
    )
    checks["official_exact_counts"] = Check(
        exact_counts_ok,
        f"n={expected_examples}, initial={official_initial_count!r}, "
        f"last_attack={official_last_attack_count!r}, final={official_final_count!r}; "
        "official .2% output is exact only for the full 10,000-example protocol",
    )

    final_values = [float(percent) / 100.0 for percent in final_percent_texts]
    final_metric = adv_values["AUTOATTACK_ADV_ACC"]
    # The official log formats percentages to two decimal places (half a
    # displayed unit is 0.00005).  AutoAttack's internal robust flags and our
    # final batched re-prediction may additionally differ on one borderline
    # example after floating-point normalization, just like the two clean
    # evaluations.  The accepted count discrepancy is therefore at most one.
    log_rounding_tolerance = 0.00005 + 1.0 / expected_examples + 1e-8
    final_ok = (
        log_error is None
        and len(initial_values) == 1
        and clean_reference is not None
        and abs(initial_values[0] - clean_reference) <= log_rounding_tolerance
        and len(final_values) == 1
        and final_metric is not None
        and math.isfinite(final_metric)
        and abs(final_values[0] - final_metric) <= log_rounding_tolerance
        and bool(log_values)
        and abs(final_values[0] - log_values[-1]) <= log_rounding_tolerance
    )
    checks["final_robust_log"] = Check(
        final_ok,
        log_error
        or f"initial_lines={initial_values!r}, clean_metric={clean_reference!r}, "
        f"final_lines={final_values!r}, metric={final_metric!r}, "
        f"last_attack={log_values[-1] if log_values else None!r}, "
        f"rounding_tolerance={log_rounding_tolerance:g}",
    )

    log_warnings = [
        line.strip()
        for line in log_text.splitlines()
        if line.lstrip().lower().startswith("warning:")
    ]

    # Schema-2 legacy jobs used AUTOATTACK_ADV_ACC for the post-hoc value.
    # Future jobs expose the canonical attack-time value there and retain the
    # re-prediction under an explicit diagnostic alias.
    posthoc_adv_acc = _numeric_value(metrics, "AUTOATTACK_ADV_ACC_POSTHOC")
    if posthoc_adv_acc is None:
        posthoc_adv_acc = adv_values["AUTOATTACK_ADV_ACC"]
    posthoc_adv_count = (
        int(round(posthoc_adv_acc * expected_examples))
        if posthoc_adv_acc is not None and math.isfinite(posthoc_adv_acc)
        else None
    )
    canonical_adv_acc = (
        official_final_count / expected_examples if exact_counts_ok else None
    )
    canonical_asr = (
        1.0 - official_final_count / official_initial_count
        if exact_counts_ok and official_initial_count
        else None
    )

    summary = {
        "experiment": record.get("experiment"),
        "checkpoint_sha256": record_sha,
        "n_examples": top_n,
        "attack_epsilon": actual_epsilon,
        "clean_acc": clean_values["AUTOATTACK_CLEAN_ACC"],
        "pgd_adv_acc": _numeric_value(metrics, "attack/pgd_adv_acc"),
        # The official attack-time robust flags are canonical.  The raw schema-2
        # metric predates return_labels=True and is a post-hoc re-prediction of
        # returned x_adv; retain it explicitly as a numerical diagnostic.
        "autoattack_adv_acc": canonical_adv_acc,
        "autoattack_asr": canonical_asr,
        "official_initial_clean_count": official_initial_count,
        "official_final_robust_count": official_final_count,
        "posthoc_reprediction_adv_acc": posthoc_adv_acc,
        "posthoc_reprediction_adv_count": posthoc_adv_count,
        "posthoc_minus_official_count": (
            None
            if posthoc_adv_count is None or official_final_count is None
            else posthoc_adv_count - official_final_count
        ),
        "elapsed_seconds": record.get("elapsed_seconds"),
        # Official AutoAttack reliability warnings are scientifically useful,
        # but do not mean that the standard ensemble was incomplete.  Surface
        # them without silently changing the acceptance contract.
        "autoattack_warnings": log_warnings,
    }
    return CaseReport(
        model=spec.checkpoint.stem,
        jsonl=str(spec.jsonl),
        log=str(spec.log),
        checkpoint=str(spec.checkpoint),
        checks=checks,
        summary=summary,
    )


def aggregate_reports(reports: Iterable[CaseReport], expected_count: int | None = None) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda report: report.model)
    accepted_count = sum(report.accepted for report in ordered)
    count_ok = expected_count is None or len(ordered) == expected_count
    return {
        "schema_version": 1,
        "accepted": count_ok and accepted_count == len(ordered) and bool(ordered),
        "case_count": len(ordered),
        "accepted_count": accepted_count,
        "expected_count": expected_count,
        "count_check": {
            "passed": count_ok,
            "detail": (
                "no expected count requested"
                if expected_count is None
                else f"found {len(ordered)} case(s), expected {expected_count}"
            ),
        },
        "cases": [report.as_dict() for report in ordered],
    }


def _percent(value: Any) -> str:
    return "—" if not _is_number(value) or not math.isfinite(float(value)) else f"{100.0 * float(value):.2f}%"


def render_markdown(aggregate: dict[str, Any]) -> str:
    """Render a compact publication-check report from :func:`aggregate_reports`."""

    accepted = aggregate["accepted_count"]
    total = aggregate["case_count"]
    expected = aggregate["expected_count"]
    expected_text = "not constrained" if expected is None else str(expected)
    lines = [
        "# AutoAttack result validation",
        "",
        f"Overall: **{'PASS' if aggregate['accepted'] else 'FAIL'}**; "
        f"accepted {accepted}/{total}; expected count: {expected_text}.",
        "",
        "| model | accepted | n | clean | PGD-20 | AutoAttack official | post-hoc x_adv | ε attack | SHA-256 |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in aggregate["cases"]:
        summary = case["summary"]
        sha = summary.get("checkpoint_sha256")
        sha_short = "—" if not isinstance(sha, str) else f"`{sha[:12]}`"
        epsilon = summary.get("attack_epsilon")
        epsilon_text = "—" if not _is_number(epsilon) else f"{float(epsilon):.8f}"
        lines.append(
            f"| `{case['model']}` | {'yes' if case['accepted'] else '**no**'} | "
            f"{summary.get('n_examples', '—')} | {_percent(summary.get('clean_acc'))} | "
            f"{_percent(summary.get('pgd_adv_acc'))} | {_percent(summary.get('autoattack_adv_acc'))} | "
            f"{_percent(summary.get('posthoc_reprediction_adv_acc'))} | "
            f"{epsilon_text} | {sha_short} |"
        )

    failures = [case for case in aggregate["cases"] if not case["accepted"]]
    if not aggregate["count_check"]["passed"] or failures:
        lines.extend(["", "## Failed checks", ""])
        if not aggregate["count_check"]["passed"]:
            lines.append(f"- `expected_count`: {aggregate['count_check']['detail']}")
        for case in failures:
            lines.append(f"- `{case['model']}`")
            failed_checks = [
                (name, check)
                for name, check in case["checks"].items()
                if not check["passed"]
            ]
            root_failures = [
                (name, check)
                for name, check in failed_checks
                if not check["detail"].startswith("not evaluated:")
            ]
            if root_failures:
                failed_checks = root_failures
            for name, check in failed_checks:
                lines.append(f"  - `{name}`: {check['detail']}")

    warned = [
        case
        for case in aggregate["cases"]
        if case["summary"].get("autoattack_warnings")
    ]
    if warned:
        lines.extend(["", "## AutoAttack warnings (non-blocking)", ""])
        for case in warned:
            for warning in case["summary"]["autoattack_warnings"]:
                lines.append(f"- `{case['model']}`: {warning}")
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _discover_cases(
    result_dirs: Sequence[Path],
    checkpoint_dirs: Sequence[Path],
) -> tuple[list[CaseSpec], list[CaseReport]]:
    checkpoints: dict[str, Path] = {}
    duplicate_checkpoints: dict[str, list[Path]] = {}
    for directory in checkpoint_dirs:
        for path in sorted(directory.rglob("*.pt")) if directory.is_dir() else []:
            if path.stem in checkpoints:
                duplicate_checkpoints.setdefault(path.stem, [checkpoints[path.stem]]).append(path)
            else:
                checkpoints[path.stem] = path
    if duplicate_checkpoints:
        formatted = "; ".join(
            f"{stem}: {[str(path) for path in paths]}" for stem, paths in duplicate_checkpoints.items()
        )
        raise ValueError(f"duplicate checkpoint stems in discovery: {formatted}")
    if not checkpoints:
        raise ValueError("checkpoint discovery found no .pt files")

    jsonls: dict[str, list[Path]] = {stem: [] for stem in checkpoints}
    logs: dict[str, list[Path]] = {stem: [] for stem in checkpoints}
    for directory in result_dirs:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
            if path.stem in checkpoints:
                jsonls[path.stem].append(path)
        for path in directory.rglob("*.log"):
            if path.stem in checkpoints:
                logs[path.stem].append(path)

    specs: list[CaseSpec] = []
    unavailable: list[CaseReport] = []
    for stem, checkpoint in sorted(checkpoints.items()):
        json_candidates = sorted(set(jsonls[stem]))
        log_candidates = sorted(set(logs[stem]))
        if len(json_candidates) == 1 and len(log_candidates) == 1:
            specs.append(CaseSpec(json_candidates[0], log_candidates[0], checkpoint))
            continue
        detail = (
            f"expected exactly one JSONL and one log for {stem}; "
            f"jsonl={[str(path) for path in json_candidates]}, "
            f"logs={[str(path) for path in log_candidates]}"
        )
        unavailable.append(
            _unavailable_report(
                checkpoint,
                detail,
                json_candidates[0] if len(json_candidates) == 1 else None,
                log_candidates[0] if len(log_candidates) == 1 else None,
            )
        )
    return specs, unavailable


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, help="single extracted robustness JSONL")
    parser.add_argument("--log", type=Path, help="single official AutoAttack log")
    parser.add_argument("--checkpoint", type=Path, help="single expected local checkpoint")
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("JSONL", "LOG", "CHECKPOINT"),
        default=[],
        help="explicit batch triplet; repeat for multiple jobs",
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        default=[],
        help="download/extracted root to search recursively; repeatable",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        type=Path,
        default=[],
        help="expected checkpoint directory for batch discovery; repeatable",
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-examples", type=int, default=10_000)
    parser.add_argument("--expected-epsilon", type=float, default=DEFAULT_ATTACK_EPSILON)
    parser.add_argument("--expected-commit", default=AUTOATTACK_COMMIT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    single_values = (args.jsonl, args.log, args.checkpoint)
    if any(value is not None for value in single_values) and not all(value is not None for value in single_values):
        parser.error("--jsonl, --log, and --checkpoint must be supplied together")
    if bool(args.results_dir) != bool(args.checkpoint_dir):
        parser.error("--results-dir and --checkpoint-dir must be supplied together")
    if args.expected_count is not None and args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    if args.expected_examples <= 0:
        parser.error("--expected-examples must be positive")
    if not math.isfinite(args.expected_epsilon) or args.expected_epsilon <= 0:
        parser.error("--expected-epsilon must be finite and positive")

    specs = [CaseSpec(Path(values[0]), Path(values[1]), Path(values[2])) for values in args.case]
    if all(value is not None for value in single_values):
        specs.append(CaseSpec(args.jsonl, args.log, args.checkpoint))

    reports: list[CaseReport] = []
    if args.results_dir:
        try:
            discovered, unavailable = _discover_cases(args.results_dir, args.checkpoint_dir)
        except ValueError as exc:
            parser.error(str(exc))
        specs.extend(discovered)
        reports.extend(unavailable)
    if not specs and not reports:
        parser.error("provide a single triplet, one or more --case triplets, or discovery directories")

    seen: set[tuple[str, str, str]] = set()
    for spec in specs:
        identity = (str(spec.jsonl.resolve()), str(spec.log.resolve()), str(spec.checkpoint.resolve()))
        if identity in seen:
            continue
        seen.add(identity)
        reports.append(
            validate_case(
                spec,
                expected_examples=args.expected_examples,
                expected_epsilon=args.expected_epsilon,
                expected_commit=args.expected_commit,
            )
        )

    aggregate = aggregate_reports(reports, expected_count=args.expected_count)
    markdown = render_markdown(aggregate)
    if args.json_output is not None:
        _write_text(args.json_output, json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.markdown_output is not None:
        _write_text(args.markdown_output, markdown)
    print(markdown, end="")
    return 0 if aggregate["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

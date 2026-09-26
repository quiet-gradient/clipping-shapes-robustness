"""One-run epsilon audit statistics (Steinke, Nasr & Jagielski, NeurIPS 2023).

"Privacy Auditing with One (1) Training Run", arXiv:2305.08846. m canaries are
each included in training independently with probability 1/2 (S_i = +1) or
held out (S_i = -1). After a single training run the auditor guesses T_i = +1
for the k+ highest canary scores and T_i = -1 for the k- lowest (Algorithm 3,
line 22), abstaining elsewhere. Theorem 5.2 / Corollary 5.4 turn the number of
correct guesses into a one-sided lower bound on epsilon.

``p_value_dp_audit`` and ``get_eps_audit`` are line-for-line ports of the
Python listing in the paper's appendix; only names and float handling differ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


def p_value_dp_audit(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """P[>= v correct guesses out of r] under the null that training is (eps, delta)-DP.

    m = number of canaries, each included independently with probability 0.5,
    r = number of guesses (excluding abstentions), v = number of correct guesses.
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1.0 / (1.0 + math.exp(-eps))  # accuracy of eps-DP randomized response
    beta = float(stats.binom.sf(v - 1, r, q))  # P[Binomial(r, q) >= v]
    alpha = 0.0
    running = 0.0  # P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        running += float(stats.binom.pmf(v - i, r, q))
        if running > i * alpha:
            alpha = running / i
    p = beta + alpha * delta * 2 * m
    return min(p, 1.0)


def get_eps_audit(m: int, r: int, v: int, delta: float, p: float) -> float:
    """Largest eps whose (eps, delta)-DP null is rejected at level p (p=0.05 -> 95%)."""
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0.0  # maintain p_value_dp_audit(eps_min) < p
    eps_max = 1.0  # maintain p_value_dp_audit(eps_max) >= p
    while p_value_dp_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1.0
    for _ in range(30):  # binary search
        eps = (eps_min + eps_max) / 2.0
        if p_value_dp_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


@dataclass
class GuessResult:
    k_pos: int
    k_neg: int
    r: int
    v: int
    accuracy: float
    eps_lb: float
    p_value_at_target: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def count_correct_guesses(
    scores: np.ndarray, s_true: np.ndarray, k_pos: int, k_neg: int
) -> int:
    """Algorithm 3, line 22: T=+1 for the k+ largest scores, T=-1 for the k- smallest."""
    if k_pos + k_neg > scores.shape[0]:
        raise ValueError("k_pos + k_neg cannot exceed the number of canaries.")
    order = np.argsort(scores, kind="stable")  # ascending
    v = 0
    if k_pos > 0:
        v += int((s_true[order[-k_pos:]] == 1).sum())
    if k_neg > 0:
        v += int((s_true[order[:k_neg]] == -1).sum())
    return v


def default_k_grid(m: int) -> list[int]:
    """Log-spaced total guess counts up to m (the paper sweeps k+ and k-)."""
    if m < 2:
        return [m]
    low = min(10, m)
    grid = {int(round(k)) for k in np.geomspace(low, m, num=20)}
    grid.add(m)
    return sorted(k for k in grid if 2 <= k <= m)


def audit_epsilon_sweep(
    scores: np.ndarray,
    s_true: np.ndarray,
    delta: float,
    p: float,
    k_values: list[int] | None = None,
    target_epsilon: float | None = None,
) -> list[GuessResult]:
    """Lower-bound epsilon for each guess count k (split evenly into k+ and k-).

    The paper evaluates different k+/k- and reports the highest lower bound
    (acknowledging the multiple-comparison caveat); we return the full sweep so
    both the max and any pre-registered k can be reported.
    """
    scores = np.asarray(scores, dtype=np.float64)
    s_true = np.asarray(s_true, dtype=np.int64)
    if scores.shape != s_true.shape:
        raise ValueError("scores and s_true must have the same shape.")
    m = int(scores.shape[0])
    if k_values is None:
        k_values = default_k_grid(m)
    results: list[GuessResult] = []
    for k in k_values:
        k = int(min(int(k), m))
        if k < 1:
            continue
        k_pos = k // 2
        k_neg = k - k_pos
        r = k_pos + k_neg
        v = count_correct_guesses(scores, s_true, k_pos, k_neg)
        eps_lb = get_eps_audit(m, r, v, delta, p)
        p_at_target = (
            p_value_dp_audit(m, r, v, float(target_epsilon), delta)
            if target_epsilon is not None
            else None
        )
        results.append(
            GuessResult(
                k_pos=k_pos,
                k_neg=k_neg,
                r=r,
                v=v,
                accuracy=v / max(1, r),
                eps_lb=eps_lb,
                p_value_at_target=p_at_target,
            )
        )
    return results


def score_auc(scores: np.ndarray, s_true: np.ndarray) -> float:
    """AUC of scores separating included (S=+1) from excluded (S=-1) canaries.

    Effect-size diagnostic independent of the choice of k; 0.5 = no signal.
    """
    scores = np.asarray(scores, dtype=np.float64)
    s_true = np.asarray(s_true, dtype=np.int64)
    pos = scores[s_true == 1]
    neg = scores[s_true == -1]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = float(ranks[: pos.size].sum()) - pos.size * (pos.size + 1) / 2.0
    return u / (pos.size * neg.size)

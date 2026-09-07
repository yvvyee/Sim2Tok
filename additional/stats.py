"""
Interval estimates and tests for the additional-experiment results.

Two things make the naive statistics on this data misleading, and both are
handled here.

**Rare events.** A hallucination rate of 0 out of N probes is not evidence that
the rate is zero; it bounds the rate from above.  ``clopper_pearson`` returns
the exact binomial interval, so a run that observes no hallucination reports
"0 events, 95% upper bound p" rather than "0%".

**Clustered frames.** SMD frames come from videos, and frames from one video
are far from independent.  Treating each frame as an independent draw
understates every interval.  ``cluster_bootstrap_mean`` and
``cluster_bootstrap_paired_diff`` resample whole videos, which keeps the video
as the unit of independence.

The module deliberately has no SciPy dependency: the regularised incomplete
beta function is implemented directly so the same numbers come out on any of
the machines involved.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, asdict

# ── Cluster identity ─────────────────────────────────────────────────────────

#: ``MVI_1448_VIS_Haze_f00000`` -> ``MVI_1448_VIS_Haze``
_FRAME_SUFFIX = re.compile(r"_f\d+$")


def video_of(sample: dict) -> str:
    """The video a frame came from, which is the cluster the intervals resample.

    Sets built by ``smd_full_to_gaa`` carry an explicit ``video`` field.  The
    evaluation sets used by the submitted work do not, so it is recovered from
    the frame id by stripping the ``_f<index>`` suffix.  Falling back to the id
    itself would make every frame its own cluster and report an interval far
    narrower than the data supports: on the 288-frame set that is 288 clusters
    instead of the 36 videos they came from.
    """
    if sample.get("video"):
        return sample["video"]
    stem = _FRAME_SUFFIX.sub("", sample.get("id", ""))
    return stem or sample.get("id", "")

# ── Regularised incomplete beta ──────────────────────────────────────────────

_MAXIT = 300
_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse of ``betainc`` in x, by bisection (monotone in x)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── Binomial proportion interval ─────────────────────────────────────────────

@dataclass(frozen=True)
class Proportion:
    """A rate with its exact binomial confidence interval."""

    k: int
    n: int
    point: float
    lo: float
    hi: float
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        conf = int(round((1 - self.alpha) * 100))
        return (f"{self.k}/{self.n} = {self.point * 100:.4f}% "
                f"({conf}% CI [{self.lo * 100:.4f}%, {self.hi * 100:.4f}%])")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Proportion:
    """Exact (Clopper-Pearson) interval for ``k`` successes in ``n`` trials.

    With ``k == 0`` the lower bound is 0 and the upper bound is the quantity to
    report: the largest rate consistent with having seen no event.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return Proportion(k, n, k / n, lo, hi, alpha)


# ── Fisher exact test ────────────────────────────────────────────────────────

def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table ``[[a, b], [c, d]]``.

    Sums the hypergeometric probabilities of every table at least as extreme as
    the observed one, so it is valid at the tiny event counts this study deals
    with.  Computed in log space to stay exact for five-figure totals.
    """
    n = a + b + c + d
    if n == 0:
        raise ValueError("empty table")
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(_log_choose(col1, x) + _log_choose(n - col1, row1 - x)
                        - _log_choose(n, row1))

    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    p_obs = prob(a)
    # 1 + 1e-9 keeps ties from being dropped by floating-point noise.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= p_obs * (1 + 1e-9)))


# ── Cluster bootstrap ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval over clusters."""

    point: float
    lo: float
    hi: float
    n_clusters: int
    n_items: int
    n_boot: int
    alpha: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        conf = int(round((1 - self.alpha) * 100))
        return (f"{self.point:.4f} ({conf}% CI [{self.lo:.4f}, {self.hi:.4f}], "
                f"{self.n_clusters} clusters / {self.n_items} items)")


def _group(values, clusters) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for v, g in zip(values, clusters):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        by.setdefault(g, []).append(float(v))
    if not by:
        raise ValueError("no finite values")
    return by


def _percentiles(draws: list[float], alpha: float) -> tuple[float, float]:
    draws = sorted(draws)
    n = len(draws)
    def q(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return draws[i]
    return q(alpha / 2), q(1 - alpha / 2)


def cluster_bootstrap_mean(values, clusters, n_boot: int = 2000,
                           alpha: float = 0.05, seed: int = 0) -> Estimate:
    """Mean of ``values`` with a percentile interval from resampling clusters.

    Clusters (videos) are drawn with replacement; every item inside a drawn
    cluster is kept.  The point estimate is the plain mean over all items, so it
    matches what a per-frame average reports; only the interval changes.
    """
    by = _group(values, clusters)
    keys = list(by)
    flat = [v for k in keys for v in by[k]]
    point = sum(flat) / len(flat)

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        picked = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        pool = [v for chunk in picked for v in chunk]
        if pool:
            draws.append(sum(pool) / len(pool))
    lo, hi = _percentiles(draws, alpha)
    return Estimate(point, lo, hi, len(keys), len(flat), n_boot, alpha)


def cluster_bootstrap_paired_diff(values_a, values_b, clusters,
                                  n_boot: int = 2000, alpha: float = 0.05,
                                  seed: int = 0) -> Estimate:
    """Interval for ``mean(b) - mean(a)`` on paired per-item measurements.

    Both sequences must be aligned item-for-item (the same frame measured under
    two encoders).  Resampling is over clusters, which keeps the pairing intact
    and accounts for frames of one video moving together.
    """
    if len(values_a) != len(values_b) or len(values_a) != len(clusters):
        raise ValueError("values_a, values_b and clusters must be the same length")

    by: dict[str, list[tuple[float, float]]] = {}
    for va, vb, g in zip(values_a, values_b, clusters):
        if va is None or vb is None:
            continue
        if (isinstance(va, float) and math.isnan(va)) or \
           (isinstance(vb, float) and math.isnan(vb)):
            continue
        by.setdefault(g, []).append((float(va), float(vb)))
    if not by:
        raise ValueError("no finite pairs")

    keys = list(by)
    flat = [p for k in keys for p in by[k]]
    point = sum(b - a for a, b in flat) / len(flat)

    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        picked = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        pool = [p for chunk in picked for p in chunk]
        if pool:
            draws.append(sum(b - a for a, b in pool) / len(pool))
    lo, hi = _percentiles(draws, alpha)
    return Estimate(point, lo, hi, len(keys), len(flat), n_boot, alpha)


def cluster_proportion(events, clusters, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Event rate reported two ways: exact binomial and cluster bootstrap.

    ``events`` is a sequence of 0/1 per probe and ``clusters`` the video each
    probe came from.  The exact interval assumes independent probes and is the
    optimistic bound; the bootstrap interval accounts for clustering and is the
    one to quote.
    """
    ev = [int(bool(e)) for e in events]
    exact = clopper_pearson(sum(ev), len(ev), alpha)
    boot = cluster_bootstrap_mean(ev, clusters, n_boot, alpha, seed)
    return {
        "k": exact.k,
        "n": exact.n,
        "rate": exact.point,
        "exact_binomial": exact.as_dict(),
        "cluster_bootstrap": boot.as_dict(),
    }

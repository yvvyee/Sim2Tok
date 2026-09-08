"""
Prior-invariant read-outs of the COLREGs encounter classification.

``colregs_eval.py`` reports accuracy, which on this corpus is not a measure of
discrimination.  Every condition puts most of its predicted mass on a single
class, so accuracy tracks which class a condition collapses onto and how large
that ground-truth class happens to be.  A model that always answers "crossing"
scores 0.3575 on this corpus while separating nothing at all, and a model that
always answers "head-on" scores 0.3528.  Comparing two such numbers says which
constant a condition drifted toward, not whether the encoder made the encounter
types distinguishable.

Three read-outs here are invariant to the prediction prior:

``macro_recall``
    The mean of the per-class recalls.  A constant prediction gives 1/3
    regardless of which class it names, so 1/3 is the floor to compare against
    rather than the accuracy of a majority guess.

``cramers_v``
    The chi-squared association between ground truth and prediction, scaled to
    [0, 1].  Zero means the prediction carries no information about the label.

``nmi``
    Mutual information between label and prediction, normalised by the
    geometric mean of the two entropies.  Reported alongside V because the two
    weight a sparse off-diagonal differently.

Confidence intervals come from a nonparametric bootstrap over samples.  The
AIVN-SDS identifier is an opaque hash with no episode field, so the sample is
the only resampling unit the data supports; whether two samples share a
simulator episode cannot be checked from the identifier, and the intervals are
therefore conditional on samples being independent.

Usage::

    python -m additional.colregs_invariant \\
        --preds additional_exp/raw/colregs_10k/preds_stock__with.jsonl \\
        --preds additional_exp/raw/colregs_10k/preds_gaa__with.jsonl \\
        --paired stock__with gaa__with \\
        --out additional_exp/colregs_invariant.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Ground-truth classes present in the corpus.  "safe" and "unknown" can be
#: predicted but never occur as a label, so they are columns and not rows.
GT_CLASSES = ["crossing", "head-on", "overtaking"]
PRED_CLASSES = ["crossing", "head-on", "overtaking", "safe", "unknown"]


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def confusion(records: list[dict]) -> list[list[int]]:
    """``[len(GT_CLASSES)][len(PRED_CLASSES)]`` counts."""
    gi = {c: i for i, c in enumerate(GT_CLASSES)}
    pi = {c: i for i, c in enumerate(PRED_CLASSES)}
    t = [[0] * len(PRED_CLASSES) for _ in GT_CLASSES]
    for r in records:
        g, p = r.get("gt"), r.get("pred")
        if g in gi and p in pi:
            t[gi[g]][pi[p]] += 1
    return t


def accuracy(t: list[list[int]]) -> float:
    n = sum(sum(r) for r in t)
    if n == 0:
        return float("nan")
    return sum(t[i][i] for i in range(len(GT_CLASSES))) / n


def macro_recall(t: list[list[int]]) -> tuple[float, list[float]]:
    rec = []
    for i in range(len(GT_CLASSES)):
        n = sum(t[i])
        rec.append(t[i][i] / n if n else float("nan"))
    present = [r for r in rec if not math.isnan(r)]
    return (sum(present) / len(present) if present else float("nan")), rec


def cramers_v(t: list[list[int]]) -> float:
    """Chi-squared association, scaled to [0, 1].

    ``k`` counts only rows and columns with a non-zero margin.  A condition
    that predicts one class puts every count in one column, which is a 3x1
    table: the association is zero by construction, and the usual ``k - 1``
    denominator would be zero.  Returning 0.0 there is the correct value, not a
    guard against a division error.
    """
    n = sum(sum(r) for r in t)
    if n == 0:
        return float("nan")
    row = [sum(r) for r in t]
    col = [sum(t[i][j] for i in range(len(t))) for j in range(len(t[0]))]
    chi2 = 0.0
    for i in range(len(t)):
        for j in range(len(t[0])):
            e = row[i] * col[j] / n
            if e > 0:
                chi2 += (t[i][j] - e) ** 2 / e
    k = min(sum(1 for r in row if r), sum(1 for c in col if c))
    if k < 2:
        return 0.0
    return math.sqrt(chi2 / (n * (k - 1)))


def nmi(t: list[list[int]]) -> float:
    n = sum(sum(r) for r in t)
    if n == 0:
        return float("nan")
    row = [sum(r) / n for r in t]
    col = [sum(t[i][j] for i in range(len(t))) / n for j in range(len(t[0]))]
    mi = 0.0
    for i in range(len(t)):
        for j in range(len(t[0])):
            p = t[i][j] / n
            if p > 0:
                mi += p * math.log(p / (row[i] * col[j]))

    def entropy(ps):
        return -sum(p * math.log(p) for p in ps if p > 0)

    denom = math.sqrt(entropy(row) * entropy(col))
    return mi / denom if denom > 0 else 0.0


METRICS = {"accuracy": accuracy,
           "macro_recall": lambda t: macro_recall(t)[0],
           "cramers_v": cramers_v,
           "nmi": nmi}


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def bootstrap(records: list[dict], n_boot: int, seed: int) -> dict:
    """Percentile intervals for every metric, resampling samples with replacement."""
    rng = random.Random(seed)
    n = len(records)
    draws: dict[str, list[float]] = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        t = confusion([records[i] for i in idx])
        for k, fn in METRICS.items():
            v = fn(t)
            if not math.isnan(v):
                draws[k].append(v)
    return {k: {"lo": _percentile(v, 0.025), "hi": _percentile(v, 0.975),
                "n_draws": len(v)} for k, v in draws.items()}


def paired_diff(a: list[dict], b: list[dict], n_boot: int, seed: int) -> dict:
    """``b - a`` per metric, resampling the shared samples once per draw.

    The two conditions ran over the same corpus, so resampling them jointly
    keeps the comparison paired and removes the corpus variance that both
    share.  Samples present in only one file are dropped and counted.
    """
    by_a = {r["id"]: r for r in a}
    by_b = {r["id"]: r for r in b}
    shared = sorted(set(by_a) & set(by_b))
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = [rng.randrange(len(shared)) for _ in range(len(shared))]
        ids = [shared[i] for i in idx]
        ta = confusion([by_a[i] for i in ids])
        tb = confusion([by_b[i] for i in ids])
        for k, fn in METRICS.items():
            va, vb = fn(ta), fn(tb)
            if not (math.isnan(va) or math.isnan(vb)):
                draws[k].append(vb - va)
    point = {}
    for k, fn in METRICS.items():
        point[k] = fn(confusion([by_b[i] for i in shared])) - \
            fn(confusion([by_a[i] for i in shared]))
    return {
        "n_shared": len(shared),
        "n_only_a": len(set(by_a) - set(by_b)),
        "n_only_b": len(set(by_b) - set(by_a)),
        "diff": {k: {"point": point[k],
                     "lo": _percentile(v, 0.025),
                     "hi": _percentile(v, 0.975),
                     "n_draws": len(v)} for k, v in draws.items()},
    }


def label_of(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("preds_") and base.endswith(".jsonl"):
        return base[len("preds_"):-len(".jsonl")]
    return base


def parse_args():
    p = argparse.ArgumentParser("prior-invariant COLREGs read-outs")
    p.add_argument("--preds", action="append", required=True,
                   help="preds_<label>.jsonl from colregs_eval; repeatable")
    p.add_argument("--paired", nargs=2, action="append", default=[],
                   metavar=("BASE", "OTHER"),
                   help="report OTHER - BASE per metric; repeatable")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    loaded = {label_of(p): load(p) for p in args.preds}
    out: dict[str, dict] = {}

    hdr = (f"{'condition':18s} {'acc':>7s} {'macroRec':>9s} {'V':>7s} "
           f"{'NMI':>7s} {'maxPredShare':>13s}  recall " +
           "/".join(GT_CLASSES))
    print(hdr, flush=True)
    for label, recs in loaded.items():
        t = confusion(recs)
        n = sum(sum(r) for r in t)
        mr, rec = macro_recall(t)
        col = [sum(t[i][j] for i in range(len(t))) for j in range(len(t[0]))]
        ci = bootstrap(recs, args.n_boot, args.seed)
        out[label] = {
            "n": n,
            "confusion": {g: {p: t[i][j] for j, p in enumerate(PRED_CLASSES)
                              if t[i][j]}
                          for i, g in enumerate(GT_CLASSES)},
            "prediction_distribution": {p: col[j]
                                        for j, p in enumerate(PRED_CLASSES)
                                        if col[j]},
            "max_pred_share": max(col) / n if n else float("nan"),
            "per_class_recall": dict(zip(GT_CLASSES, rec)),
            "metrics": {k: {"point": fn(t), **ci[k]}
                        for k, fn in METRICS.items()},
        }
        print(f"{label:18s} {accuracy(t):7.4f} {mr:9.4f} {cramers_v(t):7.4f} "
              f"{nmi(t):7.4f} {max(col) / n if n else float('nan'):13.4f}  "
              + " / ".join(f"{r:.4f}" for r in rec), flush=True)

    pairs = {}
    for base, other in args.paired:
        if base not in loaded or other not in loaded:
            raise SystemExit(f"--paired {base} {other}: unknown label; "
                             f"have {sorted(loaded)}")
        res = paired_diff(loaded[base], loaded[other], args.n_boot, args.seed)
        pairs[f"{other}_minus_{base}"] = res
        print(f"\n[paired] {other} - {base}  (n_shared={res['n_shared']:,})",
              flush=True)
        for k, d in res["diff"].items():
            crosses = "0 포함" if d["lo"] <= 0.0 <= d["hi"] else "0 제외"
            print(f"  {k:14s} {d['point']:+.4f} "
                  f"95% [{d['lo']:+.4f}, {d['hi']:+.4f}]  {crosses}",
                  flush=True)

    # A constant prediction is the reference point for macro recall, so the
    # floor is stated in the artefact rather than left for a reader to derive.
    ref = None
    for recs in loaded.values():
        t = confusion(recs)
        rows = [sum(r) for r in t]
        n = sum(rows)
        if n:
            ref = {
                "label_distribution": dict(zip(GT_CLASSES, rows)),
                "macro_recall_of_any_constant_prediction": 1.0 / len(GT_CLASSES),
                "accuracy_of_each_constant_prediction":
                    {g: rows[i] / n for i, g in enumerate(GT_CLASSES)},
                "cramers_v_of_any_constant_prediction": 0.0,
                "nmi_of_any_constant_prediction": 0.0,
            }
            break

    payload = {
        "note": ("accuracy on this corpus follows the prediction prior; "
                 "macro_recall, cramers_v and nmi are invariant to it"),
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed,
                      "resampling_unit": "sample",
                      "caveat": ("the AIVN-SDS id is an opaque hash with no "
                                 "episode field, so episode-level clustering "
                                 "cannot be checked from the data")},
        "reference": ref,
        "conditions": out,
        "paired": pairs,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[invariant] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

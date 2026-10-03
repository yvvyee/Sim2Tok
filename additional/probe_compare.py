"""
Metrics and paired differences for the frozen-encoder probe (``eval_token_probe``).

Per encoder and dataset:

``presence_auroc``  AUROC of the image presence score (highest token occupancy)
                    for "a vessel is inside the centre crop"; needs vessel and
                    vessel-free images
``token_ap``        average precision of token occupancy against the tokens a
                    vessel box covers, pooled over images
``count_mae``       mean absolute error of the predicted count, at the centre
                    threshold that minimises the error on the simulator
                    validation split for that encoder (chosen without real data)
``gap_*``           the same metric on the simulator validation split minus on
                    the real dataset

B - A for every metric gets a 95% interval from a paired bootstrap over the
resampling unit (``video``) on the images both encoders scored.

    python -m additional.probe_compare --root results/probe --seed 0 \\
        --ref stock --cmp gaa tok_occ tok_occ_state --out probe_s0.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

import numpy as np


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap_from_hist(hp, hn):
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1])
    P = hp.sum()
    if P == 0:
        return None
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


class Units:
    """Per-unit sufficient statistics, so a bootstrap draw is a cheap sum."""

    def __init__(self, rows_by_unit, thr_idx):
        self.keys = sorted(rows_by_unit)
        self.score, self.label, self.hp, self.hn, self.err, self.n = {}, {}, {}, {}, {}, {}
        for u in self.keys:
            rs = rows_by_unit[u]
            self.score[u] = np.array([r["max_occ"] for r in rs])
            self.label[u] = np.array([r["n_visible"] > 0 for r in rs])
            self.hp[u] = np.sum([r["hist_pos"] for r in rs], axis=0)
            self.hn[u] = np.sum([r["hist_neg"] for r in rs], axis=0)
            self.err[u] = float(sum(abs(r["counts"][thr_idx] - r["n_visible"]) for r in rs))
            self.n[u] = len(rs)

    def metrics(self, draw):
        out = {}
        a = auroc(np.concatenate([self.score[u] for u in draw]),
                  np.concatenate([self.label[u] for u in draw]))
        if a is not None:
            out["presence_auroc"] = a
        ap = ap_from_hist(np.sum([self.hp[u] for u in draw], axis=0),
                          np.sum([self.hn[u] for u in draw], axis=0))
        if ap is not None:
            out["token_ap"] = ap
        out["count_mae"] = sum(self.err[u] for u in draw) / sum(self.n[u] for u in draw)
        return out


def by_unit(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["video"]].append(r)
    return g


def metrics(rows, thr_idx):
    u = Units(by_unit(rows), thr_idx)
    return u.metrics(u.keys)


def best_thr(rows):
    maes = [np.mean([abs(r["counts"][k] - r["n_visible"]) for r in rows]) for k in range(9)]
    return int(np.argmin(maes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--cmp", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    rng = random.Random(0)

    def d(label):
        return os.path.join(args.root, f"{label}_s{args.seed}")

    labels = [args.ref] + args.cmp
    data = {}
    for lb in labels:
        meta = json.load(open(os.path.join(d(lb), "meta.json")))
        data[lb] = {s: load(os.path.join(d(lb), f"{s}.jsonl")) for s in meta["sets"]}
    thr = {lb: best_thr(list(data[lb]["aivn_val"].values())) for lb in labels}
    sets = list(data[args.ref])
    result = {"seed": args.seed, "ref": args.ref, "thresholds_idx": thr, "per_encoder": {},
              "paired": {}}
    for lb in labels:
        result["per_encoder"][lb] = {s: metrics(list(data[lb][s].values()), thr[lb]) for s in sets}
        sim = result["per_encoder"][lb]["aivn_val"]
        for s in sets:
            if s == "aivn_val":
                continue
            m = result["per_encoder"][lb][s]
            for k in list(m):
                if not k.startswith("gap_") and k in sim:
                    m["gap_" + k] = sim[k] - m[k]

    for lb in args.cmp:
        result["paired"][lb] = {}
        for s in sets:
            A, B = data[args.ref][s], data[lb][s]
            ids = sorted(set(A) & set(B))
            ua = Units(by_unit([A[i] for i in ids]), thr[args.ref])
            ub = Units(by_unit([B[i] for i in ids]), thr[lb])
            ukeys = ua.keys
            ma, mb = ua.metrics(ukeys), ub.metrics(ukeys)
            keys = [k for k in ma if k in mb]
            boots = {k: [] for k in keys}
            for _ in range(args.n_boot):
                draw = rng.choices(ukeys, k=len(ukeys))
                xa, xb = ua.metrics(draw), ub.metrics(draw)
                for k in keys:
                    if k in xa and k in xb:
                        boots[k].append(xb[k] - xa[k])
            res = {"n_images": len(ids), "n_units": len(ukeys)}
            for k in keys:
                bs = sorted(boots[k])
                res[k] = {"a": ma[k], "b": mb[k], "diff": mb[k] - ma[k],
                          "ci95": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]]}
            result["paired"][lb][s] = res
    json.dump(result, open(args.out, "w"), indent=2)

    for lb in labels:
        print(f"== {lb} (count threshold {0.1 * (thr[lb] + 1):.1f})")
        for s in sets:
            m = result["per_encoder"][lb][s]
            print(f"   {s:13s} " + "  ".join(f"{k} {v:+.4f}" if k.startswith("gap") else f"{k} {v:.4f}"
                                             for k, v in m.items()))
    for lb in args.cmp:
        print(f"== {lb} - {args.ref}")
        for s, res in result["paired"][lb].items():
            print(f"   {s:13s} " + "  ".join(
                f"{k} {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                for k, x in res.items() if isinstance(x, dict)))


if __name__ == "__main__":
    main()

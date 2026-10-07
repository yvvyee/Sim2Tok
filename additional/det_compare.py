"""
AP50 / AP75 for the frozen-encoder detection probe (``eval_det_probe``), with
paired differences bootstrapped over videos.

AP is all-point interpolated precision over recall, computed from every
detection of the pooled images ranked by score.  For B - A, only images both
encoders scored are used, and the same video draw is applied to both.

    python -m additional.det_compare --root results/det --seed 0 \\
        --ref gaa --cmp tok_occ --out det_tok_vs_gaa_s0.json
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


def ap(scores, tps, n_gt):
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    tp = np.asarray(tps, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]       # monotone envelope
    r = np.concatenate([[0.0], rec])
    return float(np.sum((r[1:] - r[:-1]) * prec))


class Units:
    def __init__(self, rows):
        g = collections.defaultdict(list)
        for r in rows:
            g[r["video"]].append(r)
        self.keys = sorted(g)
        self.s, self.t50, self.t75, self.n = {}, {}, {}, {}
        for u, rs in g.items():
            self.s[u] = np.concatenate([np.asarray(r["scores"], float) for r in rs]) if rs else np.zeros(0)
            self.t50[u] = np.concatenate([np.asarray(r["tp50"], float) for r in rs])
            self.t75[u] = np.concatenate([np.asarray(r["tp75"], float) for r in rs])
            self.n[u] = sum(r["n_gt"] for r in rs)

    def metrics(self, draw):
        s = np.concatenate([self.s[u] for u in draw])
        n = sum(self.n[u] for u in draw)
        out = {}
        for k, t in (("ap50", self.t50), ("ap75", self.t75)):
            v = ap(s, np.concatenate([t[u] for u in draw]), n)
            if v is not None:
                out[k] = v
        return out


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
    d = lambda lb: os.path.join(args.root, f"{lb}_s{args.seed}")
    labels = [args.ref] + args.cmp
    data = {}
    for lb in labels:
        meta = json.load(open(os.path.join(d(lb), "meta.json")))
        data[lb] = {s: load(os.path.join(d(lb), f"{s}.jsonl")) for s in meta["sets"]}
    sets = list(data[args.ref])
    res = {"seed": args.seed, "ref": args.ref, "per_encoder": {}, "paired": {}}
    for lb in labels:
        res["per_encoder"][lb] = {}
        for s in sets:
            u = Units(list(data[lb][s].values()))
            m = u.metrics(u.keys)
            m["n_gt"] = sum(u.n.values()); m["n_images"] = len(data[lb][s])
            res["per_encoder"][lb][s] = m
        sim = res["per_encoder"][lb]["aivn_val"]
        for s in sets:
            if s != "aivn_val":
                for k in ("ap50", "ap75"):
                    if k in sim and k in res["per_encoder"][lb][s]:
                        res["per_encoder"][lb][s]["gap_" + k] = sim[k] - res["per_encoder"][lb][s][k]
    for lb in args.cmp:
        res["paired"][lb] = {}
        for s in sets:
            A, B = data[args.ref][s], data[lb][s]
            ids = sorted(set(A) & set(B))
            ua, ub = Units([A[i] for i in ids]), Units([B[i] for i in ids])
            ma, mb = ua.metrics(ua.keys), ub.metrics(ua.keys)
            boots = {k: [] for k in ma if k in mb}
            for _ in range(args.n_boot):
                draw = rng.choices(ua.keys, k=len(ua.keys))
                xa, xb = ua.metrics(draw), ub.metrics(draw)
                for k in boots:
                    if k in xa and k in xb:
                        boots[k].append(xb[k] - xa[k])
            r = {"n_images": len(ids), "n_units": len(ua.keys)}
            for k, bs in boots.items():
                bs.sort()
                r[k] = {"a": ma[k], "b": mb[k], "diff": mb[k] - ma[k],
                        "ci95": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]]}
            res["paired"][lb][s] = r
    json.dump(res, open(args.out, "w"), indent=2)
    for lb in labels:
        print(f"== {lb}")
        for s in sets:
            m = res["per_encoder"][lb][s]
            print(f"   {s:13s} " + "  ".join(f"{k} {v:+.4f}" if k.startswith("gap") else
                                             (f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}")
                                             for k, v in m.items()))
    for lb in args.cmp:
        print(f"== {lb} - {args.ref}")
        for s, r in res["paired"][lb].items():
            print(f"   {s:13s} " + "  ".join(f"{k} {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                                             for k, x in r.items() if isinstance(x, dict)))


if __name__ == "__main__":
    main()

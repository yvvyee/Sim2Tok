"""
Paired comparison of two verification VLMs on the yes/no probes.

Reads the per-question part files ``yesno_eval`` writes for condition A and
condition B, keeps the (frame, question) pairs both answered, and reports per
dataset and question family:

``vessel``   recall (said yes where a vessel is visible), false-positive rate
             (said yes where none is), balanced accuracy
``count``    balanced accuracy over the count thresholds
``side``     accuracy over the left/right questions
``absent``   affirmative rate on absent non-maritime categories, which is the
             hallucination rate; per category as well
``present``  affirmative rate

Every quantity is computed for A and B on the same pairs, and B - A gets a 95%
interval from a bootstrap over the resampling unit recorded in the part files
(``video``: SMD video, LaRS sequence, SeaDronesSee source video, a frame for
MCAD and the buoy set, a frame for the AIVN validation set).

    python -m additional.verify_s2_compare \\
        --a results/verify_s2/M0_s0 --a_label M0 \\
        --b results/verify_s2/M1_s0 --b_label M1 \\
        --out results/verify_s2/compare_M1_vs_M0_s0.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random


def load_parts(ds_dir: str, label: str) -> dict:
    """{(id, question): record} for one model on one dataset."""
    out = {}
    for f in glob.glob(os.path.join(ds_dir, "parts", f"ans_{label}__with__*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["id"], r["question"])] = r
    return out


def tally(recs):
    """Sufficient statistics for every metric, per resampling unit."""
    t = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        c = t[r["video"]]
        e, y = r["expected_yes"], r["said_yes"]
        c["n"] += 1
        c["yes"] += y
        c["correct"] += r["correct"]
        c[f"n{e}"] += 1
        c[f"y{e}"] += y
    return t


def metrics(c: collections.Counter) -> dict:
    m = {"n": c["n"]}
    if c["n"]:
        m["yes_rate"] = c["yes"] / c["n"]
        m["accuracy"] = c["correct"] / c["n"]
    if c["n1"]:
        m["recall"] = c["y1"] / c["n1"]
    if c["n0"]:
        m["fp_rate"] = c["y0"] / c["n0"]
    if c["n1"] and c["n0"]:
        m["balanced_accuracy"] = (m["recall"] + 1 - m["fp_rate"]) / 2
    return m


def summed(tal, units):
    s = collections.Counter()
    for u in units:
        s.update(tal[u])
    return s


def compare(recs_a, recs_b, n_boot, rng):
    ta, tb = tally(recs_a), tally(recs_b)
    units = sorted(ta)
    ma, mb = metrics(summed(ta, units)), metrics(summed(tb, units))
    keys = [k for k in ma if k != "n" and k in mb]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choices(units, k=len(units))
        xa, xb = metrics(summed(ta, draw)), metrics(summed(tb, draw))
        for k in keys:
            if k in xa and k in xb:
                boots[k].append(xb[k] - xa[k])
    out = {"n_pairs": ma["n"], "n_units": len(units)}
    for k in keys:
        bs = sorted(boots[k])
        lo = bs[int(0.025 * len(bs))] if bs else None
        hi = bs[int(0.975 * len(bs)) - 1] if bs else None
        out[k] = {"a": ma[k], "b": mb[k], "diff": mb[k] - ma[k], "ci95": [lo, hi]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--a_label", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--b_label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    result = {"a": args.a, "b": args.b, "a_label": args.a_label,
              "b_label": args.b_label, "n_boot": args.n_boot, "datasets": {}}
    datasets = sorted(set(os.listdir(args.a)) & set(os.listdir(args.b)))
    for ds in datasets:
        da = os.path.join(args.a, ds)
        db = os.path.join(args.b, ds)
        if not (os.path.exists(os.path.join(da, "summary.json"))
                and os.path.exists(os.path.join(db, "summary.json"))):
            continue                       # one side still running
        A, B = load_parts(da, args.a_label), load_parts(db, args.b_label)
        common = sorted(set(A) & set(B))
        by_family = collections.defaultdict(list)
        for key in common:
            by_family[A[key]["family"]].append(key)
        res = {}
        for fam, keys in sorted(by_family.items()):
            res[fam] = compare([A[k] for k in keys], [B[k] for k in keys],
                               args.n_boot, rng)
            if fam == "absent":
                per = collections.defaultdict(list)
                for k in keys:
                    per[k[1]].append(k)
                res["absent_by_category"] = {
                    q: compare([A[k] for k in ks], [B[k] for k in ks],
                               args.n_boot, rng)
                    for q, ks in sorted(per.items())}
        result["datasets"][ds] = res
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    rows = [("vessel", "recall"), ("vessel", "fp_rate"),
            ("vessel", "balanced_accuracy"), ("count", "balanced_accuracy"),
            ("side", "accuracy"), ("absent", "yes_rate"), ("present", "yes_rate")]
    for ds, res in result["datasets"].items():
        print(f"== {ds}")
        for fam, met in rows:
            if fam in res and met in res[fam]:
                x = res[fam][met]
                print(f"   {fam:8s} {met:18s} {args.a_label} {x['a']:.4f}  "
                      f"{args.b_label} {x['b']:.4f}  diff {x['diff']:+.4f} "
                      f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]  "
                      f"(pairs {res[fam]['n_pairs']}, units {res[fam]['n_units']})")


if __name__ == "__main__":
    main()

"""
Split a training corpus by the stock encoder's attention alignment.

This is the paper's visual-culprit test.  Its claim is a causal one: a scene
whose CLS attention is misaligned is not merely correlated with hallucination
but is the step before it.  The way the paper tests that is to split its
training pool by measured Attn-IoU, train one encoder on each half, and compare
the hallucination rates the two produce.  If low-alignment examples are the
culprit, the encoder trained on them should hallucinate more.

The paper ran this on 264 samples, 127 per half.  This module does the same
split on the 10,000-sample corpus, so each half carries 5,000.

The split threshold is the median of the measured scores rather than a fixed
value, which keeps the halves equal in size; an unequal split would confound
the comparison with a difference in training-set size.  Samples whose score is
undefined (no foreground patch, so Attn-IoU is NaN) are dropped and counted,
because they cannot be placed on either side of the threshold.

Usage::

    python -m additional.culprit_split \\
        --data_json data/ok_aivn_10000.json \\
        --scores results/ok_aivn_stock_attn/per_sample_stock.jsonl \\
        --out_low data/ok_aivn_low.json \\
        --out_high data/ok_aivn_high.json
"""
from __future__ import annotations

import argparse
import json
import math
import os


def parse_args():
    p = argparse.ArgumentParser("split a corpus by stock attention alignment")
    p.add_argument("--data_json", required=True)
    p.add_argument("--scores", required=True,
                   help="per_sample JSONL from eval_attn_mp on the same corpus")
    p.add_argument("--out_low", required=True)
    p.add_argument("--out_high", required=True)
    p.add_argument("--metric", default="attn_iou",
                   choices=["attn_iou", "attn_in_box"])
    p.add_argument("--stats_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.data_json) as f:
        samples = json.load(f)

    score = {}
    with open(args.scores) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(args.metric)
            if v is None or (isinstance(v, float) and
                             (math.isnan(v) or math.isinf(v))):
                continue
            score[r["id"]] = float(v)

    scored, unscored = [], 0
    for s in samples:
        if s["id"] in score:
            scored.append(s)
        else:
            unscored += 1

    if not scored:
        raise SystemExit("no sample carries a finite score; check --scores")

    scored.sort(key=lambda s: score[s["id"]])
    half = len(scored) // 2
    low, high = scored[:half], scored[half:]
    threshold = score[scored[half]["id"]] if half < len(scored) else float("nan")

    for path, part in ((args.out_low, low), (args.out_high, high)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sorted(part, key=lambda s: s["id"]), f, ensure_ascii=False)

    def summary(part):
        vals = sorted(score[s["id"]] for s in part)
        return {"n": len(vals),
                "mean": round(sum(vals) / len(vals), 6) if vals else None,
                "median": round(vals[len(vals) // 2], 6) if vals else None,
                "min": round(vals[0], 6) if vals else None,
                "max": round(vals[-1], 6) if vals else None}

    stats = {"data_json": args.data_json, "scores": args.scores,
             "metric": args.metric, "n_samples": len(samples),
             "n_scored": len(scored), "n_unscored_dropped": unscored,
             "threshold": round(threshold, 6) if threshold == threshold else None,
             "low": summary(low), "high": summary(high),
             "out_low": args.out_low, "out_high": args.out_high}

    print(f"[split] {len(samples):,} 샘플 중 점수 있는 것 {len(scored):,}, "
          f"점수 없어 제외 {unscored:,}")
    print(f"[split] 임계값(중앙값) {stats['threshold']}")
    print(f"[split] low  n={stats['low']['n']:,} 평균 {stats['low']['mean']} "
          f"범위 [{stats['low']['min']}, {stats['low']['max']}]")
    print(f"[split] high n={stats['high']['n']:,} 평균 {stats['high']['mean']} "
          f"범위 [{stats['high']['min']}, {stats['high']['max']}]")

    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)),
                    exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[split] stats -> {args.stats_json}")


if __name__ == "__main__":
    main()

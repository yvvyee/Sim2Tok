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
value, which keeps the halves equal in size.  An unequal split would confound
the comparison with a difference in training-set size.  Samples whose score is
undefined are dropped and counted, because they cannot be placed on either side
of the threshold.

The default score is ``attn_in_box``, the share of CLS attention mass that
lands inside a vessel box, not the area-matched ``attn_iou`` the paper used.
The reason is measured rather than preferred.  On this corpus a vessel covers
about four of the 576 patch cells, so the top-four attended cells miss all four
most of the time and ``attn_iou`` is exactly 0 for 90.41% of the 10,000 images,
taking only 25 distinct values.  Its median is 0, which makes the low half one
block of ties and leaves the assignment of tied samples to sort order.
``attn_in_box`` is continuous over the same images, never zero, and separates
the halves by a factor of 4.26 in the mean.

Area normalisation was considered and rejected on the same basis.  Dividing the
mass by the mask's share of the grid removes the trivial advantage a large box
has, but on this corpus it over-corrects: the normalised score correlates -0.391
with box size, while the raw mass correlates only +0.079, so normalising
introduces more size dependence than it removes.

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
    p.add_argument("--metric", default="attn_in_box",
                   choices=["attn_iou", "attn_in_box"],
                   help="attn_in_box 는 상자 안에 떨어진 CLS 주의 질량이며 "
                        "연속값이다. attn_iou 는 전경 패치 수 K 만큼 상위 K 칸을 "
                        "골라 겹침을 보는 값이라, K 가 작은 코퍼스에서는 대부분 "
                        "0 으로 붕괴해 중앙값 분할이 성립하지 않는다")
    p.add_argument("--allow_degenerate", action="store_true",
                   help="두 절반이 분리되지 않아도 진행한다. 임의 부분집합을 "
                        "정렬 높낮이로 보고하게 되므로 기본값은 거부다")
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

    # A median split only separates the halves if the score is not concentrated
    # on the threshold value.  On the 10,000-sample corpus the area-matched
    # Attn-IoU is exactly 0 for 90.41% of images, because a vessel covers about
    # four of the 576 patch cells and the top-four attended cells miss all four
    # most of the time.  The median is then 0, the low half is one block of
    # ties, and which tied sample lands on which side is decided by sort order
    # rather than by alignment.  Training on such halves compares two arbitrary
    # subsets, so the split is refused instead of being reported as low against
    # high.
    n_at_threshold = sum(1 for s in scored if score[s["id"]] == threshold)
    tie_fraction = n_at_threshold / len(scored)
    low_max = max((score[s["id"]] for s in low), default=float("nan"))
    high_min = min((score[s["id"]] for s in high), default=float("nan"))
    degenerate = low_max >= high_min
    if degenerate and not args.allow_degenerate:
        raise SystemExit(
            f"--metric {args.metric} 로는 중앙값 분할이 성립하지 않습니다. "
            f"임계값 {threshold:.6g} 에 전체의 {tie_fraction:.2%} 가 몰려 있어 "
            f"low 최대 {low_max:.6g} 가 high 최소 {high_min:.6g} 이상입니다. "
            f"동점 표본의 배정이 정렬 순서로 결정되므로 두 절반은 정렬 높낮이가 "
            f"아니라 임의 부분집합입니다. 연속적인 척도를 쓰거나 "
            f"--allow_degenerate 로 명시적으로 진행하십시오.")

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
             "n_at_threshold": n_at_threshold,
             "tie_fraction_at_threshold": round(tie_fraction, 6),
             "low_max": round(low_max, 6) if low_max == low_max else None,
             "high_min": round(high_min, 6) if high_min == high_min else None,
             "halves_separated": not degenerate,
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

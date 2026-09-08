"""
Test the visual-culprit link frame by frame, within one encoder.

The paper's mechanism claim is that a misaligned scene is the step before a
hallucination.  Comparing two encoders shows that raising alignment removes
hallucinations, but it cannot separate the mechanism from any other difference
between the two sets of weights.  A within-encoder test can: hold the encoder
fixed, and ask whether the frames it aligns badly are the frames it hallucinates
on.

For one encoder this module joins its per-frame Attn-IoU to its per-frame
hallucination count over the absence probes, then reports

* the hallucination rate in each Attn-IoU quintile, with an exact interval;
* the rate in the lowest decile against the highest, and Fisher's exact test on
  that pair, labelled as assuming frame independence;
* a video-level paired bootstrap on the same contrast, which is the quotable
  form because frames of one video are not independent draws;
* the point-biserial correlation between a frame's alignment and its
  hallucination count.

A frame contributes only if it has both a finite Attn-IoU and at least one
probe.  Both exclusion counts are reported.  The output makes no causal claim
of its own: an association within a fixed encoder is consistent with the
mechanism and does not establish it, since scene difficulty could drive both.

Usage::

    python -m additional.culprit_link \\
        --attn results/full_smd_stock/per_sample_stock.jsonl \\
        --probes results/ok_halluc_culprit/probes_stock__with__logit.jsonl \\
        --out_json results/ok_culprit_link.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

from additional.stats import (clopper_pearson, cluster_bootstrap_paired_diff,
                              fisher_exact_2x2)


def load_attn(path: str, metric: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(metric)
            if v is None or (isinstance(v, float) and
                             (math.isnan(v) or math.isinf(v))):
                continue
            out[r["id"]] = {"score": float(v), "video": r.get("video", r["id"]),
                            "condition": r.get("condition", "")}
    return out


def load_probes(path: str) -> tuple[dict[str, list[int]], dict]:
    """Per-frame hallucination outcomes, absent categories only.

    A probe file may also contain categories that are present in the frame,
    which exist to measure whether the read-out discriminates.  Those records
    carry ``expected_yes == 1`` and their ``halluc`` is 0 by definition, so
    including them would divide every frame's hallucination count by a larger
    denominator and pull all rates toward zero by a constant factor.  Since the
    factor is the same for every frame it would not invert the quintile
    ordering, but it would make the reported rates unreadable as hallucination
    rates, so they are dropped and counted.

    Files written before the control categories existed have no
    ``expected_yes`` field.  Those records are kept, because there the absence
    of the field means every probe was an absent category.
    """
    by_frame: dict[str, list[int]] = defaultdict(list)
    n_kept = n_dropped_present = n_dropped_unknown = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "expected_yes" in r:
                exp = r["expected_yes"]
                if exp == 1:
                    n_dropped_present += 1
                    continue
                if exp is None:
                    n_dropped_unknown += 1
                    continue
            by_frame[r["id"]].append(int(r.get("halluc", 0)))
            n_kept += 1
    return by_frame, {"n_probes_kept_absent": n_kept,
                      "n_probes_dropped_present": n_dropped_present,
                      "n_probes_dropped_unknown_polarity": n_dropped_unknown}


def parse_args():
    p = argparse.ArgumentParser("within-encoder alignment/hallucination link")
    p.add_argument("--attn", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--metric", default="attn_iou",
                   choices=["attn_iou", "attn_in_box"])
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    attn = load_attn(args.attn, args.metric)
    probes, probe_counts = load_probes(args.probes)

    frames = []
    for fid, pr in probes.items():
        a = attn.get(fid)
        if a is None or not pr:
            continue
        frames.append({"id": fid, "score": a["score"], "video": a["video"],
                       "condition": a["condition"],
                       "n_probe": len(pr), "n_halluc": sum(pr)})
    n_probe_only = len(probes) - len(frames)
    n_attn_only = len(attn) - len(frames)

    if len(frames) < 20:
        raise SystemExit(f"only {len(frames)} frames carry both a score and a "
                         f"probe; nothing to test")

    frames.sort(key=lambda f: f["score"])
    n = len(frames)

    def band(lo_i, hi_i, label):
        part = frames[lo_i:hi_i]
        k = sum(f["n_halluc"] for f in part)
        tot = sum(f["n_probe"] for f in part)
        cp = clopper_pearson(k, tot)
        return {"label": label, "n_frames": len(part),
                "score_range": [round(part[0]["score"], 6),
                                round(part[-1]["score"], 6)],
                "mean_score": round(sum(f["score"] for f in part) / len(part), 6),
                "n_probes": tot, "n_halluc": k, "rate": cp.point,
                "ci_lo": cp.lo, "ci_hi": cp.hi}

    quintiles = [band(i * n // 5, (i + 1) * n // 5, f"Q{i + 1}")
                 for i in range(5)]
    low_d = band(0, n // 10, "lowest decile")
    high_d = band(n - n // 10, n, "highest decile")

    p_fisher = fisher_exact_2x2(
        low_d["n_halluc"], low_d["n_probes"] - low_d["n_halluc"],
        high_d["n_halluc"], high_d["n_probes"] - high_d["n_halluc"])

    # Per-frame rates, paired by video, low decile against high.
    lows = frames[:n // 10]
    highs = frames[n - n // 10:]
    by_video_low: dict[str, list[float]] = defaultdict(list)
    by_video_high: dict[str, list[float]] = defaultdict(list)
    for f in lows:
        by_video_low[f["video"]].append(f["n_halluc"] / f["n_probe"])
    for f in highs:
        by_video_high[f["video"]].append(f["n_halluc"] / f["n_probe"])
    shared = sorted(set(by_video_low) & set(by_video_high))
    paired = None
    if len(shared) >= 3:
        a = [sum(by_video_low[v]) / len(by_video_low[v]) for v in shared]
        b = [sum(by_video_high[v]) / len(by_video_high[v]) for v in shared]
        est = cluster_bootstrap_paired_diff(a, b, shared)
        paired = {**est.as_dict(), "n_videos_shared": len(shared),
                  "direction": "high decile minus low decile"}

    # Point-biserial: correlation of a frame's score with its hallucination rate.
    xs = [f["score"] for f in frames]
    ys = [f["n_halluc"] / f["n_probe"] for f in frames]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

    out = {
        "attn_source": args.attn, "probe_source": args.probes,
        "metric": args.metric,
        "n_frames_used": n,
        "n_probe_frames_without_score": n_probe_only,
        "n_scored_frames_without_probe": n_attn_only,
        "probe_selection": probe_counts,
        "quintiles": quintiles,
        "lowest_decile": low_d, "highest_decile": high_d,
        "fisher_low_vs_high_decile": p_fisher,
        "fisher_note": "assumes independent probes; the paired bootstrap below "
                       "accounts for frames clustering by video",
        "paired_by_video_low_vs_high": paired,
        "correlation_score_vs_halluc_rate": (round(corr, 6)
                                             if corr == corr else None),
        "interpretation_note": "an association within a fixed encoder is "
                               "consistent with the visual-culprit mechanism "
                               "and does not establish it; scene difficulty "
                               "could drive both quantities",
    }

    if probe_counts["n_probes_dropped_present"]:
        print(f"[link] 부재 범주 프로브 "
              f"{probe_counts['n_probes_kept_absent']:,}건 사용, "
              f"존재 범주 {probe_counts['n_probes_dropped_present']:,}건 제외")
    print(f"[link] 프레임 {n:,}개 사용 "
          f"(점수 없는 프로브 프레임 {n_probe_only:,}, "
          f"프로브 없는 채점 프레임 {n_attn_only:,})")
    print(f"{'구간':<16}{'프레임':>8}{'평균 정렬':>10}{'프로브':>9}"
          f"{'환각':>8}{'환각률':>9}  95% 구간")
    for q in quintiles:
        print(f"{q['label']:<16}{q['n_frames']:>8,}{q['mean_score']:>10.4f}"
              f"{q['n_probes']:>9,}{q['n_halluc']:>8,}"
              f"{q['rate'] * 100:>8.3f}%  "
              f"[{q['ci_lo'] * 100:.3f}%, {q['ci_hi'] * 100:.3f}%]")
    print()
    print(f"최저 10% 환각률 {low_d['rate'] * 100:.3f}%  "
          f"최고 10% {high_d['rate'] * 100:.3f}%  "
          f"Fisher p = {p_fisher:.4g} (프레임 독립 가정)")
    if paired:
        print(f"비디오 단위 짝지은 차이(최고 − 최저) "
              f"{paired['point'] * 100:+.3f}%p "
              f"[{paired['lo'] * 100:+.3f}, {paired['hi'] * 100:+.3f}] "
              f"(공통 비디오 {paired['n_videos_shared']}편)")
    print(f"정렬-환각률 상관 {out['correlation_score_vs_halluc_rate']}")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),
                    exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[link] -> {args.out_json}")


if __name__ == "__main__":
    main()

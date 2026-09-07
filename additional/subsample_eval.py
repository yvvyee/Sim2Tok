"""
Build a video-stratified subsample of an evaluation set.

The full-SMD pass over 29,253 frames is the right basis for the headline
stock-versus-GAA comparison, but running it once per configuration would spend
most of the compute re-measuring near-duplicate frames: consecutive frames of
one video are highly correlated, so the marginal information per extra frame
inside a video is small while the cost is not.

For sweeps (the scaling curve, the lambda sweep, the control baselines) this
module draws a fixed number of frames per video, evenly spaced through the
video so the sample spans its whole duration.  Every video that appears in the
full set also appears in the subsample, so the video-level interval keeps the
same number of clusters and stays comparable with the full pass.

The selection depends only on the input file and ``--per_video``, never on a
random seed, so every configuration in a sweep is scored on exactly the same
frames.

Usage::

    python -m additional.subsample_eval \\
        --in_json  data/smd_full.json \\
        --out_json data/smd_sweep_60.json \\
        --per_video 60
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def stratify(samples: list[dict], per_video: int) -> list[dict]:
    """Evenly spaced ``per_video`` frames from each video, order preserved."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_video[s.get("video", s["id"])].append(s)

    picked: list[dict] = []
    for video in sorted(by_video):
        frames = sorted(by_video[video], key=lambda s: s.get("frame", 0))
        n = len(frames)
        if n <= per_video:
            picked.extend(frames)
            continue
        # Evenly spaced indices spanning the whole video, endpoints included.
        step = (n - 1) / (per_video - 1) if per_video > 1 else 0
        idx = sorted({int(round(i * step)) for i in range(per_video)})
        picked.extend(frames[i] for i in idx)
    return picked


def parse_args():
    p = argparse.ArgumentParser("video-stratified evaluation subsample")
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--per_video", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.in_json) as f:
        samples = json.load(f)
    picked = stratify(samples, args.per_video)

    with open(args.out_json, "w") as f:
        json.dump(picked, f)

    n_videos = len({s.get("video", s["id"]) for s in picked})
    print(f"[subsample] {len(samples):,} -> {len(picked):,} frames "
          f"from {n_videos} videos (per_video={args.per_video})")
    print(f"[subsample] by condition {dict(Counter(s.get('condition','') for s in picked))}")
    print(f"[subsample] by split     {dict(Counter(s.get('split','') for s in picked))}")
    print(f"[subsample] -> {args.out_json}")


if __name__ == "__main__":
    main()

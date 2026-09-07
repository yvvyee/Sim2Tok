"""
Quantify what the encoder's preprocessing does to the privileged boxes.

``CLIPImageProcessor`` for ViT-L/14-336 resizes the shortest edge to 336 and
then centre-crops to 336x336.  For a 1920x1080 frame that keeps only the
horizontal band ``[0.2188, 0.7812]`` of the original normalised width and
discards 43.75% of the frame, verified by feeding marked columns through the
processor.

Both the submitted ``gaa/evaluate_attn.py`` and the training path normalise
boxes against the *full* frame and hand them to ``compute_geometric_mask``,
which lays them on the patch grid as if the grid spanned the whole frame.  Two
consequences follow, and this script measures both on real data rather than
arguing about them:

1. a box entirely outside the kept band is in the target mask but not in the
   pixels the encoder sees, so no attention can ever land on it;
2. a box inside the band sits at the wrong grid column, because the band is
   stretched to fill the grid.  A box at normalised ``x`` belongs at
   ``(x - 0.2188) / 0.5625``, not at ``x``.

The audit reports, per corpus: how many boxes are fully outside, partly
outside, and fully inside, and how far the naive placement is from the correct
one.  It changes no measurement by itself; the corrected geometry is offered as
a separate evaluation so the paper-comparable numbers stay comparable.

Usage::

    python -m additional.crop_audit --data_path data/smd_full.json \\
        --image_w 1920 --image_h 1080 --out_json results/crop_audit_smd.json
"""
from __future__ import annotations

import argparse
import json
import os


def kept_band(image_w: int, image_h: int, crop: int = 336) -> tuple[float, float]:
    """Normalised horizontal range the centre crop keeps (or vertical if tall).

    Returns ``(lo, hi)`` on the longer axis; the shorter axis is kept whole.
    """
    short = min(image_w, image_h)
    scale = crop / short
    long_scaled = max(image_w, image_h) * scale
    if long_scaled <= crop:
        return 0.0, 1.0
    lo = ((long_scaled - crop) / 2) / long_scaled
    return lo, 1.0 - lo


def audit(samples: list[dict], image_w: int, image_h: int) -> dict:
    lo, hi = kept_band(image_w, image_h)
    horizontal = image_w >= image_h
    span = hi - lo

    n_box = 0
    n_scored = 0
    outside = 0
    partial = 0
    inside = 0
    shift_sum = 0.0
    shift_max = 0.0
    frames_all_outside = 0
    frames_any_outside = 0

    for s in samples:
        boxes = s.get("bboxes") or []
        if not boxes:
            continue
        f_out = 0
        for b in boxes:
            n_box += 1
            a, c = (b[0], b[2]) if horizontal else (b[1], b[3])
            if c <= lo or a >= hi:
                outside += 1
                f_out += 1
                # No placement error is defined for a box the encoder cannot
                # see at all: its "correct" position would be negative or past
                # one.  Averaging those in would inflate the statistic with the
                # very boxes the neighbouring column already reports as unseen,
                # and the inflated mean would then be quoted as how far the
                # *surviving* boxes sit from where they appear.
                continue
            if a < lo or c > hi:
                partial += 1
            else:
                inside += 1
            mid = (a + c) / 2
            correct = (mid - lo) / span
            shift = abs(correct - mid)
            shift_sum += shift
            shift_max = max(shift_max, shift)
            n_scored += 1
        if f_out == len(boxes):
            frames_all_outside += 1
        if f_out:
            frames_any_outside += 1

    return {
        "image_size": [image_w, image_h],
        "kept_band": [round(lo, 4), round(hi, 4)],
        "discarded_fraction": round(1 - span, 4),
        "axis": "horizontal" if horizontal else "vertical",
        "n_frames": sum(1 for s in samples if s.get("bboxes")),
        "n_boxes": n_box,
        "boxes_fully_outside": outside,
        "boxes_partly_outside": partial,
        "boxes_fully_inside": inside,
        "boxes_fully_outside_pct": round(100 * outside / max(n_box, 1), 3),
        "frames_with_any_box_outside": frames_any_outside,
        "frames_with_every_box_outside": frames_all_outside,
        "n_boxes_scored_for_placement": n_scored,
        "mean_centre_placement_error_surviving": round(
            shift_sum / max(n_scored, 1), 4),
        "max_centre_placement_error_surviving": round(shift_max, 4),
        "note": ("placement error is in normalised units on the cropped axis "
                 "and is averaged over surviving boxes only (inside + partly "
                 "outside); 0.0417 equals one patch column of a 24-wide grid"),
    }


def parse_args():
    p = argparse.ArgumentParser("audit box survival through the centre crop")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_w", type=int, default=1920)
    p.add_argument("--image_h", type=int, default=1080)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.data_path) as f:
        samples = json.load(f)
    r = audit(samples, args.image_w, args.image_h)

    print(f"[crop-audit] {os.path.basename(args.data_path)}")
    print(f"  입력 {r['image_size']}, 유지 대역 {r['kept_band']} "
          f"({r['axis']}), 폐기 {r['discarded_fraction'] * 100:.2f}%")
    print(f"  프레임 {r['n_frames']:,}  박스 {r['n_boxes']:,}")
    print(f"  완전히 잘려나간 박스 {r['boxes_fully_outside']:,} "
          f"({r['boxes_fully_outside_pct']}%)")
    print(f"  일부만 남은 박스     {r['boxes_partly_outside']:,}")
    print(f"  온전히 남은 박스     {r['boxes_fully_inside']:,}")
    print(f"  박스가 하나라도 잘린 프레임 {r['frames_with_any_box_outside']:,}")
    print(f"  모든 박스가 잘린 프레임     {r['frames_with_every_box_outside']:,}")
    print(f"  중심 배치 오차 평균 {r['mean_centre_placement_error_surviving']} "
          f"(최대 {r['max_centre_placement_error_surviving']}), "
          f"생존 박스 {r['n_boxes_scored_for_placement']:,}개 기준, "
          f"24칸 격자에서 1칸 = 0.0417")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),
                    exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[crop-audit] -> {args.out_json}")


if __name__ == "__main__":
    main()

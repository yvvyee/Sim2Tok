"""
Vessel-free and vessel-containing sub-crops of the AIVN-SDS simulator frames,
plus the fixed train/validation split of the 10,000 source frames.

Why crops.  Every AIVN frame shows at least one target vessel, and every one
still does after the CLIP centre crop.  A VLM trained on those frames alone
never sees a correct "no vessel" answer, so a vessel-absence probe on real
imagery would measure a constant.  Cutting windows that avoid every box gives
simulator-only negatives; nothing real enters.

Why positives at the same scale.  A negative crop is a zoomed-in view.  If only
negatives were zoomed, "zoomed means no vessel" would be a shortcut.  Every
image therefore contributes, when possible, one negative and one positive
window drawn from the same size distribution.

Rules, all in source-frame pixels:

* Window: a square of side ``s`` in ``--sizes`` (fractions of frame height).
* Horizon: the median box-centre height of the frame stands in for the horizon.
  It must fall inside the central ``--horizon_band`` of the window's height, so
  negatives show sea and horizon rather than open sky.
* Negative: no vessel box intersects the window dilated by ``--margin`` px.
* Positive: at least one box lies fully inside, and no box is cut by the edge.
  Partly visible vessels make the label ambiguous, so such windows are skipped.

Split.  Source frames are split once (``--val_frac``, ``--seed``) and every
crop inherits its source frame's split, so no source frame feeds both sides.

    python -m additional.make_sim_crops \\
        --data_path data/ok_aivn_10000.json --image_dir aivn_correct/png \\
        --out_dir aivn_crops --split_out data/aivn_split.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics

from PIL import Image


def to_px(b, W, H):
    return [b[0] * W, b[1] * H, b[2] * W, b[3] * H]


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def inside(box, win):
    return (box[0] >= win[0] and box[1] >= win[1]
            and box[2] <= win[2] and box[3] <= win[3])


def sample_window(rng, W, H, side, horizon_y, band):
    """A square window whose central ``band`` of height contains the horizon."""
    half = side * band / 2
    lo_top = max(0.0, horizon_y - side / 2 - half)
    hi_top = min(H - side, horizon_y - side / 2 + half)
    if hi_top < lo_top or side > W or side > H:
        return None
    top = rng.uniform(lo_top, hi_top)
    left = rng.uniform(0, W - side)
    return [left, top, left + side, top + side]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_out", required=True)
    p.add_argument("--sizes", default="0.3333,0.5,0.6667",
                   help="square side as a fraction of frame height")
    p.add_argument("--horizon_band", type=float, default=0.6)
    p.add_argument("--margin", type=float, default=8.0)
    p.add_argument("--tries", type=int, default=60)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    data = json.load(open(args.data_path))
    ids = sorted(r["id"] for r in data)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(ids) * args.val_frac))
    val = set(shuffled[:n_val])
    split = {i: ("val" if i in val else "train") for i in ids}
    with open(args.split_out, "w") as f:
        json.dump({"seed": args.seed, "val_frac": args.val_frac,
                   "n_train": len(ids) - n_val, "n_val": n_val,
                   "split": split}, f)

    fracs = [float(x) for x in args.sizes.split(",")]
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    records = []
    stats = collections.Counter()

    for r in data:
        path = os.path.join(args.image_dir, r["image"])
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            boxes = [to_px(b, W, H) for b in r["bboxes"]]
            if not boxes:
                stats["no_box_frame"] += 1
                continue
            horizon = statistics.median((b[1] + b[3]) / 2 for b in boxes)
            dil = [[b[0] - args.margin, b[1] - args.margin,
                    b[2] + args.margin, b[3] + args.margin] for b in boxes]

            for kind in ("neg", "pos"):
                frac = rng.choice(fracs)
                side = frac * H
                win = None
                for _ in range(args.tries):
                    w = sample_window(rng, W, H, side, horizon, args.horizon_band)
                    if w is None:
                        break
                    hit = [intersects(d, w) for d in dil]
                    if kind == "neg":
                        if not any(hit):
                            win = w
                            break
                    else:
                        full = [inside(b, w) for b in boxes]
                        cut = [intersects(b, w) and not f for b, f in zip(boxes, full)]
                        if any(full) and not any(cut):
                            win = w
                            break
                if win is None:
                    stats[f"{kind}_not_found"] += 1
                    continue
                crop = im.crop(tuple(int(round(v)) for v in win))
                cid = f"{r['id']}__{kind}"
                rel = os.path.join("images", cid + ".png")
                crop.save(os.path.join(args.out_dir, rel))
                cw, ch = crop.size
                vis = [b for b in boxes if inside(b, win)] if kind == "pos" else []
                nb = [[round((b[0] - win[0]) / cw, 6), round((b[1] - win[1]) / ch, 6),
                       round((b[2] - win[0]) / cw, 6), round((b[3] - win[1]) / ch, 6)]
                      for b in vis]
                records.append({
                    "id": cid, "image": rel, "width": cw, "height": ch,
                    "source_id": r["id"], "split": split[r["id"]],
                    "kind": kind, "side_frac": frac,
                    "window_px": [round(v, 1) for v in win],
                    "bboxes": nb, "n_vessels": len(nb),
                    "scenario": r.get("scenario"),
                })
                stats[f"{kind}_{split[r['id']]}"] += 1
                stats[f"{kind}_size_{frac}"] += 1

    with open(os.path.join(args.out_dir, "crops.json"), "w") as f:
        json.dump(records, f)
    summary = {"source_frames": len(data), "crops": len(records),
               "stats": dict(sorted(stats.items())),
               "args": vars(args)}
    with open(os.path.join(args.out_dir, "crops_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

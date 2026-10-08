"""
Cue-perturbed copies of the AIVN validation split for the texture test.

Two sets are written, each in the ``real_items`` JSON format that
``eval_token_probe --real`` reads, so every probe run scores them alongside
the clean validation split:

``aivn_val_tex``    texture suppressed, shape kept.  The whole image goes
                    through an edge-preserving recursive filter
                    (``cv2.edgePreservingFilter``, sigma_s=60, sigma_r=0.4),
                    which flattens hull, water and sky texture while keeping
                    object outlines.
``aivn_val_shape``  shape disrupted, texture kept.  Inside every vessel box
                    whose sides are both at least ``--min_side`` source pixels,
                    the box is cut into a 3x3 grid of blocks that are permuted
                    (seeded per image, never the identity).  Local texture
                    survives; the silhouette does not.  Smaller boxes are left
                    unchanged and counted.  Each record carries ``n_shuffled`` so that the
                    comparison can be restricted to images whose vessels were
                    actually shuffled.

Images and boxes are those of the clean split, so a probe's score drop from
``aivn_val`` to each set measures how much it relies on that cue.

    python -m additional.make_cue_sets --frames_json data/ok_aivn_10000.json \\
        --split_json data/aivn_split.json --crops_json aivn_crops/crops.json \\
        --frame_dir aivn_correct/png --crop_dir aivn_crops --out_dir cue_sets
"""
from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import Pool

import cv2
import numpy as np


def items(args):
    split = json.load(open(args.split_json))["split"]
    out = []
    for r in json.load(open(args.frames_json)):
        if split[r["id"]] == "val":
            out.append({"id": r["id"], "src": os.path.join(args.frame_dir, r["image"]),
                        "width": 1920, "height": 1080, "bboxes": r["bboxes"],
                        "video": r["id"]})
    for c in json.load(open(args.crops_json)):
        if c["split"] == "val":
            out.append({"id": c["id"], "src": os.path.join(args.crop_dir, c["image"]),
                        "width": c["width"], "height": c["height"],
                        "bboxes": c["bboxes"], "video": c["source_id"]})
    return out


def shuffle_box(img, box, rng):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(box[0] * w)), int(round(box[1] * h)),
                      int(round(box[2] * w)), int(round(box[3] * h)))
    bw, bh = (x2 - x1) // 3, (y2 - y1) // 3
    blocks = [img[y1 + r * bh:y1 + (r + 1) * bh, x1 + c * bw:x1 + (c + 1) * bw].copy()
              for r in range(3) for c in range(3)]
    order = list(range(9))
    while order == list(range(9)):
        rng.shuffle(order)
    for k, src in enumerate(order):
        r, c = divmod(k, 3)
        img[y1 + r * bh:y1 + (r + 1) * bh, x1 + c * bw:x1 + (c + 1) * bw] = blocks[src]


def work(job):
    it, out_dir, min_side = job
    img = cv2.imread(it["src"], cv2.IMREAD_COLOR)
    if img is None:
        return it["id"], None
    name = it["id"] + ".png"
    tex = cv2.edgePreservingFilter(img, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
    cv2.imwrite(os.path.join(out_dir, "tex", name), tex)
    shp = img.copy()
    rng = random.Random(it["id"])
    n_sh = n_small = 0
    h, w = img.shape[:2]
    for b in it["bboxes"]:
        if (b[2] - b[0]) * w >= min_side and (b[3] - b[1]) * h >= min_side:
            shuffle_box(shp, b, rng); n_sh += 1
        else:
            n_small += 1
    cv2.imwrite(os.path.join(out_dir, "shape", name), shp)
    return it["id"], (name, n_sh, n_small)


def main():
    p = argparse.ArgumentParser()
    for k in ("frames_json", "split_json", "crops_json", "frame_dir", "crop_dir", "out_dir"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--min_side", type=int, default=24)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    for sub in ("tex", "shape"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    its = items(args)
    with Pool(args.workers) as pool:
        res = dict(pool.map(work, [(it, args.out_dir, args.min_side) for it in its], chunksize=16))
    tex, shape = [], []
    n_sh = n_small = n_bad = 0
    for it in its:
        r = res[it["id"]]
        if r is None:
            n_bad += 1
            continue
        name, a, b = r
        n_sh += a; n_small += b
        base = {"id": it["id"], "width": it["width"], "height": it["height"],
                "bboxes": it["bboxes"], "video": it["video"], "vessel_labelled": True,
                "n_shuffled": a, "n_small": b}
        tex.append(dict(base, image=os.path.join("tex", name), dataset="aivn_val_tex"))
        shape.append(dict(base, image=os.path.join("shape", name), dataset="aivn_val_shape"))
    json.dump(tex, open(os.path.join(args.out_dir, "aivn_val_tex.json"), "w"))
    json.dump(shape, open(os.path.join(args.out_dir, "aivn_val_shape.json"), "w"))
    stats = {"images": len(tex), "unreadable": n_bad, "boxes_shuffled": n_sh,
             "boxes_too_small": n_small, "min_side": args.min_side}
    json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=2)
    print("[cue]", stats)


if __name__ == "__main__":
    main()

"""
Export a GAA JSON set to Ultralytics YOLO layout.

The detector comparison exists to scope the paper's claim, not to win it: a
specialist detector localises vessels far better than any attention read-out,
and the study reports that gap rather than hiding it.  To keep the comparison
fair the detector has to see exactly the same supervision as GAA, so this
exporter reads the same JSON files the geometric loss trains on and writes them
in the layout ``ultralytics`` expects::

    <out_root>/images/<split>/<name>.png      (symlink to the original frame)
    <out_root>/labels/<split>/<name>.txt      (class cx cy w h, normalised)
    <out_root>/data.yaml

Images are symlinked rather than copied.  The AIVN corpus is 18 GB and the SMD
frame set is another 46 GB; duplicating either to satisfy a directory
convention would waste disk for no gain.

A single class, ``vessel``, is used throughout, matching the paper's setup.

Usage::

    python -m additional.to_yolo \\
        --train_json data/aivn_10k_describe.json \\
        --train_images /home/yvvyee/data/AIVN-SDS/20260728/png \\
        --val_json   data/smd_full.json \\
        --val_images frames \\
        --out_root   yolo_10k
"""
from __future__ import annotations

import argparse
import json
import os

CLASS_NAMES = ["vessel"]


def to_yolo_line(box: list[float], cls: int = 0) -> str:
    """``[x1, y1, x2, y2]`` normalised -> ``cls cx cy w h`` normalised."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_split(samples: list[dict], image_dir: str, out_root: str,
                 split: str, keep_empty: bool = False) -> dict:
    """Write one split.

    ``keep_empty`` writes a frame that has no box as an image plus an empty
    label file, which is how YOLO represents a true negative.  Dropping such
    frames from a validation split means mAP never penalises a detection on
    open water, so the reported precision is an optimistic bound and nothing in
    the export tells a reader the val set is smaller than the frame set it
    names.  Training splits do not need them, so the default is to drop and
    count.
    """
    img_dir = os.path.join(out_root, "images", split)
    lbl_dir = os.path.join(out_root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_written = 0
    n_boxes = 0
    missing = 0
    n_empty_kept = 0
    n_empty_dropped = 0
    for s in samples:
        boxes = s.get("bboxes") or []
        if not boxes:
            if not keep_empty:
                n_empty_dropped += 1
                continue
            n_empty_kept += 1
        src = s["image"]
        if not os.path.isabs(src):
            src = os.path.join(image_dir, src)
        if not os.path.exists(src):
            missing += 1
            continue
        # A flat name: SMD records carry a "SPLIT/name.png" relative path.
        stem = s["id"].replace("/", "_")
        ext = os.path.splitext(src)[1] or ".png"
        link = os.path.join(img_dir, stem + ext)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(src), link)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for b in boxes:
                f.write(to_yolo_line(b) + "\n")
        n_written += 1
        n_boxes += len(boxes)

    return {"split": split, "n_images": n_written, "n_boxes": n_boxes,
            "n_missing_images": missing,
            "n_empty_frames_kept": n_empty_kept,
            "n_empty_frames_dropped": n_empty_dropped,
            "keep_empty": keep_empty}


def parse_args():
    p = argparse.ArgumentParser("GAA JSON -> YOLO dataset layout")
    p.add_argument("--train_json", required=True)
    p.add_argument("--train_images", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--stats_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    stats = []
    for split, js, imgs in (("train", args.train_json, args.train_images),
                            ("val", args.val_json, args.val_images)):
        with open(js) as f:
            samples = json.load(f)
        info = export_split(samples, imgs, args.out_root, split,
                            keep_empty=(split == "val"))
        info["source_json"] = js
        info["source_images"] = imgs
        stats.append(info)
        print(f"[yolo] {split:5s} {info['n_images']:,} images, "
              f"{info['n_boxes']:,} boxes"
              + (f", {info['n_missing_images']} missing"
                 if info["n_missing_images"] else "")
              + (f", 빈 프레임 유지 {info['n_empty_frames_kept']:,}"
                 if info["n_empty_frames_kept"] else "")
              + (f", 빈 프레임 제외 {info['n_empty_frames_dropped']:,}"
                 if info["n_empty_frames_dropped"] else ""))

    yaml_path = os.path.join(args.out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out_root)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"[yolo] -> {yaml_path}")

    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)),
                    exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump({"out_root": os.path.abspath(args.out_root),
                       "classes": CLASS_NAMES, "splits": stats}, f, indent=2)
        print(f"[yolo] stats -> {args.stats_json}")


if __name__ == "__main__":
    main()

"""
Full Singapore Maritime Dataset (SMD) -> GAA evaluation JSON converter.

``gaa/smd_to_gaa.py`` samples a fixed number of frames per video, which is what
the submitted paper used (8 frames from each of 36 VIS_Onshore videos, 288
images).  This converter instead emits **every labelled frame**, so the
follow-up study can report on the whole split rather than a 1.6% sample.

Layout produced by the published archives::

    <root>/<SPLIT>/Videos/MVI_xxxx_<SPLIT-tag>.avi
    <root>/<SPLIT>/ObjectGT/MVI_xxxx_<SPLIT-tag>_ObjectGT.mat
    <root>/<SPLIT>/HorizonGT/...          (unused here)
    <root>/<SPLIT>/TrackGT/...            (unused here)

Not every video carries object ground truth.  In the published archives
VIS_Onshore has 40 videos but 36 ObjectGT files, and NIR has 30 videos but 23,
so a converter that walks ``Videos/`` would silently produce unlabelled frames.
This one walks ``ObjectGT/`` and reports the videos it had to skip.

Each record carries the fields the statistics module needs to treat a video as
the unit of independence::

    {"id", "image", "bboxes", "split", "video", "frame", "condition"}

Frames are written as lossless PNG so that no image-encoding difference can
confound a before/after attention comparison.  Expect roughly 1.6 MB per frame.

Usage::

    python -m additional.smd_full_to_gaa \\
        --smd_root /mnt/ssd1/yvvyee/gaa_additional/SMD \\
        --split VIS_Onshore NIR \\
        --out_json /mnt/ssd1/yvvyee/gaa_additional/data/smd_full.json \\
        --out_images /mnt/ssd1/yvvyee/gaa_additional/frames \\
        --workers 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ALL_SPLITS = ("VIS_Onshore", "VIS_Onboard", "NIR")
VIDEO_EXTS = (".avi", ".AVI", ".mp4", ".mov")


def condition_of(split: str, stem: str) -> str:
    """Imaging condition tag for a video, from the split and the file name.

    The published VIS_Onshore archive marks its hazy captures in the file name
    (``MVI_1448_VIS_Haze``); everything else in a VIS split is clear daylight,
    and the NIR split is night/near-infrared throughout.
    """
    if split == "NIR":
        return "NIR"
    if "haze" in stem.lower():
        return "VIS_Haze"
    return "VIS_Clear"


# SMD ObjectGT class ids, read from the numeric ``Object`` field.  The string
# ``ObjectType`` field is inconsistent in the published archives (empty or
# disagreeing entries), so the id is authoritative.  Counts over VIS_Onshore and
# NIR: 1 Ferry, 2 Buoy, 3 Vessel/ship, 4 Speed boat, 5 Boat, 6 Kayak,
# 7 Sail boat, 9 Flying bird/plane, 10 Other (id 8, swimming person, does not
# occur; one box has id 0).
SMD_CLASSES = {1: "Ferry", 2: "Buoy", 3: "Vessel/ship", 4: "Speed boat", 5: "Boat",
               6: "Kayak", 7: "Sail boat", 8: "Swimming person",
               9: "Flying bird/plane", 10: "Other"}
SMD_VESSEL_IDS = (1, 3, 4, 5, 6, 7)


def load_object_gt(mat_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-frame ``(N, 4)`` ``[x, y, w, h]`` pixel boxes and their ``(N,)`` class ids.

    The boxes live in the ``BB`` field of the ``structXML`` struct array and the
    class ids in the sibling ``Object`` field.  Reading ``Object`` as geometry
    is the mistake ``gaa/smd_to_gaa.py`` documents.
    """
    import scipy.io as sio

    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for fr in np.atleast_1d(m["structXML"]):
        bb = getattr(fr, "BB", None)
        if bb is None or np.size(bb) == 0:
            frames.append((np.zeros((0, 4)), np.zeros((0,), dtype=int)))
            continue
        bb = np.atleast_2d(np.asarray(bb, dtype=float))
        if bb.shape[1] != 4 and bb.shape[0] == 4:      # tolerate a transposed store
            bb = bb.T
        cls = np.atleast_1d(np.asarray(getattr(fr, "Object", []), dtype=int))
        if len(cls) != len(bb):
            raise ValueError(f"{mat_path}: {len(bb)} boxes but {len(cls)} class ids")
        frames.append((bb, cls))
    return frames


def find_video(videos_dir: str, stem: str) -> str | None:
    for ext in VIDEO_EXTS:
        hit = glob.glob(os.path.join(videos_dir, stem + ext))
        if hit:
            return hit[0]
    return None


def _normalise(bb: np.ndarray, w: int, h: int, cls: np.ndarray | None = None,
               keep: set[int] | None = None) -> list[list[float]]:
    """Normalised ``[x1, y1, x2, y2]`` boxes, optionally only those whose class id is in ``keep``."""
    boxes: list[list[float]] = []
    for k, b in enumerate(bb):
        if keep is not None and int(cls[k]) not in keep:
            continue
        x, y, bw, bh = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0.0, x / w)
        y1 = max(0.0, y / h)
        x2 = min(1.0, (x + bw) / w)
        y2 = min(1.0, (y + bh) / h)
        if x2 > x1 and y2 > y1:
            boxes.append([round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)])
    return boxes


def convert_video(args_tuple) -> dict:
    """Decode one video and write every labelled frame. Runs in a worker process."""
    split, mat_path, videos_dir, out_images, write_images, vessel_ids = args_tuple
    import cv2

    stem = os.path.basename(mat_path).replace("_ObjectGT.mat", "")
    result = {"split": split, "video": stem, "samples": [], "skip": None,
              "n_gt_frames": 0, "n_frames": 0}

    video = find_video(videos_dir, stem)
    if video is None:
        result["skip"] = "no_video"
        return result

    frames_gt = load_object_gt(mat_path)
    result["n_gt_frames"] = len(frames_gt)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        result["skip"] = "unreadable_video"
        return result
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result["n_frames"] = n_video
    n = min(n_video, len(frames_gt)) if n_video > 0 else len(frames_gt)

    cond = condition_of(split, stem)
    out_dir = os.path.join(out_images, split)
    if write_images:
        os.makedirs(out_dir, exist_ok=True)

    # Sequential decode: seeking per frame is far slower than reading in order.
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        bb, cls = frames_gt[idx]
        if np.size(bb):
            h, w = frame.shape[:2]
            bb = np.atleast_2d(np.asarray(bb, dtype=float))
            # A frame is kept when it carries any valid annotation, so the image
            # set does not depend on the class filter.  Only vessel classes go
            # into ``bboxes``; the rest are kept apart as ``other_boxes``.
            if _normalise(bb, w, h):
                other = set(int(c) for c in cls) - set(vessel_ids)
                boxes = _normalise(bb, w, h, cls, set(vessel_ids))
                name = f"{stem}_f{idx:05d}.png"
                if write_images:
                    path = os.path.join(out_dir, name)
                    if not os.path.exists(path):
                        cv2.imwrite(path, frame)
                result["samples"].append({
                    "id": f"{stem}_f{idx}",
                    "image": os.path.join(split, name),
                    "bboxes": boxes,
                    "other_boxes": _normalise(bb, w, h, cls, other),
                    "vessel_labelled": True,
                    "split": split,
                    "video": stem,
                    "frame": idx,
                    "condition": cond,
                })
        idx += 1
    cap.release()
    return result


def parse_args():
    p = argparse.ArgumentParser("full SMD -> GAA evaluation JSON")
    p.add_argument("--smd_root", required=True, help="extracted SMD root")
    p.add_argument("--split", nargs="+", default=list(ALL_SPLITS),
                   choices=list(ALL_SPLITS))
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_images", required=True,
                   help="frame root; images are written under <root>/<SPLIT>/")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no_images", action="store_true",
                   help="build the JSON only (frames must already exist)")
    p.add_argument("--stats_json", default=None)
    p.add_argument("--vessel_ids", default=",".join(map(str, SMD_VESSEL_IDS)),
                   help="ObjectGT class ids counted as vessels")
    return p.parse_args()


def main():
    args = parse_args()

    jobs = []
    for split in args.split:
        base = os.path.join(args.smd_root, split)
        if not os.path.isdir(base):
            print(f"[smd] split directory missing, skipped: {base}")
            continue
        gt_dir = os.path.join(base, "ObjectGT")
        vid_dir = os.path.join(base, "Videos")
        if not os.path.isdir(gt_dir):
            print(f"[smd] no ObjectGT directory, skipped: {gt_dir}")
            continue
        mats = sorted(glob.glob(os.path.join(gt_dir, "*ObjectGT.mat")))
        n_videos = len([p for p in os.listdir(vid_dir)
                        if p.endswith(VIDEO_EXTS)]) if os.path.isdir(vid_dir) else 0
        print(f"[smd] {split}: {len(mats)} ObjectGT / {n_videos} videos")
        for mat in mats:
            jobs.append((split, mat, vid_dir, args.out_images,
                         not args.no_images,
                         tuple(int(x) for x in args.vessel_ids.split(","))))

    if not jobs:
        raise SystemExit("[smd] nothing to convert")

    samples: list[dict] = []
    per_video: dict[str, int] = {}
    skipped: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_video, j): j[1] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["skip"]:
                skipped[r["video"]] = r["skip"]
                print(f"  [{i}/{len(jobs)}] {r['video']}: skipped ({r['skip']})")
                continue
            samples.extend(r["samples"])
            per_video[f"{r['split']}/{r['video']}"] = len(r["samples"])
            print(f"  [{i}/{len(jobs)}] {r['video']}: "
                  f"{len(r['samples'])} labelled / {r['n_frames']} frames",
                  flush=True)

    samples.sort(key=lambda s: (s["split"], s["video"], s["frame"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(samples, f)

    stats = {
        "splits": args.split,
        "n_samples": len(samples),
        "vessel_ids": args.vessel_ids,
        "n_boxes": sum(len(s["bboxes"]) for s in samples),
        "n_other_boxes": sum(len(s["other_boxes"]) for s in samples),
        "n_vessel_free": sum(1 for s in samples if not s["bboxes"]),
        "n_videos": len(per_video),
        "by_split": dict(Counter(s["split"] for s in samples)),
        "by_condition": dict(Counter(s["condition"] for s in samples)),
        "videos_by_split": dict(Counter(v.split("/")[0] for v in per_video)),
        "frames_per_video": per_video,
        "skipped_videos": skipped,
        "image_format": "png (lossless)",
    }
    print(f"\n[smd] wrote {len(samples):,} labelled frames from "
          f"{len(per_video)} videos -> {args.out_json}")
    print(f"[smd] by split      {stats['by_split']}")
    print(f"[smd] by condition  {stats['by_condition']}")
    print(f"[smd] boxes         {stats['n_boxes']:,}")
    if skipped:
        print(f"[smd] skipped       {skipped}")

    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)), exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[smd] stats -> {args.stats_json}")


if __name__ == "__main__":
    main()

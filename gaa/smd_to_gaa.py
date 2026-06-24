"""
Singapore Maritime Dataset (SMD) -> GAA eval JSON converter.

Builds a REAL-world target-domain evaluation set for sim-to-real testing:
samples frames from SMD videos and pairs them with the ground-truth vessel
bounding boxes from the ObjectGT .mat files, in the same format evaluate_attn.py
consumes ({id, image, bboxes} with normalized [x1,y1,x2,y2]).

SMD layout (after unzip):
  <root>/<SPLIT>/<SPLIT>/Videos/MVI_xxxx_VIS.avi
  <root>/<SPLIT>/<SPLIT>/ObjectGT/MVI_xxxx_VIS_ObjectGT.mat
ObjectGT .mat: variable `structXML`, struct array over frames; field `Object`
is an (N,4) [x, y, w, h] matrix of boxes for that frame (pixel coords).

Usage:
  python gaa/smd_to_gaa.py \
      --smd_root /tmp/smd \
      --split VIS_Onshore \
      --frames_per_video 3 --max_videos 40 \
      --out_json data/smd_vis_onshore_eval.json \
      --out_images /home/ywlee/SSD/smd_eval_frames
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import scipy.io as sio
import cv2


def parse_args():
    p = argparse.ArgumentParser("SMD -> GAA eval converter")
    p.add_argument("--smd_root", required=True, help="extracted SMD root dir")
    p.add_argument("--split", default="VIS_Onshore",
                   choices=["VIS_Onshore", "VIS_Onboard", "NIR"])
    p.add_argument("--frames_per_video", type=int, default=3)
    p.add_argument("--max_videos", type=int, default=40)
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_images", required=True)
    return p.parse_args()


def load_object_gt(mat_path):
    """Return list (per frame) of (N,4) [x,y,w,h] arrays.

    In SMD ObjectGT, the bounding boxes live in the `BB` field (N,4)=[x,y,w,h].
    `Object` is a per-object class id, NOT a box.
    """
    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    frames = []
    for fr in np.atleast_1d(m["structXML"]):
        bb = getattr(fr, "BB", None)
        if bb is None or np.size(bb) == 0:
            frames.append(np.zeros((0, 4)))
            continue
        bb = np.atleast_2d(np.asarray(bb, dtype=float))
        if bb.shape[1] != 4 and bb.shape[0] == 4:   # tolerate transposed (4,N)
            bb = bb.T
        frames.append(bb)
    return frames


def find_video(videos_dir, stem):
    for ext in (".avi", ".mp4", ".mov", ".AVI"):
        cands = glob.glob(os.path.join(videos_dir, stem + ext))
        if cands:
            return cands[0]
    return None


def main():
    args = parse_args()
    base = os.path.join(args.smd_root, args.split, args.split)
    gt_dir = os.path.join(base, "ObjectGT")
    vid_dir = os.path.join(base, "Videos")
    if not os.path.isdir(gt_dir):
        # tolerate single-nesting
        base = os.path.join(args.smd_root, args.split)
        gt_dir = os.path.join(base, "ObjectGT")
        vid_dir = os.path.join(base, "Videos")

    mats = sorted(glob.glob(os.path.join(gt_dir, "*ObjectGT.mat")))[:args.max_videos]
    print(f"[smd] split={args.split}  ObjectGT files={len(mats)}")
    os.makedirs(args.out_images, exist_ok=True)

    samples, n_box = [], 0
    for mat in mats:
        stem = os.path.basename(mat).replace("_ObjectGT.mat", "")
        video = find_video(vid_dir, stem)
        if not video:
            print(f"  [skip] no video for {stem}")
            continue
        frames_gt = load_object_gt(mat)
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(frames_gt)
        n = min(total, len(frames_gt))
        if n == 0:
            cap.release(); continue
        # evenly sample frame indices
        idxs = np.linspace(0, n - 1, args.frames_per_video).astype(int).tolist()
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            boxes = []
            for b in frames_gt[fi]:
                x, y, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                if bw <= 0 or bh <= 0:
                    continue
                x1, y1 = max(0.0, x / w), max(0.0, y / h)
                x2, y2 = min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)
                if x2 > x1 and y2 > y1:
                    boxes.append([round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)])
            if not boxes:
                continue
            img_name = f"{stem}_f{fi:05d}.png"
            cv2.imwrite(os.path.join(args.out_images, img_name), frame)
            samples.append({"id": f"{stem}_f{fi}", "image": img_name, "bboxes": boxes})
            n_box += len(boxes)
        cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(samples, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    print(f"[smd] wrote {len(samples)} frames, {n_box} boxes -> {args.out_json}")
    print(f"[smd] images -> {args.out_images}")


if __name__ == "__main__":
    main()

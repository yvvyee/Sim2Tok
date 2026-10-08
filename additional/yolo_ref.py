"""
Simulator-only YOLOv8s reference for the detection numbers.

The frozen-encoder detection probe reads 24x24 tokens and scores close to zero
AP50 on real data.  This module measures what a standard detector reaches
under the same rule: COCO-initialised YOLOv8s fine-tuned on the AIVN-SDS
training split only (frames, vessel crops and vessel-free crops), model
selection on the AIVN validation split, no real image anywhere.

``export``  writes the YOLO layout (symlinked images, label files).
``eval``    sharded over the allowed GPUs with torchrun.  Every image is cut to
            the centre square the CLIP encoder sees, the detector runs on that
            square, and boxes are compared in the same crop frame as the probe
            (``transform_boxes``).  Rows have the ``eval_det_probe`` format, so
            ``det_compare`` reads them unchanged.

    python -m additional.yolo_ref export --frames_json data/ok_aivn_10000.json \\
        --split_json data/aivn_split.json --crops_json aivn_crops/crops.json \\
        --frame_dir aivn_correct/png --crop_dir aivn_crops --out_root yolo_sim
    CUDA_VISIBLE_DEVICES=2,3,4 torchrun --nproc_per_node 3 -m additional.yolo_ref eval \\
        --weights runs/best.pt --frames_json ... --real lars=data/real/lars.json:/mnt/... \\
        --out_dir results/yolo_ref
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from additional.box_geometry import transform_boxes


def sim_records(args, split_name, with_neg=True):
    split = json.load(open(args.split_json))["split"]
    out = []
    for r in json.load(open(args.frames_json)):
        if split[r["id"]] == split_name:
            out.append({"id": r["id"], "path": os.path.join(args.frame_dir, r["image"]),
                        "w": 1920, "h": 1080, "boxes": r["bboxes"], "video": r["id"]})
    for c in json.load(open(args.crops_json)):
        if c["split"] != split_name or (c["kind"] == "neg" and not with_neg):
            continue
        out.append({"id": c["id"], "path": os.path.join(args.crop_dir, c["image"]),
                    "w": c["width"], "h": c["height"], "boxes": c["bboxes"],
                    "video": c["source_id"]})
    return out


def export(args):
    root = os.path.abspath(args.out_root)
    n = {}
    for sp in ("train", "val"):
        os.makedirs(f"{root}/images/{sp}", exist_ok=True)
        os.makedirs(f"{root}/labels/{sp}", exist_ok=True)
        recs = sim_records(args, sp)
        for r in recs:
            ext = os.path.splitext(r["path"])[1]
            dst = f"{root}/images/{sp}/{r['id']}{ext}"
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(r["path"]), dst)
            with open(f"{root}/labels/{sp}/{r['id']}.txt", "w") as f:
                for x1, y1, x2, y2 in r["boxes"]:
                    x1, y1 = max(0.0, x1), max(0.0, y1)
                    x2, y2 = min(1.0, x2), min(1.0, y2)
                    if x2 > x1 and y2 > y1:
                        f.write(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                                f"{x2 - x1:.6f} {y2 - y1:.6f}\n")
        n[sp] = len(recs)
    with open(f"{root}/data.yaml", "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: vessel\n")
    print("[yolo export]", n)


def real_records(name, json_path, image_dir):
    out = []
    for r in json.load(open(json_path)):
        if r.get("vessel_labelled") is False:
            continue
        out.append({"id": r["id"], "path": os.path.join(image_dir, r["image"]),
                    "w": r.get("width", 1920), "h": r.get("height", 1080),
                    "boxes": r.get("bboxes") or [], "video": r.get("video", r["id"])})
    return out


def evaluate(args):
    import torch
    import torch.distributed as dist
    from PIL import Image
    from ultralytics import YOLO

    from additional.config import assert_policy_or_die
    from additional.eval_det_probe import match, match_center

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    assert_policy_or_die()
    model = YOLO(args.weights)
    os.makedirs(args.out_dir, exist_ok=True)

    sets = [("aivn_val", sim_records(args, "val"))]
    for spec in args.real:
        name, rest = spec.split("=", 1)
        jp, idir = rest.rsplit(":", 1)
        sets.append((name, real_records(name, jp, idir)))
    for name, recs in sets:
        rows = []
        for i in range(rank, len(recs), world):
            r = recs[i]
            try:
                img = Image.open(r["path"]).convert("RGB")
            except (OSError, ValueError):
                continue
            W, H = img.size
            s = min(W, H)
            x0, y0 = (W - s) // 2, (H - s) // 2
            crop = img.crop((x0, y0, x0 + s, y0 + s))
            gts = transform_boxes(r["boxes"], r["w"], r["h"])
            res = model.predict(crop, imgsz=args.imgsz, conf=args.conf, max_det=100,
                                device=local_rank, verbose=False)[0]
            b = res.boxes
            dets = []
            if b is not None and len(b):
                xyxy = (b.xyxy.cpu().numpy() / s).clip(0, 1)
                for sc, bx in sorted(zip(b.conf.cpu().numpy().tolist(), xyxy.tolist()),
                                     key=lambda t: -t[0]):
                    dets.append([sc] + bx)
            rows.append({"id": r["id"], "video": r["video"], "dataset": name,
                         "n_gt": len(gts), "scores": [round(d[0], 5) for d in dets],
                         "tp50": match(dets, gts, 0.5), "tp75": match(dets, gts, 0.75),
                         "tpc": match_center(dets, gts)})
        with open(os.path.join(args.out_dir, f"{name}.rank{rank}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        if distributed:
            dist.barrier()
        if rank == 0:
            n = 0
            with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
                for k in range(world):
                    pp = os.path.join(args.out_dir, f"{name}.rank{k}.jsonl")
                    for line in open(pp):
                        f.write(line); n += 1
                    os.remove(pp)
            print(f"[yolo eval] {name}: {n}/{len(recs)}", flush=True)
    if rank == 0:
        json.dump({"weights": args.weights, "imgsz": args.imgsz, "conf": args.conf,
                   "sets": [n for n, _ in sets]},
                  open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["export", "eval"])
    for k in ("frames_json", "split_json", "crops_json", "frame_dir", "crop_dir"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--out_root")
    p.add_argument("--weights")
    p.add_argument("--real", action="append", default=[])
    p.add_argument("--out_dir")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.001)
    args = p.parse_args()
    export(args) if args.mode == "export" else evaluate(args)


if __name__ == "__main__":
    main()

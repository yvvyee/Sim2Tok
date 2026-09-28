"""
Convert the real maritime evaluation corpora into the record format the
evaluators read.  Evaluation only: nothing written here may enter training.

One record per image::

    {"id", "image", "width", "height", "dataset", "split", "condition",
     "video",            # resampling unit for the bootstrap
     "bboxes",           # vessel boxes, full-frame normalised [x1,y1,x2,y2]
     "other_boxes",      # [{"cls": name, "box": [...]}] for non-vessel objects
     "vessel_labelled",  # False when the corpus does not annotate vessels
     "meta"}             # corpus-specific extras (distance, camera azimuth ...)

``bboxes`` is what the alignment evaluator scores and what the presence and
count questions treat as ground truth, so which classes count as a vessel is a
design decision.  It is a flag per corpus, and the defaults are:

* MCAD: ``power-driven vessel`` and ``sailboat``.  ``obstacle``, ``seamark``
  and ``seaplane`` are not vessels.
* LaRS: ``Boat/ship`` and ``Row boats``.  ``Paddle board``, ``Float`` and the
  rest are not.
* SeaDronesSee ODv2: ``boat`` and ``jetski``, both watercraft.  ``swimmer``,
  ``life_saving_appliances`` and ``buoy`` are not.
* Buoy distance: the corpus labels only navigation buoys, so vessels are
  unlabelled.  A frame with no vessel box is therefore *not* evidence that no
  vessel is present, and ``vessel_labelled`` is False.

The resampling unit follows what each corpus records: the source video for
SeaDronesSee (``source.video``), the key-frame sequence for LaRS (file name
without the frame index), and the image itself for MCAD and the buoy corpus,
which carry no sequence information.  Frame-level resampling narrows an
interval when neighbouring frames are correlated; reports must say which unit
was used.

LaRS contains key frames taken from SMD (file names starting ``smd_``).  They
are kept and tagged ``meta.from_smd`` so a pooled analysis can drop them.

    python -m additional.real_eval_to_gaa --root /mnt/ssd1/yvvyee/realeval \\
        --out_dir /mnt/ssd1/yvvyee/gaa_additional/data/real
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from PIL import Image

MCAD_NAMES = ["obstacle", "power-driven vessel", "sailboat", "seamark",
              "seaplane"]


def norm_xywh(x, y, w, h, W, H):
    return [round(x / W, 6), round(y / H, 6),
            round((x + w) / W, 6), round((y + h) / H, 6)]


def yolo_to_xyxy(cx, cy, w, h):
    return [round(cx - w / 2, 6), round(cy - h / 2, 6),
            round(cx + w / 2, 6), round(cy + h / 2, 6)]


def image_size(path):
    with Image.open(path) as im:
        return im.size


# ------------------------------------------------------------------ MCAD
def convert_mcad(root, vessel_ids):
    base = os.path.join(root, "mcad", "maritime_collision_avoidance_de_gb_nl")
    out = {}
    for modality in ("rgb", "ir"):
        meta = {}
        with open(os.path.join(base, modality, "metadata.txt")) as f:
            header = next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                vals = [None if v == "null" else float(v) for v in parts[1:6]]
                meta[parts[0]] = dict(zip(
                    ["elevation_deg", "azimuth_deg", "zoom_pct",
                     "hfov_rad", "vfov_rad"], vals))
        recs = []
        for split in ("train", "val"):
            idir = os.path.join(base, modality, "images", split)
            for fn in sorted(os.listdir(idir)):
                stem = os.path.splitext(fn)[0]
                W, H = image_size(os.path.join(idir, fn))
                lab = os.path.join(base, modality, "labels", split, stem + ".txt")
                vessels, others = [], []
                for line in open(lab):
                    p = line.split()
                    if len(p) < 5:
                        continue
                    c = int(p[0])
                    box = yolo_to_xyxy(*map(float, p[1:5]))
                    if c in vessel_ids:
                        vessels.append(box)
                    else:
                        others.append({"cls": MCAD_NAMES[c], "box": box})
                recs.append({
                    "id": f"mcad_{modality}_{stem}",
                    "image": os.path.relpath(os.path.join(idir, fn), root),
                    "width": W, "height": H, "dataset": f"mcad_{modality}",
                    "split": split, "condition": modality, "video": stem,
                    "bboxes": vessels, "other_boxes": others,
                    "vessel_labelled": True,
                    "meta": meta.get(stem, {}),
                })
        out[f"mcad_{modality}"] = recs
    return out


# ------------------------------------------------------------------ LaRS
def convert_lars(root, vessel_ids):
    recs = []
    for split in ("train", "val"):
        ann = json.load(open(os.path.join(root, "lars", split,
                                          "panoptic_annotations.json")))
        cats = {c["id"]: c["name"] for c in ann["categories"]}
        things = {c["id"] for c in ann["categories"] if c.get("isthing")}
        segs = {a["image_id"]: a["segments_info"] for a in ann["annotations"]}
        for im in ann["images"]:
            W, H, fn = im["width"], im["height"], im["file_name"]
            vessels, others = [], []
            for s in segs.get(im["id"], []):
                if s["category_id"] not in things:
                    continue
                box = norm_xywh(*s["bbox"], W, H)
                if s["category_id"] in vessel_ids:
                    vessels.append(box)
                else:
                    others.append({"cls": cats[s["category_id"]], "box": box})
            stem = os.path.splitext(fn)[0]
            recs.append({
                "id": f"lars_{stem}",
                "image": os.path.join("lars", split, "images", fn),
                "width": W, "height": H, "dataset": "lars", "split": split,
                "condition": stem.split("_")[0],
                "video": stem.rsplit("_", 1)[0],
                "bboxes": vessels, "other_boxes": others,
                "vessel_labelled": True,
                "meta": {"from_smd": stem.startswith("smd_")},
            })
    return {"lars": recs}


# ------------------------------------------------------------ SeaDronesSee
def convert_sds(root, vessel_ids):
    recs = []
    base = os.path.join(root, "sds", "compressed")
    for split in ("train", "val"):
        ann = json.load(open(os.path.join(base, "annotations",
                                          f"instances_{split}.json")))
        cats = {c["id"]: c["name"] for c in ann["categories"]}
        by_img = collections.defaultdict(list)
        for a in ann["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in ann["images"]:
            W, H = im["width"], im["height"]
            vessels, others = [], []
            for a in by_img.get(im["id"], []):
                if a["category_id"] == 0:          # "ignored" regions
                    continue
                box = norm_xywh(*a["bbox"], W, H)
                if a["category_id"] in vessel_ids:
                    vessels.append(box)
                else:
                    others.append({"cls": cats[a["category_id"]], "box": box})
            src = im.get("source") or {}
            m = im.get("meta") or {}
            recs.append({
                "id": f"sds_{split}_{os.path.splitext(im['file_name'])[0]}",
                "image": os.path.join("sds", "compressed", "images", split,
                                      im["file_name"]),
                "width": W, "height": H, "dataset": "seadronessee",
                "split": split, "condition": src.get("drone", ""),
                "video": src.get("video") or src.get("folder_name")
                         or im["file_name"],
                "bboxes": vessels, "other_boxes": others,
                "vessel_labelled": True,
                "meta": {"altitude_m": m.get("height_above_takeoff(meter)"),
                         "gimbal_pitch_deg": m.get("gimbal_pitch(degrees)"),
                         "gimbal_heading_deg": m.get("gimbal_heading(degrees)"),
                         "compass_heading_deg": m.get("compass_heading(degrees)")},
            })
    return {"seadronessee": recs}


# ------------------------------------------------------------ buoy distance
def convert_buoy(root):
    recs = []
    base = os.path.join(root, "buoy")
    for fn in sorted(os.listdir(os.path.join(base, "images"))):
        stem = os.path.splitext(fn)[0]
        W, H = image_size(os.path.join(base, "images", fn))
        others = []
        lab = os.path.join(base, "labels", stem + ".txt")
        if os.path.exists(lab):
            for line in open(lab):
                p = line.split()
                if len(p) < 6:
                    continue
                others.append({"cls": "nav_buoy",
                               "box": yolo_to_xyxy(*map(float, p[1:5])),
                               "distance_m": float(p[5])})
        recs.append({
            "id": f"buoy_{stem}", "image": os.path.join("buoy", "images", fn),
            "width": W, "height": H, "dataset": "buoy_distance",
            "split": "train", "condition": "", "video": stem,
            "bboxes": [], "other_boxes": others, "vessel_labelled": False,
            "meta": {},
        })
    return {"buoy_distance": recs}


def summarise(name, recs):
    n = len(recs)
    nv = sum(1 for r in recs if r["bboxes"])
    return {"dataset": name, "images": n,
            "images_with_vessel": nv,
            "images_without_vessel": n - nv if recs and recs[0]["vessel_labelled"] else None,
            "vessel_boxes": sum(len(r["bboxes"]) for r in recs),
            "other_boxes": dict(collections.Counter(
                o["cls"] for r in recs for o in r["other_boxes"])),
            "resampling_units": len({r["video"] for r in recs}),
            "sizes": dict(collections.Counter(
                f"{r['width']}x{r['height']}" for r in recs).most_common(6))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mcad_vessel_ids", default="1,2")
    p.add_argument("--lars_vessel_ids", default="11,12")
    p.add_argument("--sds_vessel_ids", default="2,3")
    args = p.parse_args()
    ids = lambda s: {int(x) for x in s.split(",")}

    os.makedirs(args.out_dir, exist_ok=True)
    corpora = {}
    corpora.update(convert_mcad(args.root, ids(args.mcad_vessel_ids)))
    corpora.update(convert_lars(args.root, ids(args.lars_vessel_ids)))
    corpora.update(convert_sds(args.root, ids(args.sds_vessel_ids)))
    corpora.update(convert_buoy(args.root))

    stats = []
    for name, recs in corpora.items():
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(recs, f)
        stats.append(summarise(name, recs))
    stats_meta = {"root": args.root,
                  "vessel_ids": {"mcad": args.mcad_vessel_ids,
                                 "lars": args.lars_vessel_ids,
                                 "seadronessee": args.sds_vessel_ids},
                  "corpora": stats}
    with open(os.path.join(args.out_dir, "real_eval_stats.json"), "w") as f:
        json.dump(stats_meta, f, indent=2)
    print(json.dumps(stats_meta, indent=2))


if __name__ == "__main__":
    main()

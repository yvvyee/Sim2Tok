"""
Per-vessel navigation state for the AIVN-SDS 10k frames, from the simulator CSV.

Each frame has one CSV with the own ship (``my_ship == 1``) and the target
ships: position, heading, speed, size, CPA and TCPA, and the target's pixel box
(top-left x, y, width, height on the 1920x1080 frame).  For every target whose
box matches a box of the training record, this writes

``bearing``     relative bearing from own bow, degrees in (-180, 180]
``dist_nm``     great-circle distance, nautical miles
``rel_course``  target heading minus own heading, degrees in (-180, 180]
``cpa``         closest point of approach as the CSV gives it
``tcpa``        time to CPA as the CSV gives it
``box``         the matched record box, full-frame normalised

Matching is by IoU against the record boxes, which are the CSV boxes
normalised; a target with no record box above ``--min_iou`` is reported, not
guessed.

    python -m additional.build_nav_state \\
        --data_json data/ok_aivn_10000.json --csv_dir aivn_correct/csv \\
        --out_json data/aivn_nav_state.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os

EARTH_NM = 3440.065


def wrap(deg: float) -> float:
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def dist_nm(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_NM * math.asin(math.sqrt(a))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_json", required=True)
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--image_w", type=int, default=1920)
    p.add_argument("--image_h", type=int, default=1080)
    p.add_argument("--min_iou", type=float, default=0.9)
    args = p.parse_args()

    recs = json.load(open(args.data_json))
    out, stats = {}, collections.Counter()
    for r in recs:
        path = os.path.join(args.csv_dir, r["id"] + ".csv")
        if not os.path.exists(path):
            stats["frame_no_csv"] += 1
            continue
        rows = list(csv.DictReader(open(path)))
        own = [x for x in rows if x["my_ship"] == "1"]
        if len(own) != 1:
            stats["frame_own_ship_count_not_1"] += 1
            continue
        o = own[0]
        olat, olon, ohdg = float(o["latitude"]), float(o["longitude"]), float(o["heading"])
        boxes = r.get("bboxes") or []
        used = set()
        vessels = []
        for x in rows:
            if x["my_ship"] == "1":
                continue
            w, h = float(x["bbox_width"]), float(x["bbox_height"])
            if w <= 0 or h <= 0:
                stats["target_no_box"] += 1
                continue
            bx, by = float(x["bbox_x"]), float(x["bbox_y"])
            cb = [bx / args.image_w, by / args.image_h,
                  (bx + w) / args.image_w, (by + h) / args.image_h]
            best, bi = 0.0, -1
            for i, rb in enumerate(boxes):
                if i in used:
                    continue
                v = iou(cb, rb)
                if v > best:
                    best, bi = v, i
            if best < args.min_iou:
                stats["target_box_unmatched"] += 1
                continue
            used.add(bi)
            tlat, tlon = float(x["latitude"]), float(x["longitude"])
            vessels.append({
                "box": boxes[bi],
                "bearing": round(wrap(bearing_deg(olat, olon, tlat, tlon) - ohdg), 3),
                "dist_nm": round(dist_nm(olat, olon, tlat, tlon), 4),
                "rel_course": round(wrap(float(x["heading"]) - ohdg), 3),
                "cpa": float(x["cpa"]), "tcpa": float(x["tcpa"]),
            })
            stats["target_matched"] += 1
        stats["record_box_unmatched"] += len(boxes) - len(used)
        stats["record_boxes"] += len(boxes)
        out[r["id"]] = vessels
        stats["frames"] += 1
    json.dump(out, open(args.out_json, "w"))
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()

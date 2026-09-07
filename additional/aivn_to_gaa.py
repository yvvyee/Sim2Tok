"""
AIVN-SDS (20260728) -> GAA training JSON converter.

The corpus holds 10,000 Unity3D captures, each with four sibling files::

    png/<stem>.png                 1920x1080 capture
    csv/<stem>.csv                 per-ship sensor row (own ship + targets)
    describe_ko/<stem>.describe_ko.txt   Korean situation description
    advice_ko/<stem>.advice_ko.txt       Korean navigational advice

The csv carries the privileged information this study needs.  Every row is one
ship; ``my_ship=1`` marks the own ship, whose bounding box is all zeros because
the own hull is not a visible object in the capture.  Target ships carry pixel
boxes ``bbox_x, bbox_y, bbox_width, bbox_height`` in 1920x1080 coordinates,
which are normalised here to the ``[x1, y1, x2, y2]`` form that
``gaa.geometric_loss.compute_geometric_mask`` consumes.

The file stem also encodes the COLREGs scenario, e.g.
``tango_sds-unity_crossing-20260710-v3.0-0015363a``.  The ``source`` field is
kept as ``scenario`` (``crossing``/``headon``/``overtaking``) and ``collision``
so the COLREGs-reasoning story can score against it without re-parsing names.

Usage::

    python -m additional.aivn_to_gaa \\
        --aivn_root /home/yvvyee/data/AIVN-SDS/20260728 \\
        --out_json  /mnt/ssd2/gaa_additional/data/aivn_10k.json \\
        --task describe
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter

# ── Prompts ──────────────────────────────────────────────────────────────────

#: Instruction used for the situation-description task (describe_ko target).
PROMPT_DESCRIBE = (
    "<image>\n이 해상 영상에서 식별되는 선박과 항해 상황을 설명하시오."
)

#: Instruction used for the navigational-advice task (advice_ko target).
PROMPT_ADVICE = (
    "<image>\n이 해상 상황에서 자선이 취해야 할 항해 조치를 국제해상충돌예방규칙에 "
    "따라 권고하시오."
)

TASKS = {"describe": PROMPT_DESCRIBE, "advice": PROMPT_ADVICE}

#: ``unity_<scenario>[_collision]`` -> (scenario, collision flag)
_SOURCE_RE = re.compile(r"^unity_(crossing|headon|overtaking)(_collision)?$")


def parse_stem(stem: str) -> dict:
    """Split an AIVN-SDS file stem into its documented fields.

    Naming rule (dataset README): ``{proj}-{source}-{yyyymmdd}-v{version}-{shortid}``.
    Returns ``{}`` when the stem does not follow the rule, so callers can skip
    the sample instead of guessing.
    """
    parts = stem.split("-")
    if len(parts) != 5:
        return {}
    proj, source, date, version, shortid = parts
    m = _SOURCE_RE.match(source)
    if not m:
        return {}
    return {
        "proj": proj,
        "source": source,
        "scenario": m.group(1),
        "collision": bool(m.group(2)),
        "date": date,
        "version": version,
        "shortid": shortid,
    }


def read_boxes(csv_path: str, image_w: int, image_h: int) -> list[list[float]]:
    """Target-ship boxes as normalised ``[x1, y1, x2, y2]``.

    Own-ship rows (``my_ship=1``) and degenerate boxes are dropped.  Boxes are
    clamped to the image, which is a no-op on this corpus (a full pass over the
    10,000 csv files found zero out-of-bounds boxes) but keeps a future capture
    batch from producing masks outside the patch grid.
    """
    boxes: list[list[float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("my_ship") == "1":
                continue                       # own hull is not a visible object
            try:
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_width"])
                h = float(row["bbox_height"])
            except (KeyError, TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x / image_w)
            y1 = max(0.0, y / image_h)
            x2 = min(1.0, (x + w) / image_w)
            y2 = min(1.0, (y + h) / image_h)
            if x2 > x1 and y2 > y1:
                boxes.append([round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)])
    return boxes


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def build(aivn_root: str, task: str, image_w: int, image_h: int,
          limit: int, seed: int) -> tuple[list[dict], dict]:
    """Convert the corpus to GAA JSON records plus a statistics dict."""
    png_dir = os.path.join(aivn_root, "png")
    stems = sorted(p[:-4] for p in os.listdir(png_dir) if p.endswith(".png"))

    prompt = TASKS[task]
    target_suffix = {"describe": "describe_ko", "advice": "advice_ko"}[task]

    samples: list[dict] = []
    skipped: Counter[str] = Counter()
    for stem in stems:
        meta = parse_stem(stem)
        if not meta:
            skipped["bad_stem"] += 1
            continue
        csv_path = os.path.join(aivn_root, "csv", f"{stem}.csv")
        txt_path = os.path.join(aivn_root, target_suffix,
                                f"{stem}.{target_suffix}.txt")
        if not os.path.exists(csv_path):
            skipped["no_csv"] += 1
            continue
        if not os.path.exists(txt_path):
            skipped["no_text"] += 1
            continue

        boxes = read_boxes(csv_path, image_w, image_h)
        if not boxes:
            skipped["no_box"] += 1
            continue
        answer = read_text(txt_path)
        if not answer:
            skipped["empty_text"] += 1
            continue

        samples.append({
            "id": stem,
            "image": f"{stem}.png",
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
            "bboxes": boxes,
            "scenario": meta["scenario"],
            "collision": meta["collision"],
        })

    if 0 < limit < len(samples):
        random.Random(seed).shuffle(samples)
        samples = samples[:limit]
        samples.sort(key=lambda s: s["id"])

    n_box = sum(len(s["bboxes"]) for s in samples)
    stats = {
        "task": task,
        "n_png": len(stems),
        "n_samples": len(samples),
        "n_boxes": n_box,
        "boxes_per_sample": round(n_box / len(samples), 4) if samples else 0.0,
        "skipped": dict(skipped),
        "scenario": dict(Counter(s["scenario"] for s in samples)),
        "collision": dict(Counter(str(s["collision"]) for s in samples)),
        "image_size": [image_w, image_h],
        "limit": limit,
        "seed": seed,
    }
    return samples, stats


def parse_args():
    p = argparse.ArgumentParser("AIVN-SDS -> GAA training JSON")
    p.add_argument("--aivn_root", required=True,
                   help="20260728 root holding png/csv/describe_ko/advice_ko")
    p.add_argument("--out_json", required=True)
    p.add_argument("--task", default="describe", choices=sorted(TASKS))
    p.add_argument("--image_w", type=int, default=1920)
    p.add_argument("--image_h", type=int, default=1080)
    p.add_argument("--limit", type=int, default=-1,
                   help="subsample to N records (-1 = keep all); used by the "
                        "scaling-curve story")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the --limit subsample")
    p.add_argument("--stats_json", default=None,
                   help="also write the statistics dict here")
    return p.parse_args()


def main():
    args = parse_args()
    samples, stats = build(args.aivn_root, args.task, args.image_w,
                           args.image_h, args.limit, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"[aivn] wrote {len(samples):,} records -> {args.out_json}")
    print(f"[aivn] boxes {stats['n_boxes']:,} "
          f"({stats['boxes_per_sample']} per sample)")
    print(f"[aivn] scenario {stats['scenario']}")
    if stats["skipped"]:
        print(f"[aivn] skipped {stats['skipped']}")

    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)), exist_ok=True)
        with open(args.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[aivn] stats -> {args.stats_json}")


if __name__ == "__main__":
    main()

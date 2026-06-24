"""
Old-format SDS (20250922 / 20251031) -> GAA geo-pool JSON converter.

These earlier dirs differ from 20260227: each dataset{1..10}/ holds a frame
sequence (frame_1..N.png, 1024x768) plus a single input.csv (multi-ship rows
per frame, pixel bbox). Their instruction-response (output.csv) is the
low-trust legacy format, so we take ONLY images + bboxes (geometry GT) for
geo (attention) training -- text is ignored.

Output (per frame): {id, image(absolute path), bboxes:[[x1,y1,x2,y2] norm]}.
Own-ship rows (my_ship==1) and zero boxes are skipped.

Usage:
  python gaa/sds_old_to_gaa.py \
      --dataset_root /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20250922 \
      --out_json data/_old_20250922.json
"""
import os, sys, json, argparse, glob
import pandas as pd
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser("old-format SDS -> gaa geo json")
    p.add_argument("--dataset_root", required=True, help="a date dir (e.g. .../dataset/20250922)")
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def col(df, name):
    """case-insensitive, underscore/space-insensitive column lookup.
    (Some dirs use '_file_name', different casing like 'Length', etc.)"""
    def norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    target = norm(name)
    for c in df.columns:
        if norm(c) == target:
            return c
    return None


def main():
    args = parse_args()
    tag = os.path.basename(args.dataset_root.rstrip("/"))
    subdirs = sorted(d for d in glob.glob(os.path.join(args.dataset_root, "*")) if os.path.isdir(d))
    samples, n_box, skipped = [], 0, 0

    for sub in subdirs:
        csv = os.path.join(sub, "input.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        c_file = col(df, "file_name"); c_my = col(df, "my_ship")
        c_bx, c_by = col(df, "bbox_x"), col(df, "bbox_y")
        c_bw, c_bh = col(df, "bbox_width"), col(df, "bbox_height")
        if not all([c_file, c_bx, c_by, c_bw, c_bh]):
            print(f"  [skip] {sub}: missing bbox columns"); continue

        for frame_png in sorted(glob.glob(os.path.join(sub, "frame_*.png"))):
            stem = os.path.splitext(os.path.basename(frame_png))[0]   # e.g. frame_3
            try:
                W, H = Image.open(frame_png).size
            except Exception:
                skipped += 1; continue
            rows = df[df[c_file].astype(str) == stem]
            boxes = []
            for _, r in rows.iterrows():
                if c_my is not None and pd.notna(r[c_my]) and int(float(r[c_my])) == 1:
                    continue
                if any(pd.isna(r[c]) for c in (c_bx, c_by, c_bw, c_bh)):
                    continue
                bx, by, bw, bh = float(r[c_bx]), float(r[c_by]), float(r[c_bw]), float(r[c_bh])
                if bw <= 0 or bh <= 0:
                    continue
                x1, y1 = max(0.0, bx / W), max(0.0, by / H)
                x2, y2 = min(1.0, (bx + bw) / W), min(1.0, (by + bh) / H)
                if x2 > x1 and y2 > y1:
                    boxes.append([round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)])
            if not boxes:
                skipped += 1; continue
            samples.append({
                "id": f"{tag}_{os.path.basename(sub)}_{stem}",
                "image": os.path.abspath(frame_png),   # absolute -> use image_dir="/"
                "bboxes": boxes,
            })
            n_box += len(boxes)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(samples, open(args.out_json, "w"), ensure_ascii=False, indent=2)
    print(f"[old->gaa] {tag}: {len(samples)} frames, {n_box} boxes "
          f"(skipped {skipped} no-box) -> {args.out_json}")
    if samples:
        print(f"  sample: id={samples[0]['id']}  boxes={len(samples[0]['bboxes'])}  img={samples[0]['image']}")


if __name__ == "__main__":
    main()

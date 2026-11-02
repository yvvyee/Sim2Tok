"""YOLOv8s reference: AP50, AP75 and center-hit AP per dataset from yolo_ref eval outputs (supplement).

Usage: python -m additional.tables.yolo_ap
"""
import json,sys
from additional.det_compare import ap
import os
ROOT = os.environ.get("GAA_ROOT", "/mnt/ssd1/yvvyee/gaa_additional")
R=f"{ROOT}/results/yolo_ref/eval"
for s in ["aivn_val","smd","mcad_rgb","mcad_ir","lars","seadronessee"]:
    rows=[json.loads(l) for l in open(f"{R}/{s}.jsonl")]
    sc=[x for r in rows for x in r["scores"]]; ng=sum(r["n_gt"] for r in rows)
    out=[ap(sc,[t for r in rows for t in r[k]],ng) for k in ("tp50","tp75","tpc")]
    print(f"{s:13s} n_img {len(rows):6d} n_gt {ng:6d}  AP50 {out[0]:.3f}  AP75 {out[1]:.3f}  center-hit AP {out[2]:.3f}")

"""
Qualitative figure: probe occupancy maps on real images for three encoders.

For each real dataset one image is chosen deterministically (the first record,
in file order, with one to three vessels in the center crop and the largest
vessel covering at least half a patch cell).  Each encoder's seed-0 probe head
from ``eval_token_probe`` predicts token occupancy on the frozen layer-23
tokens.  Runs on CPU.

    CUDA_VISIBLE_DEVICES= python -m additional.fig_occupancy --out fig_occ.png [--smd_json PATH]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from additional.box_geometry import transform_boxes
from additional.train_token_ddp import ProbeHead

ROOT = "/mnt/ssd1/yvvyee/gaa_additional"
REAL = "/mnt/ssd1/yvvyee/realeval"
SETS = [("MCAD-RGB", f"{ROOT}/data/real/mcad_rgb.json", REAL),
        ("MCAD-IR", f"{ROOT}/data/real/mcad_ir.json", REAL),
        ("LaRS", f"{ROOT}/data/real/lars.json", REAL),
        ("SeaDronesSee", f"{ROOT}/data/real/seadronessee.json", REAL),
        ("SMD", None, f"{ROOT}/frames")]   # json from --smd_json
# Encoders of the revision protocol (training split only).  GAA is the
# variant matched to token occupancy in data and steps.
ENC = [("CLIP", "none", "rev_stock_s0"),
       ("GAA", f"{ROOT}/checkpoints/rev/rev_gaam_s0/vision_encoder.bin", "rev_gaam_s0"),
       ("Token occ. (ours)", f"{ROOT}/checkpoints/rev/rev_tok_s0/vision_encoder.bin", "rev_tok_s0")]


def pick(json_path):
    for r in json.load(open(json_path)):
        w, h = r.get("width", 1920), r.get("height", 1080)
        vis = transform_boxes(r.get("bboxes") or [], w, h)
        if 1 <= len(vis) <= 3:
            a = max((b[2] - b[0]) * (b[3] - b[1]) for b in vis)
            if a * 576 >= 0.5:
                return r, vis
    raise SystemExit(f"no image found in {json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--smd_json", default=f"{ROOT}/data/smd_vessel.json",
                   help="SMD labels with vessel classes only (smd_full_to_gaa.py)")
    args = p.parse_args()
    from transformers import CLIPImageProcessor, CLIPVisionModel
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    picks = [(n, *pick(j or args.smd_json), d) for n, j, d in SETS]
    pvs = []
    for n, r, vis, d in picks:
        img = Image.open(os.path.join(d, r["image"])).convert("RGB")
        pvs.append(proc(images=img, return_tensors="pt").pixel_values[0])
    pv = torch.stack(pvs)
    maps = {}
    for label, path, probe in ENC:
        enc = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
        if path != "none":
            enc.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        enc.eval()
        head = ProbeHead(enc.config.hidden_size, [])
        head.load_state_dict(torch.load(f"{ROOT}/results/rev_probe/{probe}/probe_head.bin", map_location="cpu"))
        head.eval()
        with torch.no_grad():
            tok = enc(pixel_values=pv, output_hidden_states=True).hidden_states[-2][:, 1:, :]
            occ = torch.sigmoid(head(tok)[0]).view(-1, 24, 24).numpy()
        maps[label] = occ
    mean = np.array(proc.image_mean)[:, None, None]
    std = np.array(proc.image_std)[:, None, None]
    rows = ["Input + GT"] + [e[0] for e in ENC]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(1.9 * len(picks), 1.9 * len(rows)))
    for i, (n, r, vis, d) in enumerate(picks):
        im = np.clip((pv[i].numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        vmax = max(0.05, max(maps[e[0]][i].max() for e in ENC))   # shared per column
        for j, row in enumerate(rows):
            ax = axes[j, i]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                for b in vis:
                    x1, y1, x2, y2 = [v * 336 for v in b]
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1))
                ax.set_title(n, fontsize=8)
            else:
                ax.imshow(maps[row][i], cmap="magma", alpha=0.6, extent=(0, 336, 336, 0),
                          interpolation="nearest", vmin=0, vmax=vmax)
            if i == 0:
                ax.set_ylabel(row, fontsize=8)
    plt.tight_layout(pad=0.2)
    plt.savefig(args.out, dpi=200)
    json.dump([{"dataset": n, "id": r["id"], "n_visible": len(vis),
                "max_occ": {lab: float(maps[lab][i].max()) for lab in maps}}
               for i, (n, r, vis, d) in enumerate(picks)],
              open(args.out.replace(".png", ".json"), "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()

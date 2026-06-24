"""
Attention heatmap visualization for ALL samples.

For each sample, renders a panel:
  [ original + GT boxes | CLS-attention overlay per encoder (with attn_in_box) ]
so texture bias (no-GAA attention on sea/waves) vs GAA (attention on vessels)
is visible per image. Saves one PNG per sample under --out_dir/<tag>/.

Usage:
  python gaa/attn_visualize.py \
      --data_path data/smd_onshore_big.json --image_dir /home/ywlee/SSD/smd_onshore_big \
      --out_dir /home/ywlee/SSD/gaa_attn_viz/smd_onshore --tag smd_onshore \
      --ckpts stock=stock,noGAA=checkpoints/ctrl_baseline_train80/vision_encoder.bin,GAA=checkpoints/gaa_inboxce_train80/vision_encoder.bin
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gaa.geometric_loss import compute_geometric_mask


def parse_args():
    p = argparse.ArgumentParser("attention heatmap visualization")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="set")
    p.add_argument("--ckpts", required=True,
                   help="comma list label=path (path='stock' for pretrained CLIP)")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--feature_size", type=int, default=-1)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args()


def load_clip(vision_model, ckpt, dtype, device):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    m = CLIPVisionModel.from_pretrained(vision_model, torch_dtype=dtype, attn_implementation="eager")
    proc = CLIPImageProcessor.from_pretrained(vision_model)
    if ckpt and ckpt != "stock" and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        if st and all(k.startswith("model.") for k in st):
            st = {k[len("model."):]: v for k, v in st.items()}
        m.load_state_dict(st, strict=False)
    return m.to(device).eval(), proc


@torch.no_grad()
def cls_attn(model, pv, F):
    a = model(pixel_values=pv, output_attentions=True).attentions[-1]
    return a[0, :, 0, 1:].mean(0).float().reshape(F, F).cpu().numpy()


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [d for d in json.load(open(args.data_path)) if d.get("bboxes")]
    if args.num_samples > 0:
        data = data[:args.num_samples]
    F = args.feature_size
    if F == -1:
        from transformers import CLIPVisionConfig
        F = (lambda c: c.image_size // c.patch_size)(CLIPVisionConfig.from_pretrained(args.vision_model))
    encoders = [kv.split("=", 1) for kv in args.ckpts.split(",")]

    # compute attention maps per encoder for all samples (cache small FxF arrays)
    attn_cache = {lab: [] for lab, _ in encoders}
    inbox_cache = {lab: [] for lab, _ in encoders}
    for lab, ckpt in encoders:
        print(f"[viz] encoder {lab} ({ckpt}) ...", flush=True)
        model, proc = load_clip(args.vision_model, ckpt, dtype, device)
        for s in data:
            try:
                img = Image.open(os.path.join(args.image_dir, s["image"])).convert("RGB")
            except Exception:
                attn_cache[lab].append(None); inbox_cache[lab].append(float("nan")); continue
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            am = cls_attn(model, pv, F)
            fg = compute_geometric_mask([s["bboxes"]], F, torch.device("cpu"), args.geo_tau)[0].numpy()
            inbox = float((am * fg).sum() / (am.sum() + 1e-8))
            attn_cache[lab].append(am); inbox_cache[lab].append(inbox)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    ncol = 1 + len(encoders)
    for i, s in enumerate(data):
        try:
            img = Image.open(os.path.join(args.image_dir, s["image"])).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        fig, axes = plt.subplots(1, ncol, figsize=(4 * ncol, 4))
        if ncol == 1:
            axes = [axes]
        ax = axes[0]; ax.imshow(img); ax.set_title("input + GT", fontsize=9); ax.axis("off")
        for b in s["bboxes"]:
            ax.add_patch(patches.Rectangle((b[0]*W, b[1]*H), (b[2]-b[0])*W, (b[3]-b[1])*H,
                         fill=False, edgecolor="lime", linewidth=1.5))
        for j, (lab, _) in enumerate(encoders):
            ax = axes[1 + j]
            am = attn_cache[lab][i]
            ax.imshow(img.convert("L"), cmap="gray")
            if am is not None:
                heat = np.array(Image.fromarray((am / (am.max()+1e-8) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR))
                ax.imshow(heat, cmap="jet", alpha=0.5)
                for b in s["bboxes"]:
                    ax.add_patch(patches.Rectangle((b[0]*W, b[1]*H), (b[2]-b[0])*W, (b[3]-b[1])*H,
                                 fill=False, edgecolor="lime", linewidth=1.0))
            ax.set_title(f"{lab}  in-box={inbox_cache[lab][i]:.2f}", fontsize=9); ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{s['id']}.png"), dpi=80, bbox_inches="tight")
        plt.close(fig)
        if (i + 1) % 25 == 0:
            print(f"[viz] {i+1}/{len(data)} saved", flush=True)

    # summary index: mean in-box per encoder
    print(f"\n[viz] saved {len(data)} panels -> {args.out_dir}")
    for lab, _ in encoders:
        vals = [v for v in inbox_cache[lab] if v == v]
        print(f"  {lab}: mean attn_in_box = {sum(vals)/max(len(vals),1):.4f}")


if __name__ == "__main__":
    main()

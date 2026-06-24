"""
GAA Attention Evaluation — Attn-IoU & Background-Attention metrics

Measures the quantity GAA actually optimizes: how much of the CLIP CLS-token
attention falls inside the ground-truth vessel boxes vs. on the background
(waves / sky / glint).  This is the perception metric reviewers asked for and
the one the paper reports as "Attn-IoU"; until now no code computed it.

Why this script is CLIP-only (no LLM):
  GAA's effect lives entirely in the CLIP vision encoder weights.  To measure
  attention alignment we only need the vision tower, so we load CLIPVisionModel
  with eager attention (output_attentions requires eager, not SDPA) and skip the
  8B language model.  This runs in seconds on a single GPU (or CPU).

Two checkpoints can be compared in one run:
  --baseline_ckpt : (optional) vision_encoder.bin from a baseline run
  --gaa_ckpt      : (optional) vision_encoder.bin from a GAA run
If neither is given, the stock pre-trained CLIP is evaluated (sanity baseline).

NOTE: vision_encoder.bin is produced by train_gaa.py's final save block.  If an
older checkpoint lacks it (the original code did not save vision weights), the
stock CLIP is used and a warning is printed — the GAA effect would be invisible.

Metrics (averaged over samples that have >=1 box):
  attn_in_box   : fraction of CLS->patch attention mass inside the GT boxes.
                  Higher is better. (1 - this) == the L_geo target.
  bg_attn_ratio : 1 - attn_in_box. Lower is better.
  attn_iou      : IoU between the binarized attention map (attn/max > thresh)
                  and the GT foreground mask. This is the paper's "Attn-IoU".

Usage:
  python gaa/evaluate_attn.py \
      --data_path data/sds_gaa_en.json \
      --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
      --vision_model openai/clip-vit-large-patch14-336 \
      --baseline_ckpt checkpoints/sds_lora_en/vision_encoder.bin \
      --gaa_ckpt      checkpoints/gaa_sds_en/vision_encoder.bin
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

import torch
from PIL import Image

from gaa.geometric_loss import compute_geometric_mask


def parse_args():
    p = argparse.ArgumentParser("GAA attention-alignment evaluation")
    p.add_argument("--data_path", required=True, help="chat.json with bboxes field")
    p.add_argument("--image_dir", required=True, help="image root directory")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--baseline_ckpt", default=None,
                   help="vision_encoder.bin from baseline run (optional)")
    p.add_argument("--gaa_ckpt", default=None,
                   help="vision_encoder.bin from GAA run (optional)")
    p.add_argument("--geo_tau", type=float, default=0.5,
                   help="foreground coverage threshold for GT mask (matches training)")
    p.add_argument("--attn_thresh", type=float, default=0.5,
                   help="binarization threshold on attn/max for Attn-IoU")
    p.add_argument("--feature_size", type=int, default=-1,
                   help="ViT grid side; -1 = auto from CLIP config (24 for L/14-336)")
    p.add_argument("--num_samples", type=int, default=-1,
                   help="limit number of samples (-1 = all)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def _strip_prefix(state: dict, prefix: str) -> dict:
    """Drop a leading key prefix (e.g. 'model.') if every key carries it."""
    if state and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_clip_eager(vision_model: str, ckpt: str | None, dtype: torch.dtype, device):
    """Load CLIPVisionModel with eager attention; optionally overlay tuned weights.

    The VLM wraps CLIP as ``VisionEncoderWrapper.model``; train_gaa.py saves
    ``model.vision_encoder.state_dict()`` whose keys are prefixed with 'model.'.
    We strip that prefix so the weights load straight into a bare CLIPVisionModel.
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor

    model = CLIPVisionModel.from_pretrained(
        vision_model, torch_dtype=dtype, attn_implementation="eager"
    )
    processor = CLIPImageProcessor.from_pretrained(vision_model)

    if ckpt:
        if not os.path.exists(ckpt):
            print(f"  [WARN] checkpoint not found, using stock CLIP: {ckpt}")
        else:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            state = _strip_prefix(state, "model.")            # wrapper -> bare CLIP
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            print(f"  [load] {os.path.basename(ckpt)}: matched {loaded}/{len(state)} "
                  f"tensors (missing={len(missing)}, unexpected={len(unexpected)})")
            if loaded == 0:
                print("  [WARN] no vision weights matched — evaluating stock CLIP.")

    return model.to(device).eval(), processor


@torch.no_grad()
def cls_attention_map(model, pixel_values, feature_size: int) -> torch.Tensor:
    """Last-layer CLS->patch attention, averaged over heads -> (F, F).

    Mirrors gaa_trainer._get_clip_attentions: attentions[-1] is (B,H,N+1,N+1);
    row 0 is the CLS query, columns 1: are the spatial patches.
    """
    out = model(pixel_values=pixel_values, output_attentions=True)
    attn = out.attentions[-1]                 # (1, H, N+1, N+1)
    cls_attn = attn[0, :, 0, 1:]              # (H, N)
    cls_attn = cls_attn.mean(dim=0)           # (N,) average over heads
    return cls_attn.reshape(feature_size, feature_size).float()


def sample_metrics(cls_attn: torch.Tensor, fg_mask: torch.Tensor,
                   attn_thresh: float) -> dict:
    """Per-sample alignment metrics. fg_mask: (F,F) 1=foreground."""
    total = cls_attn.sum().clamp(min=1e-8)
    attn_in_box = (cls_attn * fg_mask).sum() / total          # higher=better

    # Attn-IoU (area-matched): select the top-K attention patches where K is the
    # GT box area in patches, then IoU vs the GT mask.  Area-matching makes the
    # metric parameter-free and avoids the CLIP "register token" artifact where
    # a fixed threshold puts all mass on a few non-spatial high-norm tokens.
    flat_attn = cls_attn.flatten()
    flat_fg = fg_mask.flatten()
    k = int(flat_fg.sum().item())
    if k > 0:
        topk = torch.topk(flat_attn, k).indices
        pred = torch.zeros_like(flat_attn)
        pred[topk] = 1.0
        inter = (pred * flat_fg).sum()
        union = (pred + flat_fg).clamp(max=1.0).sum().clamp(min=1e-8)
        attn_iou = (inter / union).item()
    else:
        attn_iou = float("nan")

    # Also report the peak-threshold IoU for reference (paper-style binarization).
    norm = cls_attn / cls_attn.max().clamp(min=1e-8)
    pred_t = (norm > attn_thresh).float()
    inter_t = (pred_t * fg_mask).sum()
    union_t = (pred_t + fg_mask).clamp(max=1.0).sum().clamp(min=1e-8)
    attn_iou_thresh = (inter_t / union_t).item()

    return {
        "attn_in_box": attn_in_box.item(),
        "bg_attn_ratio": (1.0 - attn_in_box).item(),
        "attn_iou": attn_iou,
        "attn_iou_thresh": attn_iou_thresh,
    }


def evaluate(model, processor, samples, image_dir, feature_size,
             geo_tau, attn_thresh, device, dtype) -> dict:
    agg = {"attn_in_box": [], "bg_attn_ratio": [], "attn_iou": [], "attn_iou_thresh": []}
    skipped = 0
    for s in samples:
        boxes = s.get("bboxes", [])
        if not boxes:
            skipped += 1
            continue
        img_path = os.path.join(image_dir, s["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        cls_attn = cls_attention_map(model, pixel_values, feature_size).to(device)
        # GT foreground mask via the same coverage rule used in training (Eq.4).
        fg_mask = compute_geometric_mask([boxes], feature_size, device, geo_tau)[0]

        m = sample_metrics(cls_attn, fg_mask, attn_thresh)
        for k in agg:
            agg[k].append(m[k])

    def _nanmean(vals):
        clean = [x for x in vals if not math.isnan(x)]
        return sum(clean) / len(clean) if clean else float("nan")

    mean = {k: _nanmean(v) for k, v in agg.items()}
    mean["n_eval"] = len(agg["attn_in_box"])
    mean["n_skipped"] = skipped
    return mean


def main():
    args = parse_args()
    DTYPE = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data_path) as f:
        data = json.load(f)
    if args.num_samples > 0:
        data = data[:args.num_samples]

    # Resolve feature_size from CLIP config if not given.
    feature_size = args.feature_size
    if feature_size == -1:
        from transformers import CLIPVisionConfig
        cfg = CLIPVisionConfig.from_pretrained(args.vision_model)
        feature_size = cfg.image_size // cfg.patch_size
        print(f"[eval] auto feature_size = {feature_size}")

    runs = [("stock-CLIP", None)]
    if args.baseline_ckpt:
        runs.append(("baseline", args.baseline_ckpt))
    if args.gaa_ckpt:
        runs.append(("GAA", args.gaa_ckpt))

    results = {}
    for label, ckpt in runs:
        print(f"\nLoading vision tower: {label}")
        model, processor = load_clip_eager(args.vision_model, ckpt, DTYPE, device)
        results[label] = evaluate(
            model, processor, data, args.image_dir, feature_size,
            args.geo_tau, args.attn_thresh, device, DTYPE,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Attention alignment on {os.path.basename(args.data_path)} "
          f"(tau={args.geo_tau}, attn_thresh={args.attn_thresh})")
    print(f"{'='*72}")
    header = (f"{'model':<14}{'Attn-IoU↑':>12}{'IoU@thr↑':>10}"
              f"{'attn_in_box↑':>14}{'bg_ratio↓':>12}{'n':>6}")
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        print(f"{label:<14}{m['attn_iou']:>12.4f}{m['attn_iou_thresh']:>10.4f}"
              f"{m['attn_in_box']:>14.4f}{m['bg_attn_ratio']:>12.4f}{m['n_eval']:>6d}")
    print("\nAttn-IoU = area-matched (top-K, K=box area). IoU@thr = peak>thresh "
          "(paper-style, register-token sensitive).")
    if results and next(iter(results.values()))["n_skipped"]:
        print(f"\n(skipped {next(iter(results.values()))['n_skipped']} samples "
              f"with no box / unreadable image)")


if __name__ == "__main__":
    main()

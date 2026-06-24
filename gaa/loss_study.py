"""
GAA loss-redesign study (vision-encoder-only, no LLM).

Goal: rank geometric attention loss variants by whether they actually steer the
CLIP CLS attention onto vessel geometry, isolated from the SFT/LLM confound.

For each variant we:
  1. start from the stock pre-trained CLIP (eager attention),
  2. optimize ONLY the vision encoder on the train split with that geo loss,
  3. evaluate Attn-IoU / attn_in_box / bg_ratio on a held-out test split.

A "stock" row (no training) is reported as the reference.  This directly tests
the 2026-06-19 finding that the original bg_l2 loss does not improve alignment,
and which redesign (inbox_mass / inbox_ce / fg_bg_margin) does.

Single GPU, CLIP-L only — fast (no 8B model).

Usage:
  python gaa/loss_study.py \
      --data_path data/sds_gaa_en.json \
      --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
      --variants bg_l2,inbox_mass,inbox_ce,fg_bg_margin \
      --steps 300 --lr 1e-5 --batch_size 8
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch
from PIL import Image

from gaa.geometric_loss import compute_geometric_mask
from gaa.geo_losses import geo_loss, extract_cls_attn, VARIANTS


def parse_args():
    p = argparse.ArgumentParser("GAA loss-redesign study (vision-only)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--test_data_path", default=None,
                   help="if set, train on ALL of data_path and evaluate on this "
                        "separate json (cross-domain transfer test); disables the "
                        "internal train/test split")
    p.add_argument("--test_image_dir", default=None,
                   help="image dir for --test_data_path (default: --image_dir)")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help="comma-separated subset of " + ",".join(VARIANTS))
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--shuffle_seed", type=int, default=-1,
                   help="if >=0, shuffle records with this seed before splitting "
                        "(tests split-ordering robustness; -1 = positional split)")
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.0, help="for fg_bg_margin")
    p.add_argument("--feature_size", type=int, default=-1)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", default=None,
                   help="if set, save each trained CLIP vision encoder as "
                        "<save_dir>/<variant>_vision_encoder.bin (bare CLIPVisionModel "
                        "state_dict, loadable by evaluate_attn.py and the VLM wrapper)")
    return p.parse_args()


def load_split(data_path, train_frac, shuffle_seed=-1):
    with open(data_path) as f:
        data = [d for d in json.load(f) if d.get("bboxes")]
    if shuffle_seed >= 0:
        import random
        random.Random(shuffle_seed).shuffle(data)
    n_train = int(len(data) * train_frac)
    return data[:n_train], data[n_train:]


def preprocess(samples, processor, image_dir, device, dtype):
    """Cache (pixel_values, bboxes) for a split. pixel_values stay on CPU."""
    px, boxes = [], []
    for s in samples:
        try:
            img = Image.open(os.path.join(image_dir, s["image"])).convert("RGB")
        except Exception:
            continue
        pv = processor(images=img, return_tensors="pt").pixel_values[0]  # (3,H,W)
        px.append(pv)
        boxes.append(s["bboxes"])
    return px, boxes


@torch.no_grad()
def evaluate(model, px, boxes, feature_size, geo_tau, device, dtype):
    model.eval()
    iou_a, in_box, bg = [], [], []
    for pv, bx in zip(px, boxes):
        pixel_values = pv.unsqueeze(0).to(device=device, dtype=dtype)
        attn = model(pixel_values=pixel_values, output_attentions=True).attentions[-1]
        cls = extract_cls_attn(attn, feature_size)[0].mean(0).float()   # (F,F)
        fg = compute_geometric_mask([bx], feature_size, device, geo_tau)[0].float()

        total = cls.sum().clamp(min=1e-8)
        frac = (cls * fg).sum() / total
        in_box.append(frac.item()); bg.append((1 - frac).item())

        k = int(fg.sum().item())
        if k > 0:
            idx = torch.topk(cls.flatten(), k).indices
            pred = torch.zeros_like(cls.flatten()); pred[idx] = 1.0
            inter = (pred * fg.flatten()).sum()
            union = (pred + fg.flatten()).clamp(max=1.0).sum().clamp(min=1e-8)
            iou_a.append((inter / union).item())
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return {"attn_iou": m(iou_a), "attn_in_box": m(in_box), "bg_ratio": m(bg)}


def train_variant(variant, vision_model, train_px, train_boxes,
                  feature_size, args, device, dtype):
    from transformers import CLIPVisionModel
    model = CLIPVisionModel.from_pretrained(
        vision_model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    n = len(train_px)
    for step in range(args.steps):
        idx = torch.randint(0, n, (min(args.batch_size, n),), generator=g).tolist()
        pixel_values = torch.stack([train_px[i] for i in idx]).to(device=device, dtype=dtype)
        bxs = [train_boxes[i] for i in idx]

        attn = model(pixel_values=pixel_values, output_attentions=True).attentions[-1]
        cls = extract_cls_attn(attn, feature_size)                      # (B,H,F,F)
        fg = compute_geometric_mask(bxs, feature_size, device, args.geo_tau)
        loss = geo_loss(variant, cls.float(), fg.float(), margin=args.margin)

        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  [{variant}] step {step:4d}/{args.steps}  loss={loss.item():.5f}", flush=True)
    return model


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    from transformers import CLIPImageProcessor, CLIPVisionConfig
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    feature_size = args.feature_size
    if feature_size == -1:
        cfg = CLIPVisionConfig.from_pretrained(args.vision_model)
        feature_size = cfg.image_size // cfg.patch_size

    if args.test_data_path:
        # Cross-domain: train on ALL of data_path, evaluate on a separate set.
        train = [d for d in json.load(open(args.data_path)) if d.get("bboxes")]
        test = [d for d in json.load(open(args.test_data_path)) if d.get("bboxes")]
        test_image_dir = args.test_image_dir or args.image_dir
    else:
        train, test = load_split(args.data_path, args.train_frac, args.shuffle_seed)
        test_image_dir = args.image_dir
    print(f"[study] train={len(train)} test={len(test)} feature_size={feature_size} "
          f"variants={variants} steps={args.steps} lr={args.lr} bs={args.batch_size} "
          f"cross_domain={bool(args.test_data_path)}")
    train_px, train_boxes = preprocess(train, processor, args.image_dir, device, dtype)
    test_px, test_boxes = preprocess(test, processor, test_image_dir, device, dtype)

    results = {}
    # Reference: stock CLIP, no training.
    from transformers import CLIPVisionModel
    stock = CLIPVisionModel.from_pretrained(
        args.vision_model, torch_dtype=dtype, attn_implementation="eager").to(device)
    results["stock"] = evaluate(stock, test_px, test_boxes, feature_size, args.geo_tau, device, dtype)
    del stock
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for v in variants:
        print(f"\n=== training variant: {v} ===")
        model = train_variant(v, args.vision_model, train_px, train_boxes,
                              feature_size, args, device, dtype)
        results[v] = evaluate(model, test_px, test_boxes, feature_size, args.geo_tau, device, dtype)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            sp = os.path.join(args.save_dir, f"{v}_vision_encoder.bin")
            torch.save(model.state_dict(), sp)
            print(f"  [save] {v} -> {sp}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*64}")
    print(f"  Loss-variant study on held-out test (n={len(test_px)}), "
          f"steps={args.steps}, lr={args.lr}")
    print(f"{'='*64}")
    hdr = f"{'variant':<14}{'Attn-IoU↑':>12}{'attn_in_box↑':>14}{'bg_ratio↓':>12}"
    print(hdr); print("-" * len(hdr))
    for k, m in results.items():
        print(f"{k:<14}{m['attn_iou']:>12.4f}{m['attn_in_box']:>14.4f}{m['bg_ratio']:>12.4f}")


if __name__ == "__main__":
    main()

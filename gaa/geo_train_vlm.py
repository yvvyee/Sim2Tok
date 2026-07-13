"""
geo_train_vlm.py — Vision-encoder-only GAA geo training for any VLM

각 VLM의 비전 인코더만 추출하여 inbox_ce geo loss로 학습시킨 뒤
SMD 실제 데이터에서 Attn-IoU를 평가한다.

Architecture별 spatial attention 추출:
  CLS 계열 (clip, internvl, mllama): CLS→patch attention row
  non-CLS 계열 (siglip, qwen)     : column-sum of attention matrix
                                     (각 패치가 받는 총 어텐션 = 공간 현저성)

Usage (단일 모델):
  python gaa/geo_train_vlm.py --model smolvlm \\
      --data_path /home/ywlee/dev/gaa/data/sds_geopool.json \\
      --output_dir checkpoints/geo_smolvlm \\
      --num_epochs 5

Usage (11개 전부, 순차):
  bash gaa/run_geo_train_all_vlms.sh
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import gc

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from gaa.geometric_loss import compute_geometric_mask
from gaa.geo_losses import geo_loss
from gaa.geo_validate_vlms import (
    REGISTRY, get_vision_tower, get_processor,
    preprocess_image, extract_attn_map, compute_attn_iou,
    _reload_with_eager,
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GeoDataset(Dataset):
    def __init__(self, data_path, proc, family, img_size):
        self.samples = [s for s in json.load(open(data_path))
                        if s.get("bboxes")]
        self.proc = proc
        self.family = family
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["image"]  # absolute path in sds_geopool.json
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size))
        return image, s["bboxes"]


# ── Spatial attention extraction (grad-enabled) ───────────────────────────────

def extract_spatial_attn_train(encoder, pv, family, feature_size, ekw):
    """Return (1, H, F, F) spatial attention suitable for geo_loss.

    For CLS models: CLS→patch attention row.
    For non-CLS models: column-sum of attention matrix (attention received).
    """
    try:
        if family in ("clip", "internvl"):
            out = encoder(pixel_values=pv, output_attentions=True)
            attn = out.attentions[-1]            # (1, H, N+1, N+1)
            cls_row = attn[:, :, 0:1, 1:]        # (1, H, 1, N)
            B, H, _, N = cls_row.shape
            return cls_row.reshape(B, H, feature_size, feature_size)

        elif family == "mllama":
            # 1-tile mode: preprocess returns (1, 1, C, H, W) → 6D (1, 1, 1, C, H, W)
            ekw = ekw or {}
            pv_in = pv
            if pv_in.dim() == 5:
                pv_in = pv_in.unsqueeze(1)
            elif pv_in.dim() == 4:
                pv_in = pv_in.unsqueeze(0).unsqueeze(0)
            ar_ids = ekw.get("aspect_ratio_ids",
                             torch.ones(1, 1, dtype=torch.long, device=pv.device))
            ar_mask = ekw.get("aspect_ratio_mask",
                              torch.ones(1, 1, 1, dtype=torch.long, device=pv.device))
            N = feature_size * feature_size
            # Feature norm proxy (no output_attentions to avoid OOM)
            out = encoder(pixel_values=pv_in,
                          aspect_ratio_ids=ar_ids,
                          aspect_ratio_mask=ar_mask)
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                hs = out[0]
            hs_tile = hs[:, :N]
            return _feature_norm_attn(hs_tile, feature_size)

        elif family == "siglip":
            pv_in = pv
            if pv_in.dim() == 5:
                pv_in = pv_in[0, 0:1]
            # Try with attention first
            try:
                out = encoder(pixel_values=pv_in, output_attentions=True)
                attn_list = getattr(out, "attentions", None)
                if attn_list and len(attn_list) > 0:
                    attn = attn_list[-1]
                    if isinstance(attn, (list, tuple)):
                        attn = attn[0]
                    if attn.dim() == 3:
                        attn = attn.unsqueeze(0)   # (1, H, N, N)
                    B, H, N, _ = attn.shape
                    col_sum = attn.sum(dim=-2)     # (1, H, N)
                    return col_sum.reshape(B, H, feature_size, feature_size)
            except Exception:
                pass
            # OOM or no attention → feature norm fallback (no attention matrix needed)
            out = encoder(pixel_values=pv_in)
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                hs = out[0] if hasattr(out, "__iter__") else out
            return _feature_norm_attn(hs, feature_size)

        elif family == "qwen":
            thw = ekw.get("image_grid_thw", None)
            try:
                if thw is not None:
                    hs = encoder(pv, grid_thw=thw)
                else:
                    hs = encoder(pv)
                if isinstance(hs, torch.Tensor):
                    return _feature_norm_attn(hs.unsqueeze(0), feature_size)
                return _feature_norm_attn(hs.last_hidden_state, feature_size)
            except Exception:
                return None

        else:
            return None

    except Exception as e:
        print(f"    [WARN] attn extraction failed in train: {e}")
        return None


def _feature_norm_attn(hs, feature_size):
    """Feature L2 norm as spatial saliency proxy. Returns (1, 1, F, F)."""
    # hs: (1, N, D) or (N, D)
    if hs.dim() == 2:
        hs = hs.unsqueeze(0)
    N = hs.shape[1]
    if N == feature_size * feature_size + 1:
        hs = hs[:, 1:]   # skip CLS
        N = N - 1
    if N == feature_size * feature_size:
        norm = hs.norm(dim=-1)   # (1, N)
        return norm.reshape(1, 1, feature_size, feature_size)
    # Interpolate
    src_f = int(math.isqrt(N))
    norm = hs.norm(dim=-1).reshape(1, 1, src_f, src_f)
    return F.interpolate(norm, size=(feature_size, feature_size),
                         mode="bilinear", align_corners=False)


# ── Training ─────────────────────────────────────────────────────────────────

def train_geo(encoder, proc, family, img_size, feature_size,
              data_path, num_epochs, lr, geo_tau, device, dtype, output_dir,
              max_steps=-1):
    dataset = GeoDataset(data_path, proc, family, img_size)
    print(f"  학습 샘플: {len(dataset)}")

    opt = torch.optim.AdamW(
        [p for p in encoder.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_epochs * len(dataset))

    encoder.train()
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_valid = 0
        for i, (image, boxes) in enumerate(dataset):
            if max_steps > 0 and global_step >= max_steps:
                break
            try:
                pv, fs_actual, ekw = preprocess_image(
                    proc, image, family, img_size, device, dtype)
                use_fs = fs_actual if fs_actual > 0 else feature_size

                spatial_attn = extract_spatial_attn_train(
                    encoder, pv, family, use_fs, ekw)
                if spatial_attn is None:
                    continue

                fg_mask = compute_geometric_mask(
                    [boxes], use_fs, device, geo_tau)    # (1, F, F)

                if fg_mask.sum() == 0:
                    continue

                loss = geo_loss("inbox_ce", spatial_attn.float(),
                                fg_mask.float())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step()
                scheduler.step()

                total_loss += loss.item()
                n_valid += 1
                global_step += 1

            except Exception as e:
                continue

        avg = total_loss / max(n_valid, 1)
        print(f"  [epoch {epoch+1}/{num_epochs}]  loss={avg:.4f}  n={n_valid}")
        if max_steps > 0 and global_step >= max_steps:
            break

    encoder.eval()
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "vision_encoder.bin")
    torch.save(encoder.state_dict(), ckpt_path)
    print(f"  저장: {ckpt_path}")
    return ckpt_path


# ── Evaluation (Attn-IoU on SMD) ─────────────────────────────────────────────

@torch.no_grad()
def eval_attn_iou(encoder, proc, family, img_size, feature_size,
                  eval_data_path, eval_imgdir, geo_tau, device, dtype):
    samples = json.load(open(eval_data_path))
    encoder.eval()
    agg = []
    skipped = 0
    for s in samples:
        boxes = s.get("bboxes", [])
        if not boxes:
            skipped += 1
            continue
        img_path = os.path.join(eval_imgdir, s["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        try:
            pv, fs_actual, ekw = preprocess_image(
                proc, image, family, img_size, device, dtype)
            use_fs = fs_actual if fs_actual > 0 else feature_size
            smap = extract_attn_map(encoder, pv, family, use_fs, extra_kwargs=ekw)
            fg_mask = compute_geometric_mask([boxes], use_fs, device, geo_tau)[0]
            m = compute_attn_iou(smap.to(device), fg_mask)
            agg.append(m["attn_iou"])
        except Exception:
            skipped += 1

    def nanmean(v):
        c = [x for x in v if not math.isnan(x)]
        return sum(c) / len(c) if c else float("nan")

    result = nanmean(agg)
    print(f"  Attn-IoU = {result:.4f}  (n={len(agg)}, skip={skipped})")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("GAA geo training for arbitrary VLM vision encoder")
    p.add_argument("--model",      required=True,
                   choices=list(REGISTRY.keys()), help="VLM name from vlm_zoo REGISTRY")
    p.add_argument("--data_path",  default="/home/ywlee/dev/gaa/data/sds_geopool.json")
    p.add_argument("--eval_data",  default="/home/ywlee/dev/gaa/data/smd_onshore_big.json")
    p.add_argument("--eval_imgdir",default="/home/ywlee/SSD/smd_onshore_big")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--max_steps",  type=int, default=-1,
                   help="최대 학습 스텝 수 (-1=무제한). smoke test용.")
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--geo_tau",    type=float, default=0.5)
    p.add_argument("--dtype",      default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo, family, img_size, patch_size, notes = REGISTRY[args.model]
    feature_size = img_size // patch_size
    output_dir = args.output_dir or f"checkpoints/geo_{args.model}"

    print(f"\n{'='*60}")
    print(f"  GAA Geo Training: [{args.model}]  {notes}")
    print(f"  repo={repo}")
    print(f"  feature_size={feature_size}  epochs={args.num_epochs}  lr={args.lr}")
    print(f"{'='*60}")

    from transformers import AutoModelForImageTextToText

    # ── CLIP: load directly ───────────────────────────────────────────────────
    if family == "clip":
        from transformers import CLIPVisionModel, CLIPImageProcessor
        _CLIP_BASE = "openai/clip-vit-large-patch14-336"
        print(f"  Loading CLIP (eager, no LLM) ...")
        enc = CLIPVisionModel.from_pretrained(
            _CLIP_BASE, torch_dtype=dtype, attn_implementation="eager"
        ).to(device)
        proc = CLIPImageProcessor.from_pretrained(_CLIP_BASE)

    # ── All others: load full model GPU bf16 → extract enc → delete LLM ──────
    else:
        print(f"  Loading full model (GPU bf16) ...")
        model = None
        for local_only in [False, True]:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    repo, torch_dtype=torch.bfloat16, device_map="cuda",
                    low_cpu_mem_usage=True, local_files_only=local_only)
                break
            except Exception as e:
                if local_only:
                    print(f"  [ERROR] load failed: {e}")
                    return
        if model is None:
            return

        model.eval()
        enc_raw = get_vision_tower(model, family)
        enc_state = {k: v.cpu() for k, v in enc_raw.state_dict().items()}
        enc_cfg = getattr(enc_raw, "config", None)
        enc_cls = type(enc_raw)
        del model; gc.collect(); torch.cuda.empty_cache()

        # Rebuild encoder.
        # For siglip with large feature_size (F>36 → N>1296 patches), eager attention
        # materialises the full N×N matrix even without output_attentions and causes OOM.
        # Use default SDPA for those; feature-norm proxy is used instead of attention.
        if enc_cfg is not None:
            use_eager = not (family == "siglip" and feature_size > 36)
            if use_eager:
                try:
                    enc_cfg._attn_implementation = "eager"
                except Exception:
                    pass
            enc = enc_cls(enc_cfg)
            enc.load_state_dict(enc_state, strict=False)
        del enc_state; gc.collect()

        enc = enc.to(device=device, dtype=dtype)

        proc = get_processor(repo, family)

    # ── Baseline Attn-IoU (before training) ──────────────────────────────────
    print(f"\n  [Before] Attn-IoU on SMD (288 real samples):")
    enc.eval()
    before = eval_attn_iou(enc, proc, family, img_size, feature_size,
                           args.eval_data, args.eval_imgdir,
                           args.geo_tau, device, dtype)

    # ── Train geo loss ────────────────────────────────────────────────────────
    print(f"\n  [Train] inbox_ce geo loss on {args.data_path}")
    ckpt_path = train_geo(enc, proc, family, img_size, feature_size,
                          args.data_path, args.num_epochs, args.lr,
                          args.geo_tau, device, dtype, output_dir,
                          max_steps=args.max_steps)

    # ── After Attn-IoU (after training) ──────────────────────────────────────
    print(f"\n  [After] Attn-IoU on SMD (288 real samples):")
    after = eval_attn_iou(enc, proc, family, img_size, feature_size,
                          args.eval_data, args.eval_imgdir,
                          args.geo_tau, device, dtype)

    delta = after - before
    print(f"\n  결과 요약: {args.model}")
    print(f"    Before: {before:.4f}")
    print(f"    After:  {after:.4f}")
    print(f"    Delta:  {delta:+.4f}")
    print(f"    GAA 참조 (CLIP inbox_ce): 0.4373")

    # ── Save result JSON ──────────────────────────────────────────────────────
    result = {
        "model": args.model, "family": family, "notes": notes,
        "attn_iou_before": before, "attn_iou_after": after, "delta": delta,
        "ckpt": ckpt_path,
    }
    result_path = os.path.join(output_dir, "geo_result.json")
    json.dump(result, open(result_path, "w"), indent=2)
    print(f"  저장: {result_path}")


if __name__ == "__main__":
    main()

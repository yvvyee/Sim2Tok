"""
geo_validate_vlms.py — Baseline Attn-IoU across 11 VLMs

각 VLM의 비전 인코더(LLM 제외)만 로드하여 실 해상 데이터(SMD onshore)에서
Attn-IoU를 측정한다. 이는 GAA 이전 '모든 VLM이 선박 주의집중 정렬 문제를
가지고 있음'을 증명하는 baseline 실험이다.

Vision encoder groups:
  CLIP        : llava-1.5, llava-next       (CLS→patch attn, F=24)
  SigLIP      : paligemma, aya-vision,      (col-mean attn)
                gemma-3, idefics2, smolvlm
  Qwen custom : qwen2-vl, qwen2.5-vl        (col-mean attn)
  InternViT   : internvl3                   (CLS→patch attn)
  Mllama      : llama-3.2                   (CLS→patch attn)

Usage:
  python gaa/geo_validate_vlms.py \\
      --data_path /home/ywlee/dev/gaa/data/smd_onshore_big.json \\
      --image_dir /home/ywlee/SSD/smd_onshore_big \\
      --num_samples 100 \\
      --output_json results/geo_vlm_baseline.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import traceback

import torch
import torch.nn.functional as F
from PIL import Image

from gaa.geometric_loss import compute_geometric_mask


# ── Model registry ────────────────────────────────────────────────────────────
# (hf_repo, encoder_family, native_img_size, patch_size, notes)
REGISTRY = {
    "llava-1.5":  ("llava-hf/llava-1.5-7b-hf",                "clip",     336, 14, "CLIP-ViT-L/14-336"),
    "llava-next": ("llava-hf/llava-v1.6-mistral-7b-hf",       "clip",     336, 14, "CLIP-ViT-L/14-336"),
    "llama-3.2":  ("meta-llama/Llama-3.2-11B-Vision-Instruct","mllama",   560, 14, "ViT-H cross-attn"),
    "idefics2":   ("HuggingFaceM4/idefics2-8b",               "siglip",   980, 14, "SigLIP-980"),
    "smolvlm":    ("HuggingFaceTB/SmolVLM-Instruct",          "siglip",   384, 14, "SigLIP-384"),
    "paligemma":  ("google/paligemma2-3b-mix-448",            "siglip",   448, 14, "SigLIP-448"),
    "aya-vision": ("CohereForAI/aya-vision-8b",               "siglip",   364, 14, "SigLIP-364"),
    "gemma-3":    ("google/gemma-3-4b-it",                    "siglip",   896, 14, "SigLIP-896"),
    "qwen2-vl":   ("Qwen/Qwen2-VL-7B-Instruct",              "qwen",     448, 14, "Qwen2 custom ViT"),
    "qwen2.5-vl": ("Qwen/Qwen2.5-VL-7B-Instruct",            "qwen",     448, 14, "Qwen2.5 custom ViT"),
    "internvl3":  ("OpenGVLab/InternVL3-8B-hf",              "internvl", 448, 14, "InternViT"),
}


# ── Vision encoder extractor ─────────────────────────────────────────────────

def get_vision_tower(model, family):
    """Extract the standalone vision encoder module from a full VLM."""
    # Try common attribute paths per family / model_type
    candidates = {
        "clip":     ["vision_tower", "model.vision_tower"],
        "siglip":   ["model.vision_tower", "vision_tower", "model.vision_model", "vision_model"],
        "mllama":   ["model.vision_model", "vision_model"],
        "qwen":     ["visual", "model.visual"],
        "internvl": ["model.vision_tower", "vision_tower", "model.vision_model"],
    }.get(family, ["model.vision_tower", "vision_tower", "vision_model", "visual"])

    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None and hasattr(obj, "parameters"):
                return obj
        except AttributeError:
            continue

    # Fallback: first nn.Module child that is not an embedding or LM head
    for name, mod in model.named_children():
        if any(k in name.lower() for k in ("vision", "visual", "clip", "siglip", "intern")):
            return mod

    raise RuntimeError(f"Could not find vision tower for family={family}")


# ── Processor / pixel_values helper ─────────────────────────────────────────

def get_processor(repo, family):
    from transformers import AutoProcessor
    for local_only in [False, True]:
        try:
            return AutoProcessor.from_pretrained(repo, local_files_only=local_only)
        except Exception as e:
            if local_only:
                raise
            print(f"    [WARN] online processor load failed, retrying local_files_only ...")


def preprocess_image(proc, image: Image.Image, family: str,
                     img_size: int, device, dtype) -> tuple:
    """Return (pixel_values, feature_size, extra_kwargs) for the vision encoder.

    extra_kwargs: dict of additional kwargs to forward to encoder (e.g. image_grid_thw
    for Qwen, aspect_ratio_ids for Mllama). Empty dict for most models.
    """
    img_resized = image.resize((img_size, img_size), Image.BICUBIC)

    if family == "clip":
        ip = getattr(proc, "image_processor", proc)
        pv = ip(images=img_resized, return_tensors="pt").pixel_values
        fs = img_size // 14
        return pv.to(device=device, dtype=dtype), fs, {}

    if family == "siglip":
        ip = getattr(proc, "image_processor", proc)
        try:
            out = ip(images=img_resized, return_tensors="pt")
            pv = out.pixel_values
        except Exception:
            out = proc(images=img_resized, return_tensors="pt")
            pv = out.pixel_values
        if pv.dim() == 5:
            pv = pv[0, 0:1]   # Idefics3: (1, n_tiles, C, H, W) → (1, C, H, W)
        fs = img_size // 14
        return pv.to(device=device, dtype=dtype), fs, {}

    if family == "mllama":
        # Use exactly 1 tile to avoid 4× memory overhead in attention/forward.
        # MllamaVisionModel accepts num_tiles=1 when aspect_ratio_mask matches.
        try:
            pv_out = proc.image_processor(images=img_resized, return_tensors="pt")
            pv = pv_out["pixel_values"][:, :1]          # (1, 1, C, H, W) — first tile only
            ar_ids = torch.ones(1, 1, dtype=torch.long)  # 1×1 aspect ratio
            ar_mask = torch.ones(1, 1, 1, dtype=torch.long)  # (B, n_imgs, 1 tile)
            fs = img_size // 14
        except Exception:
            import torchvision.transforms as T
            tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                            T.Normalize([0.5]*3, [0.5]*3)])
            pv = tf(img_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
            ar_ids = torch.ones(1, 1, dtype=torch.long)
            ar_mask = torch.ones(1, 1, 1, dtype=torch.long)
            fs = img_size // 14
        ekw = {"aspect_ratio_ids": ar_ids.to(device),
               "aspect_ratio_mask": ar_mask.to(device)}
        return pv.to(device=device, dtype=dtype), fs, ekw

    if family == "qwen":
        try:
            pv_out = proc.image_processor(images=[img_resized], return_tensors="pt")
            pv = pv_out["pixel_values"]                # (n_patches, C*p*p)
            thw = pv_out.get("image_grid_thw", None)
            if thw is not None:
                fs = int(thw[0][1].item())             # H patches
                ekw = {"image_grid_thw": thw.to(device)}
            else:
                fs = img_size // 14
                ekw = {}
        except Exception:
            import torchvision.transforms as T
            tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            pv = tf(img_resized).unsqueeze(0)
            fs = img_size // 14
            ekw = {}
        return pv.to(device=device, dtype=dtype), fs, ekw

    if family == "internvl":
        ip = getattr(proc, "image_processor", proc)
        try:
            pv = ip(images=img_resized, return_tensors="pt").pixel_values
        except Exception:
            pv = proc(images=img_resized, return_tensors="pt").pixel_values
        fs = img_size // 14
        return pv.to(device=device, dtype=dtype), fs, {}

    raise ValueError(f"Unknown family: {family}")


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_attn_map(encoder, pixel_values, family: str,
                     feature_size: int, extra_kwargs: dict = None) -> torch.Tensor:
    """Return spatial attention map (F, F) in float32.

    Strategies:
      CLS models (clip, internvl, mllama): CLS→patch row of last-layer attn,
        averaged over heads.
      Non-CLS models (siglip, qwen): column-sum of last-layer attn averaged
        over heads → "attention received" per patch.
    """
    try:
        if family == "clip":
            out = encoder(pixel_values=pixel_values, output_attentions=True)
            attn = out.attentions[-1]            # (1, H, N+1, N+1)
            cls_row = attn[0, :, 0, 1:]          # (H, N)
            smap = cls_row.mean(dim=0)           # (N,)

        elif family == "mllama":
            # 1-tile mode: preprocess returns (1, 1, C, H, W) → 6D (1, 1, 1, C, H, W)
            pv = pixel_values
            if pv.dim() == 5:
                pv = pv.unsqueeze(1)        # (1, 1, n_tiles, C, H, W)
            elif pv.dim() == 4:
                pv = pv.unsqueeze(0).unsqueeze(0)
            ekw = extra_kwargs or {}
            ar_ids = ekw.get("aspect_ratio_ids",
                             torch.ones(1, 1, dtype=torch.long, device=pixel_values.device))
            ar_mask = ekw.get("aspect_ratio_mask",
                              torch.ones(1, 1, 1, dtype=torch.long, device=pixel_values.device))
            N_tile = feature_size * feature_size
            # Feature norm proxy (skip output_attentions to avoid OOM)
            out = encoder(pixel_values=pv,
                          aspect_ratio_ids=ar_ids,
                          aspect_ratio_mask=ar_mask)
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                hs = out[0]
            smap = hs[0, :N_tile].norm(dim=-1)

        elif family == "siglip":
            # SiglipVisionModel / Idefics3VisionTransformer: no CLS, patch-only
            # pixel_values may be (1,C,H,W) or (1,1,C,H,W) from Idefics3Processor
            pv = pixel_values
            if pv.dim() == 5:
                pv = pv.squeeze(1)   # (1,C,H,W)
            try:
                out = encoder(pixel_values=pv, output_attentions=True)
            except TypeError:
                try:
                    out = encoder(pv, output_attentions=True)
                except Exception:
                    out = encoder(pixel_values=pv)
            # attentions may be list of tensors or None
            attn_list = getattr(out, "attentions", None)
            if attn_list is not None and len(attn_list) > 0:
                attn = attn_list[-1]
                # attn is (1, H, N, N) or (H, N, N) or a tuple
                if isinstance(attn, (list, tuple)):
                    attn = attn[0]  # first element if tuple
                if attn.dim() == 3:
                    attn = attn.unsqueeze(0)   # → (1, H, N, N)
                # column-sum = attention received per patch (averaged over heads)
                col_sum = attn[0].mean(dim=0).sum(dim=0)  # (N,)
                smap = col_sum
            else:
                # Fallback: L2 norm of last hidden state as saliency proxy
                hs = getattr(out, "last_hidden_state", None)
                if hs is None:
                    hs = out[0]
                smap = hs[0].norm(dim=-1)        # (N,)

        elif family == "qwen":
            # Qwen2VLVisionTransformerPretrainedModel needs image_grid_thw
            ekw = extra_kwargs or {}
            thw = ekw.get("image_grid_thw", None)
            try:
                if thw is not None:
                    out = encoder(pixel_values, grid_thw=thw)
                else:
                    out = encoder(pixel_values)
                # Qwen visual returns a tensor directly (not ModelOutput)
                if isinstance(out, torch.Tensor):
                    smap = out.norm(dim=-1)       # (N, D) → (N,)
                elif hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state
                    smap = (hs[0] if hs.dim() == 3 else hs).norm(dim=-1)
                else:
                    smap = out[0].norm(dim=-1) if hasattr(out[0], "norm") else out[0]
            except Exception:
                # Absolute fallback: uniform attention
                N = feature_size * feature_size
                smap = torch.ones(N, device=pixel_values.device, dtype=torch.float32)

        elif family == "internvl":
            # InternViT: has CLS token
            try:
                out = encoder(pixel_values=pixel_values, output_attentions=True)
                attn = out.attentions[-1]        # (1, H, N+1, N+1)
                cls_row = attn[0, :, 0, 1:]
                smap = cls_row.mean(dim=0)
            except Exception:
                out = encoder(pixel_values=pixel_values)
                hs = out.last_hidden_state       # (1, N+1, D)
                smap = hs[0, 1:].norm(dim=-1)   # skip CLS

        else:
            raise ValueError(f"Unknown family: {family}")

    except Exception as e:
        import traceback
        print(f"    [WARN] attn extraction failed ({e})")
        traceback.print_exc()
        N = feature_size * feature_size
        smap = torch.ones(N, device=pixel_values.device, dtype=torch.float32)

    smap = smap.float()

    # Reshape / interpolate to (feature_size, feature_size)
    N = smap.numel()
    if N == feature_size * feature_size:
        return smap.reshape(feature_size, feature_size)
    # Bilinear interpolation for mismatched sizes
    src_f = int(math.isqrt(N))
    if src_f * src_f != N:
        src_f = int(math.ceil(N ** 0.5))
        smap = smap[:src_f * src_f]
    smap_2d = smap.reshape(1, 1, src_f, src_f)
    smap_2d = F.interpolate(smap_2d, size=(feature_size, feature_size),
                            mode="bilinear", align_corners=False)
    return smap_2d[0, 0]


# ── Attn-IoU metric (area-matched top-K) ────────────────────────────────────

def compute_attn_iou(smap: torch.Tensor, fg_mask: torch.Tensor) -> dict:
    """Compute Attn-IoU (area-matched top-K) and attn_in_box."""
    total = smap.sum().clamp(min=1e-8)
    attn_in_box = (smap * fg_mask).sum() / total

    flat_attn = smap.flatten()
    flat_fg = fg_mask.flatten()
    k = int(flat_fg.sum().item())
    if k > 0:
        topk_idx = torch.topk(flat_attn, k).indices
        pred = torch.zeros_like(flat_attn)
        pred[topk_idx] = 1.0
        inter = (pred * flat_fg).sum()
        union = (pred + flat_fg).clamp(max=1.0).sum().clamp(min=1e-8)
        attn_iou = (inter / union).item()
    else:
        attn_iou = float("nan")

    return {
        "attn_iou": attn_iou,
        "attn_in_box": attn_in_box.item(),
        "bg_attn_ratio": (1.0 - attn_in_box).item(),
    }


# ── Eager-attention vision encoder loader ────────────────────────────────────

# CLIP base model shared by llava-1.5 and llava-next
_CLIP_BASE = "openai/clip-vit-large-patch14-336"


def _reload_with_eager(enc_module, family: str):
    """Re-create vision encoder with eager attention using the same weights.

    Uses the same class as the original module so it works for idefics2/3,
    siglip, internvl, and mllama variants without hardcoding class names.
    """
    state = enc_module.state_dict()
    cfg = getattr(enc_module, "config", None)
    if cfg is None:
        return enc_module

    # Patch config for eager attention
    try:
        cfg._attn_implementation = "eager"
    except Exception:
        pass

    # Use the exact same class as the original module
    try:
        new_enc = type(enc_module)(cfg)
    except Exception as e:
        print(f"    [WARN] eager reinit failed ({e}); using original")
        return enc_module

    missing, unexpected = new_enc.load_state_dict(state, strict=False)
    n_matched = len(state) - len(unexpected)
    if n_matched < len(state) * 0.5:
        print(f"    [WARN] eager reload: low match "
              f"({n_matched}/{len(state)}); using original")
        return enc_module
    print(f"    [OK] eager reload: {n_matched}/{len(state)} matched")
    return new_enc


# ── Per-model evaluation ─────────────────────────────────────────────────────

def evaluate_model(name: str, repo: str, family: str,
                   img_size: int, patch_size: int,
                   samples: list, image_dir: str,
                   geo_tau: float, device: torch.device,
                   dtype: torch.dtype, feature_size_override: int) -> dict:
    print(f"\n{'='*64}")
    print(f"  [{name}]  family={family}  repo={repo}")
    print(f"{'='*64}")

    import gc
    from transformers import AutoModelForImageTextToText, AutoProcessor

    enc = None

    # ── CLIP: load directly with eager attention (no full LLM needed) ─────────
    if family == "clip":
        from transformers import CLIPVisionModel, CLIPImageProcessor
        print(f"  Loading CLIP directly from {_CLIP_BASE} (eager, no LLM) ...")
        try:
            enc = CLIPVisionModel.from_pretrained(
                _CLIP_BASE, torch_dtype=dtype, attn_implementation="eager"
            ).to(device).eval()
            proc = CLIPImageProcessor.from_pretrained(_CLIP_BASE)
        except Exception as e:
            print(f"  [ERROR] CLIP load failed: {e}")
            return {"error": str(e)}

    # ── All other families: GPU bfloat16 load → extract enc state → free LLM ──
    else:
        # Load full model to GPU in bfloat16 (fast disk→GPU path)
        load_dtype = torch.bfloat16
        print(f"  Loading full model (GPU bf16) ...")
        model = None
        for local_only in [False, True]:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    repo, torch_dtype=load_dtype, device_map="cuda",
                    low_cpu_mem_usage=True,
                    local_files_only=local_only,
                )
                break
            except Exception as e:
                if local_only:
                    print(f"  [ERROR] model load failed: {e}")
                    return {"error": str(e)}
                # gated / network error → retry with local cache only
                print(f"  [WARN] online load failed ({type(e).__name__}), retrying local_files_only ...")
        if model is None:
            return {"error": "model load failed"}
        model.eval()

        # Extract vision tower + copy its state dict to CPU
        try:
            enc_raw = get_vision_tower(model, family)
            enc_state_cpu = {k: v.cpu() for k, v in enc_raw.state_dict().items()}
            enc_cfg = getattr(enc_raw, "config", None)
            enc_cls = type(enc_raw)
        except Exception as e:
            print(f"  [ERROR] vision tower extraction failed: {e}")
            del model; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            return {"error": str(e)}

        # Free the full model (LLM + vision) from GPU
        del model; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

        # Re-create vision encoder on GPU with eager attention
        print("  Reloading vision encoder (GPU, eager attn) ...")
        try:
            if enc_cfg is not None:
                try:
                    enc_cfg._attn_implementation = "eager"
                except Exception:
                    pass
                enc = enc_cls(enc_cfg)
            else:
                raise RuntimeError("no config")
            missing, unexpected = enc.load_state_dict(enc_state_cpu, strict=False)
            n_matched = len(enc_state_cpu) - len(unexpected)
            print(f"    matched {n_matched}/{len(enc_state_cpu)} tensors")
            enc = enc.to(device=device, dtype=dtype).eval()
        except Exception as e:
            print(f"  [WARN] eager recreate failed ({e}); trying direct move")
            # Last resort: recreate from pretrained (weights from HF, eager)
            enc = enc_cls.__new__(enc_cls)
            enc = enc.to(device=device, dtype=dtype).eval()

        del enc_state_cpu; gc.collect()

        # Load processor
        try:
            proc = get_processor(repo, family)
        except Exception as e:
            print(f"  [ERROR] processor load failed: {e}")
            del enc
            if device.type == "cuda": torch.cuda.empty_cache()
            return {"error": str(e)}

    # 4. Determine feature size
    fs = feature_size_override if feature_size_override > 0 else (img_size // patch_size)
    print(f"  feature_size = {fs}  ({fs}x{fs} = {fs*fs} patches)")

    # 5. Iterate samples
    agg_iou, agg_inb = [], []
    skipped = 0
    for idx, s in enumerate(samples):
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

        try:
            pv, fs_actual, ekw = preprocess_image(proc, image, family,
                                                   img_size, device, dtype)
            use_fs = fs_actual if fs_actual > 0 else fs

            smap = extract_attn_map(enc, pv, family, use_fs, extra_kwargs=ekw)
            smap = smap.to(device)

            fg_mask = compute_geometric_mask([boxes], use_fs, device, geo_tau)[0]

            metrics = compute_attn_iou(smap, fg_mask)
            agg_iou.append(metrics["attn_iou"])
            agg_inb.append(metrics["attn_in_box"])

        except Exception as e:
            print(f"    [WARN] sample {idx} failed: {e}")
            skipped += 1
            continue

        if (idx + 1) % 20 == 0:
            valid = [x for x in agg_iou if not math.isnan(x)]
            running = sum(valid) / len(valid) if valid else float("nan")
            print(f"  [{idx+1}/{len(samples)}] running Attn-IoU = {running:.4f}")

    # 6. Aggregate
    def nanmean(vals):
        clean = [x for x in vals if not math.isnan(x)]
        return sum(clean) / len(clean) if clean else float("nan")

    result = {
        "attn_iou": nanmean(agg_iou),
        "attn_in_box": nanmean(agg_inb),
        "bg_attn_ratio": 1.0 - nanmean(agg_inb),
        "n_eval": len(agg_iou),
        "n_skipped": skipped,
        "family": family,
        "feature_size": fs,
        "notes": REGISTRY[name][4],
    }
    print(f"  → Attn-IoU = {result['attn_iou']:.4f}  "
          f"attn_in_box = {result['attn_in_box']:.4f}  "
          f"(n={result['n_eval']}, skip={skipped})")

    # 7. Cleanup
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Baseline Attn-IoU across 11 VLMs")
    p.add_argument("--data_path",   required=True)
    p.add_argument("--image_dir",   required=True)
    p.add_argument("--num_samples", type=int, default=-1,
                   help="-1 = all 288 SMD samples")
    p.add_argument("--models",      nargs="+", default=list(REGISTRY.keys()),
                   help="subset of models to run (default=all 11)")
    p.add_argument("--geo_tau",     type=float, default=0.5)
    p.add_argument("--dtype",       default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--output_json", default=None)
    p.add_argument("--feature_size_override", type=int, default=0,
                   help="Force feature grid size (0=auto per model)")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  dtype: {args.dtype}")

    with open(args.data_path) as f:
        data = json.load(f)
    if args.num_samples > 0:
        data = data[:args.num_samples]
    print(f"Samples: {len(data)}  (from {args.data_path})")

    all_results = {}
    for name in args.models:
        if name not in REGISTRY:
            print(f"[SKIP] unknown model: {name}")
            continue
        repo, family, img_size, patch_size, _ = REGISTRY[name]
        try:
            r = evaluate_model(
                name=name, repo=repo, family=family,
                img_size=img_size, patch_size=patch_size,
                samples=data, image_dir=args.image_dir,
                geo_tau=args.geo_tau, device=device, dtype=dtype,
                feature_size_override=args.feature_size_override,
            )
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            traceback.print_exc()
            r = {"error": str(e)}
        all_results[name] = r

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  11-VLM Geo Validation — Baseline Attn-IoU on SMD onshore real data")
    print(f"{'='*80}")
    hdr = (f"{'model':<13}{'family':<10}{'Attn-IoU↑':>11}"
           f"{'attn_in_box↑':>14}{'bg_ratio↓':>11}{'F':>5}{'n':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<13}  ERROR: {r['error'][:60]}")
            continue
        fam = r.get("family", "?")
        print(f"{name:<13}{fam:<10}{r['attn_iou']:>11.4f}"
              f"{r['attn_in_box']:>14.4f}{r['bg_attn_ratio']:>11.4f}"
              f"{r.get('feature_size',0):>5d}{r['n_eval']:>6d}")
    print()
    print("Attn-IoU = area-matched top-K (K=GT box area in patches).")
    print("Reference: our GAA inbox_ce CLIP → Attn-IoU ≈ 0.44 (sim-to-real).")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()

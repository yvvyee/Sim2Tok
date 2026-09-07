"""
Cross-architecture geometric training at 10,000-sample scale, one model per GPU.

``gaa/geo_train_vlm.py`` is the script behind the submitted cross-architecture
table and is left untouched so those numbers stay reproducible.  This module
reuses its family adapters and changes three things that matter at scale:

* ``--image_dir``, so a corpus with relative image paths can be used;
* failures are counted by exception type and reported, and a run whose failure
  rate exceeds ``--max_fail_frac`` aborts.  The original swallows every
  exception with ``continue``, which at 10,000 samples could report a finished
  epoch after skipping most of it;
* an unreadable image is skipped rather than replaced with a blank canvas, since
  a blank image still yields a gradient that pushes attention nowhere.

Batching is deliberately not attempted.  The families in the registry disagree
on preprocessing (fixed grids for CLIP and SigLIP, dynamic tiling for Qwen and
InternViT), so a uniform batch axis does not exist without per-family padding
logic that would itself become a confound.  The hardware is used instead by
running four models concurrently, one per allowed GPU; see
``run_cross_arch.sh``.

Usage::

    CUDA_VISIBLE_DEVICES=4 python -m additional.train_geo_vlm_large \\
        --model smolvlm \\
        --data_path data/aivn_10k_describe.json \\
        --image_dir /home/yvvyee/data/AIVN-SDS/20260728/png \\
        --eval_data data/smd_sweep60.json \\
        --eval_imgdir frames \\
        --output_dir checkpoints/xarch_smolvlm \\
        --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from gaa.geometric_loss import compute_geometric_mask
from gaa.geo_losses import geo_loss, VARIANTS
from gaa.geo_validate_vlms import (
    REGISTRY, get_vision_tower, get_processor, preprocess_image,
    extract_attn_map, compute_attn_iou,
)
from gaa.geo_train_vlm import extract_spatial_attn_train

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


def load_samples(data_path: str, image_dir: str) -> list[dict]:
    """Samples that carry a box, with image paths resolved against image_dir."""
    with open(data_path) as f:
        raw = json.load(f)
    out = []
    for s in raw:
        if not s.get("bboxes"):
            continue
        path = s["image"]
        if not os.path.isabs(path) and image_dir:
            path = os.path.join(image_dir, path)
        out.append({"path": path, "bboxes": s["bboxes"],
                    "video": s.get("video", s.get("id", "")),
                    "split": s.get("split", ""),
                    "condition": s.get("condition", ""),
                    "id": s.get("id", "")})
    return out


def train(encoder, proc, family, img_size, feature_size, samples, args,
          device, dtype) -> dict:
    params = [p for p in encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(samples)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))

    encoder.train()
    fails: Counter[str] = Counter()
    history = []
    step = 0
    t0 = time.time()
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0

        for i in order:
            s = samples[i]
            try:
                image = Image.open(s["path"]).convert("RGB")
            except (OSError, ValueError):
                fails["unreadable_image"] += 1
                continue
            try:
                pv, fs_actual, ekw = preprocess_image(
                    proc, image, family, img_size, device, dtype)
                use_fs = fs_actual if fs_actual > 0 else feature_size
                attn = extract_spatial_attn_train(
                    encoder, pv, family, use_fs, ekw)
                if attn is None:
                    fails["no_attention"] += 1
                    continue
                fg = compute_geometric_mask([s["bboxes"]], use_fs, device,
                                            args.geo_tau)
                if fg.sum() == 0:
                    fails["empty_mask"] += 1
                    continue
                loss = geo_loss(args.variant, attn.float(), fg.float(),
                                margin=args.margin) * args.lam
                if not torch.isfinite(loss):
                    fails["non_finite_loss"] += 1
                    opt.zero_grad(set_to_none=True)
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()

                run_loss += loss.item()
                run_n += 1
                step += 1
            except torch.cuda.OutOfMemoryError:
                fails["cuda_oom"] += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            except (RuntimeError, ValueError, TypeError, IndexError) as e:
                fails[type(e).__name__] += 1
                opt.zero_grad(set_to_none=True)

            if args.log_every and step and step % args.log_every == 0:
                print(f"  step {step}  loss={run_loss / max(run_n, 1):.4f}"
                      f"  fails={sum(fails.values())}"
                      f"  {time.time() - t0:.0f}s", flush=True)

        avg = run_loss / max(run_n, 1)
        history.append({"epoch": epoch + 1, "loss": avg, "n_trained": run_n})
        print(f"[xarch] epoch {epoch + 1}/{args.epochs} loss={avg:.4f} "
              f"trained={run_n}/{len(samples)}", flush=True)

    n_fail = sum(fails.values())
    attempted = args.epochs * len(samples)
    frac = n_fail / max(attempted, 1)
    if frac > args.max_fail_frac:
        raise RuntimeError(
            f"{n_fail}/{attempted} samples failed ({frac:.1%}), above the "
            f"--max_fail_frac {args.max_fail_frac:.1%} limit: {dict(fails)}")

    encoder.eval()
    return {"steps": step, "history": history, "fails": dict(fails),
            "fail_fraction": round(frac, 6),
            "wall_seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def evaluate(encoder, proc, family, img_size, feature_size, samples, args,
             device, dtype) -> tuple[dict, list[dict]]:
    """Per-sample alignment on the evaluation set, with a video-level interval."""
    from additional.stats import cluster_bootstrap_mean

    encoder.eval()
    records: list[dict] = []
    fails: Counter[str] = Counter()
    for s in samples:
        try:
            image = Image.open(s["path"]).convert("RGB")
        except (OSError, ValueError):
            fails["unreadable_image"] += 1
            continue
        try:
            pv, fs_actual, ekw = preprocess_image(
                proc, image, family, img_size, device, dtype)
            use_fs = fs_actual if fs_actual > 0 else feature_size
            smap = extract_attn_map(encoder, pv, family, use_fs,
                                    extra_kwargs=ekw)
            fg = compute_geometric_mask([s["bboxes"]], use_fs, device,
                                        args.geo_tau)[0]
            m = compute_attn_iou(smap.to(device), fg)
            if math.isnan(m["attn_iou"]):
                fails["nan_metric"] += 1
                continue
            records.append({"id": s["id"], "video": s["video"],
                            "split": s["split"], "condition": s["condition"],
                            "attn_iou": m["attn_iou"]})
        except torch.cuda.OutOfMemoryError:
            fails["cuda_oom"] += 1
            torch.cuda.empty_cache()
        except (RuntimeError, ValueError, TypeError, IndexError) as e:
            fails[type(e).__name__] += 1

    if not records:
        return {"n": 0, "fails": dict(fails)}, []
    est = cluster_bootstrap_mean([r["attn_iou"] for r in records],
                                 [r["video"] for r in records])
    return {"n": len(records), "attn_iou": est.as_dict(),
            "fails": dict(fails)}, records


def parse_args():
    p = argparse.ArgumentParser("cross-architecture geo training at scale")
    p.add_argument("--model", required=True, choices=list(REGISTRY))
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", default="")
    p.add_argument("--eval_data", required=True)
    p.add_argument("--eval_imgdir", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", default="inbox_ce", choices=list(VARIANTS))
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1,
                   help="use only the first N training samples")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--max_fail_frac", type=float, default=0.05,
                   help="abort if more than this fraction of samples failed")
    p.add_argument("--eval_before", action="store_true", default=True,
                   help="also measure the untrained encoder for the "
                        "before/after pair")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    train_samples = load_samples(args.data_path, args.image_dir)
    if 0 < args.limit < len(train_samples):
        train_samples = train_samples[:args.limit]
    eval_samples = load_samples(args.eval_data, args.eval_imgdir)

    print(f"[xarch] model={args.model} train={len(train_samples):,} "
          f"eval={len(eval_samples):,} dtype={args.dtype}", flush=True)

    encoder, proc, family, img_size, feature_size = load_encoder(args, dtype,
                                                                 device)

    before = None
    if args.eval_before:
        before, before_recs = evaluate(encoder, proc, family, img_size,
                                       feature_size, eval_samples, args,
                                       device, dtype)
        print(f"[xarch] before: {before.get('attn_iou', {}).get('point')}",
              flush=True)

    stats = train(encoder, proc, family, img_size, feature_size,
                  train_samples, args, device, dtype)
    after, after_recs = evaluate(encoder, proc, family, img_size, feature_size,
                                 eval_samples, args, device, dtype)

    ckpt = os.path.join(args.output_dir, "vision_encoder.bin")
    torch.save(encoder.state_dict(), ckpt)

    out = {
        "model": args.model,
        "hf_repo": REGISTRY[args.model][0],
        "encoder_note": REGISTRY[args.model][4],
        "family": family,
        "img_size": img_size,
        "feature_size": feature_size,
        "variant": args.variant,
        "lam": args.lam,
        "epochs": args.epochs,
        "lr": args.lr,
        "dtype": args.dtype,
        "seed": args.seed,
        "n_train": len(train_samples),
        "n_eval": len(eval_samples),
        "train": stats,
        "before": before,
        "after": after,
        "checkpoint": ckpt,
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.output_dir, "per_sample_after.jsonl"), "w") as f:
        for r in after_recs:
            f.write(json.dumps(r) + "\n")
    if before is not None:
        with open(os.path.join(args.output_dir,
                               "per_sample_before.jsonl"), "w") as f:
            for r in before_recs:
                f.write(json.dumps(r) + "\n")

    b = (before or {}).get("attn_iou", {}).get("point")
    a = after.get("attn_iou", {}).get("point")
    print(f"[xarch] {args.model}: before={b} after={a} "
          f"steps={stats['steps']} fails={stats['fail_fraction']:.2%}")
    print(f"[xarch] -> {args.output_dir}/result.json")


def load_encoder(args, dtype, device):
    """Standalone vision tower plus its processor, ready for gradient updates.

    Mirrors the loading sequence of ``gaa/geo_train_vlm.py`` so that the tower
    this study trains is the same module the submitted work trained: CLIP is
    instantiated directly, and every other family is pulled out of the full VLM,
    which is then freed so only the encoder occupies the GPU.

    Eager attention is needed to read attention probabilities, with one
    documented exception carried over from the original: for SigLIP at a patch
    grid wider than 36, eager materialises the full N-by-N matrix and runs out
    of memory, so those towers keep the fused kernel and are measured with the
    feature-norm saliency proxy instead of attention.
    """
    import gc

    repo, family, img_size, patch_size, notes = REGISTRY[args.model]
    feature_size = img_size // patch_size

    if family == "clip":
        from transformers import CLIPVisionModel, CLIPImageProcessor
        base = "openai/clip-vit-large-patch14-336"
        enc = CLIPVisionModel.from_pretrained(
            base, torch_dtype=dtype, attn_implementation="eager").to(device)
        proc = CLIPImageProcessor.from_pretrained(base)
    else:
        from transformers import AutoModelForImageTextToText
        model = None
        last_err = None
        for local_only in (False, True):
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    repo, torch_dtype=torch.bfloat16, device_map="cuda",
                    low_cpu_mem_usage=True, local_files_only=local_only)
                break
            except (OSError, ValueError, ImportError) as e:
                last_err = e
        if model is None:
            raise RuntimeError(f"could not load {repo}: {last_err}")

        model.eval()
        enc_raw = get_vision_tower(model, family)
        enc_state = {k: v.cpu() for k, v in enc_raw.state_dict().items()}
        enc_cfg = getattr(enc_raw, "config", None)
        enc_cls = type(enc_raw)
        del model, enc_raw
        gc.collect()
        torch.cuda.empty_cache()

        if enc_cfg is None:
            raise RuntimeError(f"{args.model}: vision tower has no config to "
                               f"rebuild from")
        if not (family == "siglip" and feature_size > 36):
            enc_cfg._attn_implementation = "eager"
        enc = enc_cls(enc_cfg)
        enc.load_state_dict(enc_state, strict=False)
        del enc_state
        gc.collect()
        enc = enc.to(device=device, dtype=dtype)
        proc = get_processor(repo, family)

    for p in enc.parameters():
        p.requires_grad = True
    return enc, proc, family, img_size, feature_size


if __name__ == "__main__":
    main()

"""
Batched, multi-GPU attention-alignment evaluation with per-sample output.

``gaa/evaluate_attn.py`` walks one image at a time on one device and reports
only the mean.  That is enough for a 288-image probe set but not for the
29,264-frame full-SMD pass, and a mean alone cannot support a video-level
interval.  This module keeps the metric definitions identical and changes only
the plumbing:

* frames are loaded by DataLoader workers and forwarded in batches;
* the work is sharded across the allowed GPUs with ``torchrun``;
* every sample is written out individually (``attn_iou``, ``attn_in_box``,
  ``video``, ``condition``), which is what ``additional.stats`` needs to
  resample videos rather than frames.

Metric definitions, unchanged from the submitted work:

``attn_iou``
    Area-matched IoU.  The top-K attended patches are selected, where K is the
    ground-truth foreground area in patches, and scored against the foreground
    mask.  Because ``|pred| == |fg| == K``, the union is ``2K - inter``, so the
    score is parameter-free.
``attn_in_box``
    Fraction of the head-averaged CLS attention mass inside the boxes.

The foreground mask comes from ``gaa.geometric_loss.compute_geometric_mask``,
so the ``tau`` rule matches training exactly.  Note that the function
thresholds *patch coverage* rather than textbook IoU, as its own comment
records; this evaluator inherits that definition on purpose.

Memory note: ``output_attentions=True`` materialises the attention tensor for
every layer, which is ``24 x B x 16 x 577^2`` floats for CLIP-ViT-L/14-336
(about 0.5 GB per unit of batch in float32).  ``--batch_size 16`` needs roughly
8 GB and leaves ample headroom on a 40 GB card.

Usage (single node, four allowed GPUs)::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.eval_attn_mp \\
        --data_path  data/smd_full.json \\
        --image_dir  frames \\
        --ckpt stock=none \\
        --ckpt gaa=checkpoints/inbox_ce_10k/vision_encoder.bin \\
        --out_dir results/full_smd
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from gaa.geometric_loss import compute_geometric_mask
from additional.stats import video_of

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


# ── Dataset ──────────────────────────────────────────────────────────────────

class FrameSet(Dataset):
    """Frames with boxes, preprocessed by the encoder's own image processor."""

    def __init__(self, samples: list[dict], image_dir: str, processor):
        self.samples = samples
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        path = os.path.join(self.image_dir, s["image"])
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return pv, s


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, []
    pv = torch.stack([b[0] for b in batch])
    return pv, [b[1] for b in batch]


# ── Model ────────────────────────────────────────────────────────────────────

def _strip_prefix(state: dict, prefix: str) -> dict:
    if state and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_encoder(vision_model: str, ckpt: str | None, dtype, device):
    """CLIPVisionModel with eager attention, optionally overlaid with a checkpoint.

    Eager attention is required because the fused SDPA kernel does not return
    attention probabilities, and the probabilities are the measured quantity.
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor

    model = CLIPVisionModel.from_pretrained(
        vision_model, torch_dtype=dtype, attn_implementation="eager")
    processor = CLIPImageProcessor.from_pretrained(vision_model)

    report = {"ckpt": ckpt, "matched": 0, "total": 0}
    if ckpt and ckpt.lower() not in ("none", ""):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = _strip_prefix(state, "model.")
        missing, unexpected = model.load_state_dict(state, strict=False)
        report = {"ckpt": ckpt, "matched": len(state) - len(unexpected),
                  "total": len(state), "missing": len(missing),
                  "unexpected": len(unexpected)}
        if report["matched"] == 0:
            raise RuntimeError(
                f"no vision weights matched from {ckpt}; the run would silently "
                f"measure stock CLIP")
    return model.to(device).eval(), processor, report


@torch.no_grad()
def cls_attention(model, pixel_values, feature_size: int) -> torch.Tensor:
    """Head-averaged last-layer CLS->patch attention, shaped ``(B, F, F)``."""
    out = model(pixel_values=pixel_values, output_attentions=True)
    attn = out.attentions[-1]                    # (B, H, N+1, N+1)
    cls = attn[:, :, 0, 1:].mean(dim=1)          # (B, N) averaged over heads
    B = cls.shape[0]
    return cls.reshape(B, feature_size, feature_size).float()


# ── Metrics ──────────────────────────────────────────────────────────────────

def batch_metrics(cls_attn: torch.Tensor, fg: torch.Tensor) -> dict:
    """Per-sample ``attn_iou`` and ``attn_in_box`` for a whole batch.

    ``cls_attn`` and ``fg`` are ``(B, F, F)``.  The area-matched selection is
    done with a rank matrix instead of a per-sample ``topk`` so that a variable
    K stays vectorised.
    """
    B = cls_attn.shape[0]
    a = cls_attn.reshape(B, -1)
    m = fg.reshape(B, -1).to(a.dtype)
    n = a.shape[1]

    total = a.sum(dim=1).clamp(min=1e-8)
    attn_in_box = (a * m).sum(dim=1) / total

    k = m.sum(dim=1)                                     # (B,) foreground area
    order = a.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    idx = torch.arange(n, device=a.device).expand(B, n)
    rank.scatter_(1, order, idx)
    pred = (rank < k.unsqueeze(1).clamp(min=0)).to(a.dtype)

    inter = (pred * m).sum(dim=1)
    union = (pred + m).clamp(max=1.0).sum(dim=1).clamp(min=1e-8)
    iou = inter / union
    iou = torch.where(k > 0, iou, torch.full_like(iou, float("nan")))

    return {"attn_iou": iou, "attn_in_box": attn_in_box, "k": k}


# ── Runner ───────────────────────────────────────────────────────────────────

def parse_ckpt_specs(specs: list[str]) -> list[tuple[str, str | None]]:
    """``["stock=none", "gaa=/path/x.bin"]`` -> ``[("stock", None), ...]``."""
    out: list[tuple[str, str | None]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--ckpt needs LABEL=PATH form, got {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label, None if path.lower() in ("none", "") else path))
    return out


def parse_args():
    p = argparse.ArgumentParser("multi-GPU attention-alignment evaluation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", action="append", required=True,
                   help="LABEL=PATH; PATH 'none' evaluates the stock encoder. "
                        "Repeatable.")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--feature_size", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    p.add_argument("--limit", type=int, default=-1,
                   help="evaluate only the first N samples (smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = DTYPES[args.dtype]

    with open(args.data_path) as f:
        samples = json.load(f)
    if args.limit > 0:
        samples = samples[:args.limit]
    shard = samples[rank::world]

    feature_size = args.feature_size
    if feature_size == -1:
        from transformers import CLIPVisionConfig
        cfg = CLIPVisionConfig.from_pretrained(args.vision_model)
        feature_size = cfg.image_size // cfg.patch_size

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[eval] {len(samples):,} samples, world={world}, "
              f"feature_size={feature_size}, batch={args.batch_size}", flush=True)

    summary: dict[str, dict] = {}
    for label, ckpt in parse_ckpt_specs(args.ckpt):
        model, processor, report = load_encoder(args.vision_model, ckpt,
                                                dtype, device)
        loader = DataLoader(
            FrameSet(shard, args.image_dir, processor),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
            pin_memory=True,
        )

        records: list[dict] = []
        for pv, metas in loader:
            if pv is None:
                continue
            pv = pv.to(device=device, dtype=dtype, non_blocking=True)
            attn = cls_attention(model, pv, feature_size)
            fg = compute_geometric_mask([m["bboxes"] for m in metas],
                                        feature_size, device, args.geo_tau)
            m = batch_metrics(attn, fg)
            iou = m["attn_iou"].tolist()
            inbox = m["attn_in_box"].tolist()
            karea = m["k"].tolist()
            for meta, i_, b_, k_ in zip(metas, iou, inbox, karea):
                records.append({
                    "id": meta["id"],
                    "video": video_of(meta),
                    "split": meta.get("split", ""),
                    "condition": meta.get("condition", ""),
                    "attn_iou": i_,
                    "attn_in_box": b_,
                    "fg_patches": int(k_),
                })

        del model
        torch.cuda.empty_cache()

        if distributed:
            gathered: list[list[dict]] = [None] * world       # type: ignore[list-item]
            dist.all_gather_object(gathered, records)
            merged = [r for part in gathered for r in part]
        else:
            merged = records

        if rank == 0:
            merged.sort(key=lambda r: r["id"])
            path = os.path.join(args.out_dir, f"per_sample_{label}.jsonl")
            with open(path, "w") as f:
                for r in merged:
                    f.write(json.dumps(r) + "\n")

            from additional.stats import cluster_bootstrap_mean
            est = cluster_bootstrap_mean([r["attn_iou"] for r in merged],
                                         [r["video"] for r in merged])
            summary[label] = {
                "load_report": report,
                "n_samples": len(merged),
                "attn_iou": est.as_dict(),
                "attn_in_box_mean": (
                    sum(r["attn_in_box"] for r in merged) / len(merged)
                    if merged else float("nan")),
                "per_sample_path": path,
            }
            print(f"[eval] {label:12s} Attn-IoU {est}", flush=True)

    if rank == 0:
        meta = {
            "data_path": args.data_path,
            "image_dir": args.image_dir,
            "vision_model": args.vision_model,
            "geo_tau": args.geo_tau,
            "feature_size": feature_size,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "world_size": world,
            "n_requested": len(samples),
            "results": summary,
        }
        out = os.path.join(args.out_dir, "summary.json")
        with open(out, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[eval] summary -> {out}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

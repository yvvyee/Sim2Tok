"""
Multi-GPU vision-only geometric-alignment training.

``gaa/geo_train_vlm.py`` trains one sample at a time on one device and wraps the
step in a bare ``except Exception: continue``, so a corpus 38 times larger would
be both slow and able to skip most of itself without saying so.  This trainer
keeps the objective identical and changes the execution:

* DistributedDataParallel over the allowed GPUs, launched with ``torchrun``;
* real batches, so the 10,000-sample corpus finishes in minutes;
* failures are counted per category and reported, never silently dropped;
* seeds are set for Python, NumPy and Torch on every rank, and the sampler is
  seeded per epoch, so a rerun reproduces the checkpoint.

Only the vision encoder is trained.  The loss is ``L_geo`` alone, which is the
paper's vision-only stage; the projector and language model are untouched here
and are fine-tuned separately in the two-stage recipe.

Memory note: gradients must flow through the attention probabilities, so
``output_attentions=True`` retains the attention tensor of every layer for the
backward pass.  For CLIP-ViT-L/14-336 that is ``24 x B x 16 x 577^2`` floats,
roughly 0.5 GB per unit of batch in float32.  A per-device batch of 8 therefore
needs about 4 GB for attention plus the rest of the ViT-L graph, which fits an
A100-40GB with wide margin.

Usage::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.train_geo_ddp \\
        --data_path data/aivn_10k_describe.json \\
        --image_dir /home/yvvyee/data/AIVN-SDS/20260728/png \\
        --output_dir checkpoints/inbox_ce_10k \\
        --variant inbox_ce --epochs 5 --lr 1e-5 --per_device_batch 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from gaa.geo_losses import VARIANTS, compute_geo_loss_from_attn

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


# ── Dataset ──────────────────────────────────────────────────────────────────

class GeoTrainSet(Dataset):
    """Images with at least one privileged box, preprocessed for the encoder.

    Samples whose image cannot be read are reported rather than replaced with a
    blank tensor: a blank image still produces a gradient and would quietly bias
    the encoder toward attending nowhere.
    """

    def __init__(self, data_path: str, image_dir: str, processor):
        with open(data_path) as f:
            raw = json.load(f)
        self.samples = [s for s in raw if s.get("bboxes")]
        self.n_dropped_no_box = len(raw) - len(self.samples)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        path = s["image"]
        if not os.path.isabs(path):
            path = os.path.join(self.image_dir, path)
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return pv, s["bboxes"]


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, 0
    n_bad = 0
    pv = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    return pv, boxes, n_bad


# ── Setup ────────────────────────────────────────────────────────────────────

def seed_everything(seed: int, rank: int) -> None:
    """Seed all RNGs. Rank offset keeps dataloader shuffling from being identical."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def parse_args():
    p = argparse.ArgumentParser("multi-GPU vision-only geo training")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--variant", default="inbox_ce", choices=list(VARIANTS))
    p.add_argument("--lam", type=float, default=1.0,
                   help="scale on L_geo; the vision-only stage has no other term, "
                        "so this only rescales the gradient")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.0,
                   help="only used by the fg_bg_margin variant")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="stop after N optimiser steps (-1 = full schedule)")
    p.add_argument("--log_every", type=int, default=20)
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
    seed_everything(args.seed, rank)

    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

    cfg = CLIPVisionConfig.from_pretrained(args.vision_model)
    feature_size = cfg.image_size // cfg.patch_size

    encoder = CLIPVisionModel.from_pretrained(
        args.vision_model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    encoder.train()
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    dataset = GeoTrainSet(args.data_path, args.image_dir, processor)
    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank,
                                  shuffle=True, seed=args.seed)
               if distributed else None)
    loader = DataLoader(dataset, batch_size=args.per_device_batch,
                        sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=True, drop_last=False)

    # ``find_unused_parameters`` is required, not a convenience.  The objective
    # reads only ``attentions[-1]``, which is produced before
    # ``vision_model.post_layernorm`` runs, so that layer's weights receive no
    # gradient and DDP would otherwise abort waiting for their reduction.  The
    # single-GPU training in the submitted work left those weights at their
    # pretrained values for the same reason, so enabling this keeps the two
    # setups numerically comparable rather than changing what is trained.
    model = (DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
             if distributed else encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(loader))
    total_steps = (args.max_steps if args.max_steps > 0
                   else args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[train] {len(dataset):,} samples "
              f"({dataset.n_dropped_no_box} dropped for having no box)")
        print(f"[train] world={world} per_device_batch={args.per_device_batch} "
              f"effective_batch={world * args.per_device_batch}")
        print(f"[train] variant={args.variant} lam={args.lam} lr={args.lr} "
              f"epochs={args.epochs} total_steps={total_steps} "
              f"dtype={args.dtype} seed={args.seed}", flush=True)

    skipped: Counter[str] = Counter()
    history: list[dict] = []
    step = 0
    t0 = time.time()
    stop = False

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        run_loss, run_n = 0.0, 0

        for pv, boxes, _ in loader:
            if pv is None:
                skipped["unreadable_batch"] += 1
                continue
            pv = pv.to(device=device, dtype=dtype, non_blocking=True)

            out = model(pixel_values=pv, output_attentions=True)
            loss = compute_geo_loss_from_attn(
                args.variant, out.attentions[-1], boxes,
                feature_size=feature_size, tau=args.geo_tau, margin=args.margin)
            if not torch.isfinite(loss):
                skipped["non_finite_loss"] += 1
                opt.zero_grad(set_to_none=True)
                continue
            loss = loss * args.lam

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()

            run_loss += loss.item()
            run_n += 1
            step += 1

            if rank == 0 and step % args.log_every == 0:
                print(f"  step {step}/{total_steps}  loss={run_loss / max(run_n, 1):.4f}"
                      f"  lr={sched.get_last_lr()[0]:.2e}"
                      f"  {time.time() - t0:.0f}s", flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break

        epoch_loss = run_loss / max(run_n, 1)
        if distributed:
            t = torch.tensor([run_loss, float(run_n)], device=device)
            dist.all_reduce(t)
            epoch_loss = (t[0] / t[1].clamp(min=1)).item()
        if rank == 0:
            history.append({"epoch": epoch + 1, "loss": epoch_loss, "steps": step})
            print(f"[train] epoch {epoch + 1}/{args.epochs} "
                  f"loss={epoch_loss:.4f} steps={step}", flush=True)
        if stop:
            break

    if rank == 0:
        core = model.module if distributed else model
        core.eval()
        ckpt = os.path.join(args.output_dir, "vision_encoder.bin")
        torch.save(core.state_dict(), ckpt)

        meta = {
            "checkpoint": ckpt,
            "data_path": args.data_path,
            "image_dir": args.image_dir,
            "vision_model": args.vision_model,
            "feature_size": feature_size,
            "variant": args.variant,
            "lam": args.lam,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "per_device_batch": args.per_device_batch,
            "effective_batch": world * args.per_device_batch,
            "world_size": world,
            "geo_tau": args.geo_tau,
            "dtype": args.dtype,
            "seed": args.seed,
            "n_samples": len(dataset),
            "n_dropped_no_box": dataset.n_dropped_no_box,
            "total_steps": step,
            "wall_seconds": round(time.time() - t0, 1),
            "history": history,
            "skipped": dict(skipped),
        }
        with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[train] saved {ckpt} after {step} steps "
              f"in {meta['wall_seconds']}s")
        if skipped:
            print(f"[train] skipped {dict(skipped)}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

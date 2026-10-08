"""
Task-level sim-to-real measurement with the vision encoder frozen.

For one encoder, a fresh per-token head (vessel occupancy and vessel centre,
the ``ProbeHead`` of ``train_token_ddp``) is trained on the AIVN-SDS training
split only: frames, vessel crops and vessel-free crops.  The encoder is frozen
and the head, data, steps and seed are the same for every encoder, so the head
measures what the encoder's layer-23 patch tokens (the tokens the VLM reads)
make linearly-ish available.  It is then applied, unchanged, to the held-out
AIVN validation split and to the real datasets.

Per image it stores, in the crop geometry the encoder sees:

``n_visible``       vessel boxes inside the centre crop
``max_occ``         highest token occupancy probability (presence score)
``max_center``      highest token centre probability
``counts``          predicted vessel count at centre thresholds 0.1..0.9
                    (3x3 local maxima above the threshold)
``hist_pos/neg``    100-bin histograms of token occupancy probability over
                    tokens a vessel covers / does not cover, from which token
                    average precision is computed over any set of images

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.eval_token_probe --encoder tok=checkpoints/x/vision_encoder.bin \\
        --sim_frames_json data/ok_aivn_10000.json --split_json data/aivn_split.json \\
        --crops_json aivn_crops/crops.json --frame_dir aivn_correct/png \\
        --crop_dir aivn_crops --real smd=data/smd_full.json:frames \\
        --real lars=data/real/lars.json:/mnt/ssd1/yvvyee/realeval \\
        --out_dir results/probe/tok
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from additional import backbone as bb
from additional.config import assert_policy_or_die
from additional.train_token_ddp import ProbeHead, token_targets

THRESH = [round(0.1 * i, 1) for i in range(1, 10)]
NBIN = 100


def sim_items(args, split_name):
    split = json.load(open(args.split_json))["split"]
    items = []
    for r in json.load(open(args.sim_frames_json)):
        if split[r["id"]] != split_name:
            continue
        items.append({"id": r["id"], "path": os.path.join(args.frame_dir, r["image"]),
                      "w": 1920, "h": 1080, "boxes": r["bboxes"], "video": r["id"],
                      "dataset": f"aivn_{split_name}"})
    for c in json.load(open(args.crops_json)):
        if c["split"] != split_name:
            continue
        items.append({"id": c["id"], "path": os.path.join(args.crop_dir, c["image"]),
                      "w": c["width"], "h": c["height"], "boxes": c["bboxes"],
                      "video": c["source_id"], "dataset": f"aivn_{split_name}"})
    return items


def real_items(name, json_path, image_dir):
    out = []
    for r in json.load(open(json_path)):
        if r.get("vessel_labelled") is False:
            continue
        out.append({"id": r["id"], "path": os.path.join(image_dir, r["image"]),
                    "w": r.get("width", 1920), "h": r.get("height", 1080),
                    "boxes": r.get("bboxes") or [], "video": r.get("video", r["id"]),
                    "dataset": name})
    return out


class ProbeSet(Dataset):
    def __init__(self, items, processor, info):
        self.items, self.processor, self.info, self.g = items, processor, info, info.g

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            img = Image.open(it["path"]).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=img, return_tensors="pt").pixel_values[0]
        vis = [{"box": b} for b in bb.map_boxes(self.info, it["boxes"], it["w"], it["h"])]
        occ, center, _, _ = token_targets(vis, self.g, {})
        return pv, occ, center, i, len(vis)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pv, occ, cen, idx, nv = zip(*batch)
    return torch.stack(pv), torch.stack(occ), torch.stack(cen), list(idx), list(nv)


def count_peaks(cen_prob: torch.Tensor, g: int) -> list[list[int]]:
    """Predicted counts per image at each threshold: 3x3 local maxima."""
    m = cen_prob.view(-1, 1, g, g)
    peak = (m == F.max_pool2d(m, 3, 1, 1)).view(m.shape[0], -1)
    return [[int(((cen_prob[b] > t) & peak[b]).sum()) for t in THRESH]
            for b in range(cen_prob.shape[0])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, help="label=path or label=none")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--sim_frames_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--crops_json", required=True)
    p.add_argument("--frame_dir", required=True)
    p.add_argument("--crop_dir", required=True)
    p.add_argument("--real", action="append", default=[],
                   help="name=json_path:image_dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="hidden_states index; default is the layer the VLM reads")
    p.add_argument("--head_steps", type=int, default=1500)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--per_device_batch", type=int, default=16)
    p.add_argument("--center_pos_weight", type=float, default=50.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank); np.random.seed(args.seed + rank)
    assert_policy_or_die()

    label, path = args.encoder.split("=", 1)
    encoder, processor, info = bb.load(args.vision_model, torch.float32)
    if args.layer is None:
        args.layer = info.vlm_layer
    if path != "none":
        sd = torch.load(path, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[probe] loaded {path} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            raise SystemExit(f"encoder weights missing: {missing[:5]}")
    encoder = encoder.to(device).eval().requires_grad_(False)
    g = info.g

    def tokens(pv):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = encoder(pixel_values=pv, output_hidden_states=True).hidden_states
        return bb.patch_tokens(info, hs, args.layer).float()

    # ── head on the simulator training split ──
    torch.manual_seed(args.seed)
    head = ProbeHead(info.hidden, []).to(device)
    head_ddp = DDP(head, device_ids=[local_rank]) if distributed else head
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.head_steps)
    train = ProbeSet(sim_items(args, "train"), processor, info)
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(train, batch_size=args.per_device_batch, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.num_workers,
                        collate_fn=collate, drop_last=True, pin_memory=True,
                        timeout=900 if args.num_workers > 0 else 0)
    pw = torch.tensor(args.center_pos_weight, device=device)
    step, epoch, t0 = 0, 0, time.time()
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[probe] {label}: head on {len(train):,} sim train images, "
              f"{args.head_steps} steps, batch {world * args.per_device_batch}", flush=True)
    while step < args.head_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in loader:
            pv, occ_t, cen_t = b[0].to(device), b[1].to(device), b[2].to(device)
            occ_p, cen_p, _ = head_ddp(tokens(pv))
            loss = (F.binary_cross_entropy_with_logits(occ_p, occ_t)
                    + F.binary_cross_entropy_with_logits(cen_p, cen_t, pos_weight=pw))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            step += 1
            if rank == 0 and step % 100 == 0:
                print(f"  head step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step >= args.head_steps:
                break
        epoch += 1
    head.eval()
    if rank == 0:
        torch.save(head.state_dict(), os.path.join(args.out_dir, "probe_head.bin"))

    # ── evaluation ──
    sets = [("aivn_val", sim_items(args, "val"))]
    for spec in args.real:
        name, rest = spec.split("=", 1)
        jp, idir = rest.rsplit(":", 1)
        sets.append((name, real_items(name, jp, idir)))
    for name, items in sets:
        ds = ProbeSet(items, processor, info)
        idx = list(range(rank, len(ds), world))
        sub = torch.utils.data.Subset(ds, idx)
        dl = DataLoader(sub, batch_size=32, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate, timeout=900 if args.num_workers > 0 else 0)
        rows = []
        for b in dl:
            if b is None:
                continue
            pv, occ_t, _, ii, nv = b
            with torch.no_grad():
                occ_p, cen_p, _ = head(tokens(pv.to(device)))
            occ_p, cen_p = torch.sigmoid(occ_p).cpu(), torch.sigmoid(cen_p).cpu()
            counts = count_peaks(cen_p, g)
            bins = (occ_p * NBIN).long().clamp(max=NBIN - 1)
            for k in range(len(ii)):
                pos = occ_t[k] > 0
                it = items[ii[k]]
                rows.append({"id": it["id"], "video": it["video"], "dataset": name,
                             "n_visible": nv[k], "max_occ": float(occ_p[k].max()),
                             "max_center": float(cen_p[k].max()), "counts": counts[k],
                             "hist_pos": torch.bincount(bins[k][pos], minlength=NBIN).tolist(),
                             "hist_neg": torch.bincount(bins[k][~pos], minlength=NBIN).tolist()})
        part = os.path.join(args.out_dir, f"{name}.rank{rank}.jsonl")
        with open(part, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if distributed:
            dist.barrier()
        if rank == 0:
            merged = os.path.join(args.out_dir, f"{name}.jsonl")
            n = 0
            with open(merged, "w") as f:
                for r_ in range(world):
                    pp = os.path.join(args.out_dir, f"{name}.rank{r_}.jsonl")
                    for line in open(pp):
                        f.write(line); n += 1
                    os.remove(pp)
            print(f"[probe] {label} {name}: {n}/{len(items)} images "
                  f"{time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        json.dump({"encoder": args.encoder, "head_steps": args.head_steps,
                   "head_lr": args.head_lr, "seed": args.seed, "layer": args.layer,
                   "n_train": len(train), "thresholds": THRESH, "nbin": NBIN,
                   "sets": [s for s, _ in sets], "vision_model": args.vision_model,
                   "grid": g, "world": world, "per_device_batch": args.per_device_batch, "wall_seconds": round(time.time() - t0, 1)},
                  open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()

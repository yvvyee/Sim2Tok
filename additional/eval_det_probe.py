"""
Standard detection metric for the frozen-encoder sim-to-real comparison.

For one encoder, a CenterNet-style head is trained on the AIVN-SDS training
split only (frames, vessel crops, vessel-free crops), with the encoder frozen.
The head reads the layer-23 patch tokens (the tokens the VLM reads), lifts the
24x24 grid to 96x96 with two transposed convolutions, and predicts a centre
heatmap, box size and centre offset.  It is then applied unchanged to the AIVN
validation split and to the real datasets.  Boxes are in the 336x336 centre
crop the encoder sees.

Per image the evaluator stores, at IoU 0.5 and 0.75, every detection's score
and whether it matched a ground-truth box (greedy by score), plus the number of
ground-truth boxes, so AP can be computed over any set of images and
bootstrapped over videos.

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.eval_det_probe --encoder tok=checkpoints/x/vision_encoder.bin \\
        --sim_frames_json data/ok_aivn_10000.json --split_json data/aivn_split.json \\
        --crops_json aivn_crops/crops.json --frame_dir aivn_correct/png \\
        --crop_dir aivn_crops --real lars=data/real/lars.json:/mnt/ssd1/yvvyee/realeval \\
        --out_dir results/det/tok_s0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from additional.box_geometry import transform_boxes
from additional.config import assert_policy_or_die
from additional.eval_token_probe import real_items, sim_items

CROP = 336
OUT = 96                       # heatmap resolution
STRIDE = CROP / OUT
MAX_DET = 100


def gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet radius for a box of h x w heatmap cells."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0, int(min(r1, r2, r3)))


def det_targets(boxes):
    """boxes: crop-normalised [x1,y1,x2,y2].  Heatmap, size, offset, mask."""
    hm = np.zeros((OUT, OUT), np.float32)
    wh = np.zeros((2, OUT, OUT), np.float32)
    off = np.zeros((2, OUT, OUT), np.float32)
    mask = np.zeros((OUT, OUT), np.float32)
    for x1, y1, x2, y2 in boxes:
        bw, bh = (x2 - x1) * OUT, (y2 - y1) * OUT
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 * OUT, (y1 + y2) / 2 * OUT
        ix, iy = min(OUT - 1, int(cx)), min(OUT - 1, int(cy))
        r = max(1, gaussian_radius(math.ceil(bh), math.ceil(bw)))
        sigma = (2 * r + 1) / 6
        ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
        g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
        t, b = max(0, iy - r), min(OUT, iy + r + 1)
        l, rr = max(0, ix - r), min(OUT, ix + r + 1)
        hm[t:b, l:rr] = np.maximum(hm[t:b, l:rr],
                                   g[t - (iy - r):b - (iy - r), l - (ix - r):rr - (ix - r)])
        wh[:, iy, ix] = (np.log(bw + 1e-3), np.log(bh + 1e-3))
        off[:, iy, ix] = (cx - ix, cy - iy)
        mask[iy, ix] = 1.0
    return hm, wh, off, mask


class DetSet(Dataset):
    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            img = Image.open(it["path"]).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=img, return_tensors="pt").pixel_values[0]
        vis = transform_boxes(it["boxes"], it["w"], it["h"])
        hm, wh, off, mask = det_targets(vis)
        return (pv, torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(off),
                torch.from_numpy(mask), i)


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    *ts, idx = zip(*batch)
    return [torch.stack(t) for t in ts] + [list(idx)]


class DetHead(nn.Module):
    def __init__(self, dim, g):
        super().__init__()
        self.g = g
        self.reduce = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        def head(o):
            return nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(64, o, 1))
        self.hm, self.wh, self.off = head(1), head(2), head(2)
        nn.init.constant_(self.hm[-1].bias, -2.19)

    def forward(self, tok):
        b = tok.shape[0]
        x = self.reduce(tok).transpose(1, 2).reshape(b, 256, self.g, self.g)
        x = self.up(x)
        x = F.interpolate(x, size=(OUT, OUT), mode="bilinear", align_corners=False)
        return self.hm(x).squeeze(1), self.wh(x), self.off(x)


def focal_loss(logit, gt):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_l.sum() + neg_l.sum()) / n


def decode(hm_logit, wh, off):
    """Top detections per image as (score, x1, y1, x2, y2) in crop-normalised units."""
    hm = torch.sigmoid(hm_logit)
    peak = hm == F.max_pool2d(hm.unsqueeze(1), 3, 1, 1).squeeze(1)
    hm = hm * peak
    b = hm.shape[0]
    scores, inds = hm.view(b, -1).topk(MAX_DET)
    ys, xs = (inds // OUT).float(), (inds % OUT).float()
    out = []
    for k in range(b):
        w = torch.exp(wh[k, 0].view(-1)[inds[k]])
        h = torch.exp(wh[k, 1].view(-1)[inds[k]])
        cx = xs[k] + off[k, 0].view(-1)[inds[k]]
        cy = ys[k] + off[k, 1].view(-1)[inds[k]]
        boxes = torch.stack([(cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)], 1) / OUT
        out.append(torch.cat([scores[k:k + 1].T, boxes], 1))
    return out


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a)[:, None], np.asarray(b)[None]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    ua = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    ub = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(ua + ub - inter, 1e-12)


def match(dets, gts, thr):
    """Greedy by score; returns tp flags aligned with dets (already sorted)."""
    m = iou_mat([d[1:] for d in dets], gts)
    used = set()
    tp = []
    for i in range(len(dets)):
        cand = [jj for jj in np.argsort(-m[i]) if jj not in used and m[i, jj] >= thr] if m.shape[1] else []
        if cand:
            used.add(cand[0]); tp.append(1)
        else:
            tp.append(0)
    return tp


def match_center(dets, gts):
    """Greedy by score: a detection is a hit when its center lies inside a
    not-yet-matched ground-truth box.  Insensitive to the predicted size."""
    used, tp = set(), []
    for d in dets:
        cx, cy = (d[1] + d[3]) / 2, (d[2] + d[4]) / 2
        hit = -1
        for j, g in enumerate(gts):
            if j not in used and g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                hit = j
                break
        if hit >= 0:
            used.add(hit); tp.append(1)
        else:
            tp.append(0)
    return tp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--sim_frames_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--crops_json", required=True)
    p.add_argument("--frame_dir", required=True)
    p.add_argument("--crop_dir", required=True)
    p.add_argument("--real", action="append", default=[])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--head_steps", type=int, default=3000)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--per_device_batch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--min_score", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_only", action="store_true",
                   help="load det_head.bin from out_dir and only evaluate")
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

    from transformers import CLIPImageProcessor, CLIPVisionModel
    label, path = args.encoder.split("=", 1)
    encoder = CLIPVisionModel.from_pretrained(args.vision_model, torch_dtype=torch.float32)
    if path != "none":
        missing, _ = encoder.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        if missing:
            raise SystemExit(f"encoder weights missing: {missing[:5]}")
    encoder = encoder.to(device).eval().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    g = encoder.config.image_size // encoder.config.patch_size

    def tokens(pv):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = encoder(pixel_values=pv, output_hidden_states=True).hidden_states
        return hs[args.layer][:, 1:, :].float()

    torch.manual_seed(args.seed)
    head = DetHead(encoder.config.hidden_size, g).to(device)
    if distributed:
        head = nn.SyncBatchNorm.convert_sync_batchnorm(head)
    head_ddp = DDP(head, device_ids=[local_rank]) if distributed else head
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.head_lr,
                                                total_steps=args.head_steps, pct_start=0.05)
    train = DetSet(sim_items(args, "train"), processor)
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(train, batch_size=args.per_device_batch, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.num_workers,
                        collate_fn=collate, drop_last=True, pin_memory=True,
                        timeout=900 if args.num_workers > 0 else 0)
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[det] {label}: head on {len(train):,} sim train images, {args.head_steps} steps, "
              f"batch {world * args.per_device_batch}", flush=True)
    step, epoch, t0 = 0, 0, time.time()
    head_ddp.train()
    if args.eval_only:
        head.load_state_dict(torch.load(os.path.join(args.out_dir, "det_head.bin"), map_location=device))
        step = args.head_steps
    while step < args.head_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in loader:
            pv, hm_t, wh_t, off_t, m_t = [x.to(device, non_blocking=True) for x in b[:5]]
            hm_p, wh_p, off_p = head_ddp(tokens(pv))
            n = m_t.sum().clamp(min=1)
            l_hm = focal_loss(hm_p, hm_t)
            l_wh = (F.l1_loss(wh_p, wh_t, reduction="none").sum(1) * m_t).sum() / n
            l_off = (F.l1_loss(off_p, off_t, reduction="none").sum(1) * m_t).sum() / n
            loss = l_hm + 0.1 * l_wh + l_off
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            opt.step(); sched.step(); step += 1
            if rank == 0 and step % 100 == 0:
                print(f"  step {step} hm {l_hm.item():.3f} wh {l_wh.item():.3f} "
                      f"off {l_off.item():.3f} {time.time() - t0:.0f}s", flush=True)
            if step >= args.head_steps:
                break
        epoch += 1
    head.eval()
    if rank == 0 and not args.eval_only:
        torch.save(head.state_dict(), os.path.join(args.out_dir, "det_head.bin"))

    sets = [("aivn_val", sim_items(args, "val"))]
    for spec in args.real:
        name, rest = spec.split("=", 1)
        jp, idir = rest.rsplit(":", 1)
        sets.append((name, real_items(name, jp, idir)))
    for name, items in sets:
        ds = DetSet(items, processor)
        sub = torch.utils.data.Subset(ds, list(range(rank, len(ds), world)))
        dl = DataLoader(sub, batch_size=32, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate, timeout=900 if args.num_workers > 0 else 0)
        rows = []
        for b in dl:
            if b is None:
                continue
            pv, idx = b[0], b[-1]
            with torch.no_grad():
                hm_p, wh_p, off_p = head(tokens(pv.to(device)))
            dets = decode(hm_p.float(), wh_p.float(), off_p.float())
            for k, ii in enumerate(idx):
                it = items[ii]
                gts = transform_boxes(it["boxes"], it["w"], it["h"])
                d = dets[k].cpu().numpy()
                d = d[d[:, 0] >= args.min_score]
                d = d[np.argsort(-d[:, 0])].tolist()
                rows.append({"id": it["id"], "video": it["video"], "dataset": name,
                             "n_gt": len(gts), "scores": [round(x[0], 5) for x in d],
                             "tp50": match(d, gts, 0.5), "tp75": match(d, gts, 0.75),
                             "tpc": match_center(d, gts)})
        part = os.path.join(args.out_dir, f"{name}.rank{rank}.jsonl")
        with open(part, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if distributed:
            dist.barrier()
        if rank == 0:
            n = 0
            with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
                for r_ in range(world):
                    pp = os.path.join(args.out_dir, f"{name}.rank{r_}.jsonl")
                    for line in open(pp):
                        f.write(line); n += 1
                    os.remove(pp)
            print(f"[det] {label} {name}: {n}/{len(items)} images {time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        json.dump({"encoder": args.encoder, "head_steps": args.head_steps, "seed": args.seed,
                   "layer": args.layer, "n_train": len(train), "out_res": OUT,
                   "sets": [s for s, _ in sets], "wall_seconds": round(time.time() - t0, 1)},
                  open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()

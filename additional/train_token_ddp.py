"""
Token-level supervision of the vision encoder with simulator navigation state.

GAA supervises the layer-24 CLS->patch attention, while the VLM reads the
layer-23 patch tokens (``hidden_states[-2]``, CLS dropped).  This trainer
supervises those tokens directly.  A light probe head, used only during
training and then discarded, predicts per token:

``occ``      fraction of the 14x14 cell the vessel boxes cover (soft target,
             no threshold, so vessels smaller than one cell still count)
``center``   whether a vessel box centre falls in the cell
``state``    for cells a vessel occupies, the binned navigation state of the
             vessel that covers it most: distance, relative course, CPA, TCPA
             (``--channels state``)

Relative bearing is not a channel: on this corpus it is a linear function of
the box centre x (r = 0.99994), which every token already knows from its
position embedding.

``--shuffle_state`` is the control for the state channels: the state values
are permuted across all vessels of the corpus, so the loss, the label
marginals and the number of supervised cells stay the same while the link to
each vessel's own state is gone.

``--absent`` adds vessel-free images (negative sub-crops and frames whose
vessels the centre crop removes) with all-zero ``occ`` and ``center`` targets.
Without it every training image shows at least one vessel.

All images are simulator images from AIVN-SDS 10k: the 10,000 frames and the
sub-crops ``make_sim_crops`` cut from them.  The total number of optimiser
steps is fixed by ``--total_steps`` so that conditions with more images do not
also get more updates.

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.train_token_ddp --channels occ,state --absent \\
        --frames_json data/ok_aivn_10000.json --nav_json data/aivn_nav_state.json \\
        --crops_json aivn_crops/crops.json --frame_dir aivn_correct/png \\
        --crop_dir aivn_crops --output_dir checkpoints/tok_occ_state_abs_s0
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from additional.box_geometry import transform_boxes
from additional.config import assert_policy_or_die

STATE_KEYS = ["dist_nm", "rel_course", "cpa", "tcpa"]
IGNORE = -100


# ── State bins ───────────────────────────────────────────────────────────────

def make_bins(nav: dict) -> dict:
    """Bin edges from the corpus distribution.

    distance, CPA: quartiles.  relative course: eight 45-degree sectors centred
    on 0 (same heading) and 180 (reciprocal).  TCPA: one bin for negative
    values (CPA already passed), then terciles of the positive values.
    """
    v = [x for vs in nav.values() for x in vs]
    q = lambda k, ps: [float(np.percentile([x[k] for x in v], p)) for p in ps]
    pos_tcpa = [x["tcpa"] for x in v if x["tcpa"] >= 0]
    return {
        "dist_nm": q("dist_nm", [25, 50, 75]),
        "cpa": q("cpa", [25, 50, 75]),
        "rel_course": [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5],
        "tcpa": [0.0] + [float(np.percentile(pos_tcpa, p)) for p in (100 / 3, 200 / 3)],
    }


def bin_of(key: str, value: float, bins: dict) -> int:
    e = bins[key]
    if key == "rel_course":                    # sector 0 wraps around +-180
        return int(np.searchsorted(e, value, side="right")) % 8
    return int(np.searchsorted(e, value, side="right"))


def n_classes(key: str, bins: dict) -> int:
    return 8 if key == "rel_course" else len(bins[key]) + 1


# ── Targets ──────────────────────────────────────────────────────────────────

def token_targets(vessels: list[dict], g: int, bins: dict):
    """Per-cell targets on a g x g grid from crop-frame boxes.

    ``vessels``: [{"box": [x1,y1,x2,y2] in [0,1], state keys...}].
    Returns occ (g*g,), center (g*g,), state (g*g, K) with IGNORE where no
    vessel covers the cell, and weight (g*g,) for the state loss: the covered
    fraction, raised to 1 in the cell holding the box centre so that a vessel
    smaller than a cell still supervises one cell at full weight.
    """
    occ = torch.zeros(g, g)
    center = torch.zeros(g, g)
    best = torch.zeros(g, g)
    state = torch.full((g, g, len(STATE_KEYS)), IGNORE, dtype=torch.long)
    weight = torch.zeros(g, g)
    for v in vessels:
        x1, y1, x2, y2 = [c * g for c in v["box"]]
        for r in range(max(0, int(y1)), min(g, int(np.ceil(y2)))):
            for c in range(max(0, int(x1)), min(g, int(np.ceil(x2)))):
                cov = max(0.0, min(x2, c + 1) - max(x1, c)) * max(0.0, min(y2, r + 1) - max(y1, r))
                if cov <= 0:
                    continue
                occ[r, c] += cov
                if cov > best[r, c]:
                    best[r, c] = cov
                    if "dist_nm" in v:
                        state[r, c] = torch.tensor([bin_of(k, v[k], bins) for k in STATE_KEYS])
                    weight[r, c] = max(weight[r, c].item(), cov)
        cr = min(g - 1, int((y1 + y2) / 2)); cc = min(g - 1, int((x1 + x2) / 2))
        center[cr, cc] = 1.0
        if "dist_nm" in v:
            if best[cr, cc] == 0:
                state[cr, cc] = torch.tensor([bin_of(k, v[k], bins) for k in STATE_KEYS])
            weight[cr, cc] = 1.0
    return (occ.clamp(max=1.0).flatten(), center.flatten(),
            state.view(g * g, -1), weight.flatten())


# ── Data ─────────────────────────────────────────────────────────────────────

def shuffle_state(nav: dict, seed: int) -> dict:
    """Control: the same state values, permuted across all vessels.

    Boxes, and hence occupancy and centre targets, are untouched; only the
    link between a vessel and its own navigation state is broken.
    """
    flat = [(k, i) for k, vs in nav.items() for i in range(len(vs))]
    vals = [{s: nav[k][i][s] for s in STATE_KEYS} for k, i in flat]
    random.Random(seed).shuffle(vals)
    out = {k: [dict(v) for v in vs] for k, vs in nav.items()}
    for (k, i), v in zip(flat, vals):
        out[k][i].update(v)
    return out


def build_items(args, nav: dict) -> tuple[list[dict], Counter]:
    """One item per training image: path, size, vessels in full-image coords."""
    frames = json.load(open(args.frames_json))
    stats = Counter()
    items = []
    for r in frames:
        vs = nav.get(r["id"])
        if vs is None:
            stats["frame_no_nav"] += 1
            continue
        items.append({"path": os.path.join(args.frame_dir, r["image"]),
                      "w": args.image_w, "h": args.image_h, "vessels": vs,
                      "kind": "frame"})
    navbox = {k: vs for k, vs in nav.items()}
    for c in json.load(open(args.crops_json)):
        if c["kind"] == "neg" and not args.absent:
            continue
        # make_sim_crops cuts the window rounded to whole pixels and normalises
        # the boxes by the saved crop size, so invert with the same numbers.
        ox, oy = round(c["window_px"][0]), round(c["window_px"][1])
        cw, ch = c["width"], c["height"]
        vs = []
        for b in c["bboxes"]:
            cx = (ox + (b[0] + b[2]) / 2 * cw) / args.image_w
            cy = (oy + (b[1] + b[3]) / 2 * ch) / args.image_h
            # Centre distance, not IoU: most boxes are a few pixels wide and
            # rounding alone moves their IoU below any fixed threshold.
            dpx = lambda v: np.hypot(((v["box"][0] + v["box"][2]) / 2 - cx) * args.image_w,
                                     ((v["box"][1] + v["box"][3]) / 2 - cy) * args.image_h)
            m = min(navbox.get(c["source_id"], []), key=dpx, default=None)
            if m is None or dpx(m) > args.match_px:
                stats["crop_box_unmatched"] += 1
                vs.append({"box": b})          # occupancy only, no state
            else:
                stats["crop_box_matched"] += 1
                vs.append(dict(m, box=b))
        items.append({"path": os.path.join(args.crop_dir, c["image"]),
                      "w": c["width"], "h": c["height"], "vessels": vs,
                      "kind": c["kind"], "crop_coords": True})
    return items, stats


class TokenSet(Dataset):
    def __init__(self, items, processor, bins, g, absent):
        self.items, self.processor, self.bins, self.g = items, processor, bins, g
        self.absent = absent

    def __len__(self):
        return len(self.items)

    def visible(self, it):
        """Vessels mapped into the encoder's centre crop, box in crop coords."""
        out = []
        for v in it["vessels"]:
            t = transform_boxes([v["box"]], it["w"], it["h"])
            if t:
                out.append(dict(v, box=t[0]))
        return out

    def __getitem__(self, i):
        it = self.items[i]
        try:
            img = Image.open(it["path"]).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=img, return_tensors="pt").pixel_values[0]
        occ, center, state, weight = token_targets(self.visible(it), self.g, self.bins)
        return pv, occ, center, state, weight


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return [torch.stack(x) for x in zip(*batch)]


# ── Model ────────────────────────────────────────────────────────────────────

class ProbeHead(nn.Module):
    """Per-token MLP; discarded after training."""

    def __init__(self, dim, n_state, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.occ = nn.Linear(hidden, 1)
        self.center = nn.Linear(hidden, 1)
        self.state = nn.ModuleList(nn.Linear(hidden, k) for k in n_state)

    def forward(self, tok):
        h = self.trunk(tok)
        return (self.occ(h).squeeze(-1), self.center(h).squeeze(-1),
                [s(h) for s in self.state])


class Wrapped(nn.Module):
    def __init__(self, encoder, head, layer):
        super().__init__()
        self.encoder, self.head, self.layer = encoder, head, layer

    def forward(self, pv):
        hs = self.encoder(pixel_values=pv, output_hidden_states=True).hidden_states
        tok = hs[self.layer][:, 1:, :]           # drop CLS, as the VLM does
        return self.head(tok.float())


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_json", required=True)
    p.add_argument("--nav_json", required=True)
    p.add_argument("--crops_json", required=True)
    p.add_argument("--frame_dir", required=True)
    p.add_argument("--crop_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--channels", default="occ", help="occ or occ,state")
    p.add_argument("--absent", action="store_true")
    p.add_argument("--shuffle_state", action="store_true",
                   help="control: permute navigation state across vessels")
    p.add_argument("--layer", type=int, default=-2)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--per_device_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--center_pos_weight", type=float, default=50.0)
    p.add_argument("--lam_state", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_w", type=int, default=1920)
    p.add_argument("--image_h", type=int, default=1080)
    p.add_argument("--match_px", type=float, default=3.0,
                   help="max centre distance when linking a crop box to its CSV vessel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    chans = set(args.channels.split(","))
    use_state = "state" in chans
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
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    assert_policy_or_die()

    from transformers import CLIPImageProcessor, CLIPVisionModel
    encoder = CLIPVisionModel.from_pretrained(args.vision_model, torch_dtype=torch.float32).to(device)
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    g = encoder.config.image_size // encoder.config.patch_size

    nav = json.load(open(args.nav_json))
    bins = make_bins(nav)               # bins from the true values either way
    if args.shuffle_state:
        nav = shuffle_state(nav, 0)
    items, build_stats = build_items(args, nav)
    if not args.absent:
        # Every image must show a vessel inside the centre crop.
        ds_probe = TokenSet(items, processor, bins, g, args.absent)
        items = [it for it in items if ds_probe.visible(it)]
    dataset = TokenSet(items, processor, bins, g, args.absent)
    kinds = Counter(it["kind"] for it in items)
    n_empty = sum(1 for it in items if not dataset.visible(it))

    torch.manual_seed(args.seed)            # same head init on every rank
    head = ProbeHead(encoder.config.hidden_size,
                     [n_classes(k, bins) for k in STATE_KEYS]).to(device)
    model = Wrapped(encoder, head, args.layer).to(device)
    model.train()
    # Layers above ``--layer`` and post_layernorm get no gradient.
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    core = model.module if distributed else model
    opt = torch.optim.AdamW([
        {"params": core.encoder.parameters(), "lr": args.lr},
        {"params": core.head.parameters(), "lr": args.head_lr}],
        weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.total_steps)

    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                  seed=args.seed) if distributed else None)
    loader = DataLoader(dataset, batch_size=args.per_device_batch, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True, drop_last=True,
                        timeout=900 if args.num_workers > 0 else 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[tok] images {len(dataset):,} {dict(kinds)} vessel-free {n_empty} "
              f"channels={sorted(chans)} absent={args.absent} steps={args.total_steps} "
              f"world={world} batch={world * args.per_device_batch} bins={bins} "
              f"build={dict(build_stats)}", flush=True)

    pw = torch.tensor(args.center_pos_weight, device=device)
    step, epoch, t0 = 0, 0, time.time()
    run = Counter()
    history = []
    while step < args.total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if batch is None:
                continue
            pv, occ_t, cen_t, st_t, w_t = [x.to(device, non_blocking=True) for x in batch]
            occ_p, cen_p, st_p = model(pv)
            l_occ = F.binary_cross_entropy_with_logits(occ_p, occ_t)
            l_cen = F.binary_cross_entropy_with_logits(cen_p, cen_t, pos_weight=pw)
            loss = l_occ + l_cen
            l_st = torch.zeros((), device=device)
            if use_state:
                parts = []
                for k in range(len(STATE_KEYS)):
                    tgt = st_t[..., k]
                    ce = F.cross_entropy(st_p[k].flatten(0, 1), tgt.flatten(),
                                         ignore_index=IGNORE, reduction="none")
                    wk = w_t.flatten() * (tgt.flatten() != IGNORE)
                    parts.append((ce * wk).sum() / wk.sum().clamp(min=1e-6))
                l_st = torch.stack(parts).mean()
                loss = loss + args.lam_state * l_st
            else:
                # Keep the unused state heads in the graph for DDP.
                loss = loss + sum(s.sum() for s in st_p) * 0.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            opt.step(); sched.step(); step += 1
            run.update({"occ": l_occ.item(), "cen": l_cen.item(), "st": l_st.item(), "n": 1})
            if rank == 0 and step % args.log_every == 0:
                n = run["n"]
                rec = {"step": step, "occ": run["occ"] / n, "center": run["cen"] / n,
                       "state": run["st"] / n, "sec": round(time.time() - t0)}
                history.append(rec); run = Counter()
                print(f"  {rec}", flush=True)
            if step >= args.total_steps:
                break
        epoch += 1

    if rank == 0:
        torch.save(core.encoder.state_dict(), os.path.join(args.output_dir, "vision_encoder.bin"))
        torch.save(core.head.state_dict(), os.path.join(args.output_dir, "probe_head.bin"))
        meta = {k: v for k, v in vars(args).items()}
        meta.update({"n_images": len(dataset), "kinds": dict(kinds), "n_vessel_free": n_empty,
                     "bins": bins, "build_stats": dict(build_stats), "epochs_seen": epoch,
                     "world": world, "wall_seconds": round(time.time() - t0, 1),
                     "history": history, "state_keys": STATE_KEYS})
        json.dump(meta, open(os.path.join(args.output_dir, "train_meta.json"), "w"), indent=2)
        print(f"[tok] saved after {step} steps, {meta['wall_seconds']} s", flush=True)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()

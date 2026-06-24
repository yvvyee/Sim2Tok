"""
Geometric attention loss variants for the GAA loss-redesign study.

Motivation (2026-06-19 finding): the original background-L2 loss
(`compute_geometric_loss`) is minimized by *spreading* CLS attention evenly,
not by concentrating it on the vessel.  Empirically it does NOT improve
attention alignment (it slightly worsens it).  These variants instead actively
reward in-box attention so the encoder learns to attend to vessel geometry.

All variants operate on:
  cls_attn : (B, H, F, F)  CLS->patch attention, per head (post-softmax, >=0)
  fg_mask  : (B, F, F)     1 = foreground (inside a GT box), 0 = background

`geo_loss(variant, ...)` dispatches by name.  Lower loss = better alignment for
every variant, so they are drop-in comparable as auxiliary terms.
"""
import torch
import torch.nn.functional as F

from .geometric_loss import compute_geometric_mask

EPS = 1e-8
VARIANTS = ("bg_l2", "inbox_mass", "inbox_ce", "fg_bg_margin")


def extract_cls_attn(attentions: torch.Tensor, feature_size: int) -> torch.Tensor:
    """(B,H,N+1,N+1) -> (B,H,F,F) CLS->patch attention."""
    B, H, _, _ = attentions.shape
    cls = attentions[:, :, 0, 1:]                     # (B, H, N)
    return cls.reshape(B, H, feature_size, feature_size)


def _fg_bg(fg_mask: torch.Tensor):
    """fg_mask (B,F,F) -> (fg, bg) broadcastable to (B,1,F,F)."""
    fg = fg_mask.unsqueeze(1)
    return fg, (1.0 - fg)


def geo_loss(variant: str, cls_attn: torch.Tensor, fg_mask: torch.Tensor,
             margin: float = 0.0) -> torch.Tensor:
    """Dispatch to a geometric attention loss variant. Lower = better aligned."""
    fg, bg = _fg_bg(fg_mask)                           # (B,1,F,F)

    if variant == "bg_l2":
        # Original GAA: penalize squared background attention, normalized by bg area.
        bg_attn = cls_attn * bg
        head_sum = bg_attn.pow(2).sum(dim=(1, 2, 3))   # (B,)
        bg_size = bg.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (head_sum / bg_size).mean()

    if variant == "inbox_mass":
        # Maximize fraction of attention mass inside the boxes.
        in_mass = (cls_attn * fg).sum(dim=(2, 3))      # (B,H)
        total = cls_attn.sum(dim=(2, 3)).clamp(min=EPS)
        frac = (in_mass / total)                       # (B,H) in [0,1]
        return (1.0 - frac).mean()

    if variant == "inbox_ce":
        # Cross-entropy / KL pushing the (renormalized) attention distribution
        # toward the foreground mask distribution.  Spreads over the box rather
        # than collapsing to one patch.
        p = cls_attn.flatten(2)                        # (B,H,N)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=EPS)
        t = fg_mask.flatten(1)                         # (B,N)
        t = t / t.sum(dim=-1, keepdim=True).clamp(min=EPS)
        t = t.unsqueeze(1)                             # (B,1,N) broadcast over heads
        ce = -(t * torch.log(p + EPS)).sum(dim=-1)     # (B,H)
        return ce.mean()

    if variant == "fg_bg_margin":
        # Contrastive: foreground mean attention should exceed background mean
        # by a margin.  softplus keeps it smooth.
        mean_fg = (cls_attn * fg).sum(dim=(2, 3)) / fg.sum(dim=(2, 3)).clamp(min=EPS)
        mean_bg = (cls_attn * bg).sum(dim=(2, 3)) / bg.sum(dim=(2, 3)).clamp(min=EPS)
        return F.softplus(margin - (mean_fg - mean_bg)).mean()

    raise ValueError(f"unknown geo loss variant: {variant!r} (choose from {VARIANTS})")


def compute_geo_loss_from_attn(variant, attentions, bboxes, feature_size=24,
                               tau=0.5, margin=0.0):
    """Convenience wrapper for trainers: extract CLS attention + GT mask, then
    apply the selected geo loss variant.

    Args:
        attentions : (B, H, N+1, N+1) last-layer ViT attention
        bboxes     : list (batch) of list of [x1,y1,x2,y2] normalized boxes
    """
    cls_attn = extract_cls_attn(attentions, feature_size)          # (B,H,F,F)
    fg_mask = compute_geometric_mask(bboxes, feature_size, attentions.device, tau)
    return geo_loss(variant, cls_attn.float(), fg_mask.float(), margin=margin)

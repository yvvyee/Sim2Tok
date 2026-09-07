"""
Map privileged boxes into the coordinate frame the encoder actually sees.

``CLIPImageProcessor`` for ViT-L/14-336 resizes the shortest edge to 336 and
centre-crops to 336x336.  On a 1920x1080 frame that keeps only the horizontal
band ``[0.2188, 0.7812]`` and stretches it across the whole patch grid.

The submitted pipeline normalises boxes against the full frame and lays them on
the grid unchanged, so the target mask and the pixels disagree: measured on the
288-frame evaluation set, 42% of boxes mark regions the encoder cannot see and
the surviving ones sit about five patch columns from where they appear.  See
``crop_audit`` for the per-corpus figures.

This module provides the transform that removes that disagreement.  It is
offered as an option, never as a silent default: the untransformed geometry is
what the submitted numbers were computed with, so changing it everywhere would
make the new numbers incomparable with the paper.  Runs therefore report both,
labelled.

``to_crop_frame`` maps a box into the kept band, clips it to the band, and
returns ``None`` for a box that falls entirely outside — such a box has no
pixels in the input and cannot be a legitimate attention target.
"""
from __future__ import annotations


def kept_band(image_w: int, image_h: int, crop: int = 336) -> tuple[float, float]:
    """Normalised range the centre crop keeps on the longer axis."""
    short = min(image_w, image_h)
    long_scaled = max(image_w, image_h) * (crop / short)
    if long_scaled <= crop:
        return 0.0, 1.0
    lo = ((long_scaled - crop) / 2) / long_scaled
    return lo, 1.0 - lo


def to_crop_frame(box: list[float], image_w: int, image_h: int,
                  crop: int = 336) -> list[float] | None:
    """One box in full-frame normalised coords -> crop-frame normalised coords.

    Returns ``None`` when the box lies entirely outside the kept band.
    """
    lo, hi = kept_band(image_w, image_h, crop)
    span = hi - lo
    if span <= 0:
        return None
    x1, y1, x2, y2 = box

    if image_w >= image_h:                       # wide frame: x is cropped
        if x2 <= lo or x1 >= hi:
            return None
        nx1 = (max(x1, lo) - lo) / span
        nx2 = (min(x2, hi) - lo) / span
        ny1, ny2 = y1, y2
    else:                                        # tall frame: y is cropped
        if y2 <= lo or y1 >= hi:
            return None
        ny1 = (max(y1, lo) - lo) / span
        ny2 = (min(y2, hi) - lo) / span
        nx1, nx2 = x1, x2

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)]


def transform_boxes(boxes: list[list[float]], image_w: int, image_h: int,
                    crop: int = 336) -> list[list[float]]:
    """Every box that survives the crop, in crop-frame coordinates."""
    out = []
    for b in boxes:
        t = to_crop_frame(b, image_w, image_h, crop)
        if t is not None:
            out.append(t)
    return out

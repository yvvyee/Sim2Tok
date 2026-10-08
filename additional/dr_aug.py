"""
Photometric domain randomization for simulator training images.

Applied to the PIL image before the encoder's processor, training images
only.  Geometry is untouched, so boxes and token targets stay valid.  Each
component is drawn independently per image:

* colour jitter (brightness, contrast, saturation 0.6-1.4, hue +-0.1), p=0.8
* grayscale, p=0.2
* Gaussian blur, sigma 0.1-2.0 px, p=0.5
* additive Gaussian noise, std 0-0.06 of the range, p=0.5
* JPEG re-encoding, quality 30-90, p=0.5
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter


_JITTER = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)


def randomize(img: Image.Image, rng: random.Random | None = None) -> Image.Image:
    r = rng or random
    if r.random() < 0.8:
        img = _JITTER(img)
    if r.random() < 0.2:
        img = img.convert("L").convert("RGB")
    if r.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.1, 2.0)))
    if r.random() < 0.5:
        a = np.asarray(img).astype(np.float32) / 255.0
        a = a + np.random.default_rng(r.randrange(1 << 30)).normal(
            0.0, r.uniform(0.0, 0.06), a.shape).astype(np.float32)
        img = Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))
    if r.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=r.randint(30, 90))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img

"""
GAA-aware Dataset and Data Collator

Extends LLaVADataset to load per-sample bounding boxes used as privileged
information for the Geometric Attention Alignment loss.

Expected JSON format (superset of LLaVA format):
  {
    "id": "...",
    "image": "filename.png",
    "conversations": [...],
    "bboxes": [[x1, y1, x2, y2], ...]   ← normalized [0, 1] coords, may be absent
  }

When "bboxes" is absent or empty the sample still trains normally via L_SFT;
only L_geo is skipped for that sample.
"""
from typing import Dict, List
from dataclasses import dataclass

import torch

from data.dataset import LLaVADataset, DataCollatorForVLM


class GAADataset(LLaVADataset):
    """LLaVADataset extended with bounding-box loading for GAA training."""

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        item["bboxes"] = self.data[idx].get("bboxes", [])
        return item


@dataclass
class GAADataCollator(DataCollatorForVLM):
    """
    DataCollatorForVLM extended to pass bounding boxes through the batch.

    Bboxes have variable cardinality per sample and cannot be stacked into
    a single tensor, so they are kept as a plain Python list.

    mask_background: if True, zero out pixel_values OUTSIDE the ground-truth
    boxes (input-level box-guided cropping).  This implements the reviewer's
    "box-guided crop" control baseline — it uses the same privileged box signal
    as GAA but at the input level instead of as an attention loss.
    """

    mask_background: bool = False

    def __call__(self, batch: List[Dict]) -> Dict:
        bboxes = [b.get("bboxes", []) for b in batch]
        # Strip bboxes so the parent collator only sees tensor-compatible fields
        clean = [{k: v for k, v in b.items() if k != "bboxes"} for b in batch]
        collated = super().__call__(clean)
        collated["bboxes"] = bboxes

        if self.mask_background and "pixel_values" in collated:
            pv = collated["pixel_values"]            # (B, 3, H, W)
            H, W = pv.shape[-2], pv.shape[-1]
            for i, boxes in enumerate(bboxes):
                if not boxes:
                    continue                          # no boxes -> leave image intact
                keep = torch.zeros(H, W, dtype=pv.dtype, device=pv.device)
                for b in boxes:
                    x1 = max(0, int(b[0] * W)); y1 = max(0, int(b[1] * H))
                    x2 = min(W, int(b[2] * W)); y2 = min(H, int(b[3] * H))
                    if x2 > x1 and y2 > y1:
                        keep[y1:y2, x1:x2] = 1.0
                pv[i] = pv[i] * keep                  # broadcast over channels
        return collated

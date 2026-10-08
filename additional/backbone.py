"""
Vision backbones for the token trainer, the GAA trainer and the probes.

Three encoders are supported, each with the token layer a LLaVA-style VLM
reads and the pooled attention that GAA supervises:

``openai/clip-vit-large-patch14-336``  24x24 grid, class token, VLM reads
                                       ``hidden_states[-2]`` without the class
                                       token, GAA on the last-layer class-token
                                       attention.
``openai/clip-vit-base-patch16``       14x14 grid at 224 px, same structure.
``google/siglip-so400m-patch14-384``   27x27 grid, no class token.  The image
                                       embedding is formed by an attention-
                                       pooling head (one learned probe query
                                       over the post-layernorm last-layer
                                       tokens), so GAA supervises that pooling
                                       attention.  LLaVA-OneVision reads the
                                       last-layer tokens, ``hidden_states[-1]``.

Box geometry follows each image processor.  CLIP resizes the short side and
centre-crops, so boxes go through ``transform_boxes``.  SigLIP resizes the
whole frame to a square without cropping, so normalised boxes are unchanged
apart from clipping to the frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from additional.box_geometry import transform_boxes


@dataclass(frozen=True)
class Info:
    name: str
    kind: str          # "clip" or "siglip"
    g: int             # patch grid side
    size: int          # input side in pixels
    hidden: int
    vlm_layer: int     # hidden_states index a LLaVA-style VLM reads


def kind_of(name: str) -> str:
    return "siglip" if "siglip" in name.lower() else "clip"


def load(name: str, dtype=torch.float32, eager: bool = False):
    """Encoder, image processor and ``Info`` for a backbone name."""
    attn = {"attn_implementation": "eager"} if eager else {}
    if kind_of(name) == "siglip":
        from transformers import SiglipImageProcessor, SiglipVisionModel
        enc = SiglipVisionModel.from_pretrained(name, torch_dtype=dtype, **attn)
        proc = SiglipImageProcessor.from_pretrained(name)
        layer = -1
    else:
        from transformers import CLIPImageProcessor, CLIPVisionModel
        enc = CLIPVisionModel.from_pretrained(name, torch_dtype=dtype, **attn)
        proc = CLIPImageProcessor.from_pretrained(name)
        layer = -2
    c = enc.config
    info = Info(name, kind_of(name), c.image_size // c.patch_size, c.image_size,
                c.hidden_size, layer)
    return enc, proc, info


def patch_tokens(info: Info, hidden_states, layer: int) -> torch.Tensor:
    """(B, g*g, D) patch tokens of ``hidden_states[layer]``, class token removed."""
    h = hidden_states[layer]
    return h[:, 1:, :] if info.kind == "clip" else h


def map_boxes(info: Info, boxes, w: int, h: int) -> list[list[float]]:
    """Full-image normalised boxes -> normalised boxes on the encoder input."""
    if info.kind == "clip":
        return transform_boxes(boxes, w, h, crop=info.size)
    out = []
    for b in boxes:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in b)
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def pooled_attention(info: Info, encoder, out, layer_index: int = -1) -> torch.Tensor:
    """(B, H, g, g) attention of the pooled query over the patch tokens.

    CLIP: the class-token row of ``out.attentions[layer_index]`` (requires
    ``output_attentions=True`` and eager attention).  ``layer_index=-2`` gives
    the layer-23 class token of ViT-L/14.

    SigLIP: the attention-pooling head's probe query over the post-layernorm
    last-layer tokens, recomputed here so that it carries a gradient.
    """
    if info.kind == "clip":
        a = out.attentions[layer_index]                 # (B, H, N+1, N+1)
        cls = a[:, :, 0, 1:]
        return cls.reshape(cls.shape[0], cls.shape[1], info.g, info.g)
    if layer_index != -1:
        raise ValueError("SigLIP has one pooling attention; layer_index must be -1")
    vm = encoder.vision_model if hasattr(encoder, "vision_model") else encoder
    head = vm.head
    x = out.last_hidden_state                            # post-layernorm tokens
    probe = head.probe.repeat(x.shape[0], 1, 1)
    _, w = head.attention(probe, x, x, need_weights=True, average_attn_weights=False)
    w = w[:, :, 0, :]                                    # (B, H, N)
    return w.reshape(w.shape[0], w.shape[1], info.g, info.g)

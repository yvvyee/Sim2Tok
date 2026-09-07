"""
Shared loading for the submitted VLM and for encoder swapping.

Both the absence-probe and the COLREGs measurement need the same thing: the
vision-language model exactly as the submitted work assembled it (CLIP tower,
trained projector, LoRA-adapted Llama), with the ability to substitute the
vision encoder and nothing else.  Keeping that in one place means the two
measurements cannot drift apart in how they build the model.

The encoder swap refuses to be a silent no-op.  A checkpoint whose keys match
nothing would leave the stock tower in place, and the run would then report a
"GAA" row that never saw GAA weights, which is worse than an error.
"""
from __future__ import annotations

import os

import torch


def build_vlm(vlm_ckpt: str, llm: str, vision_model: str,
              projector_type: str, dtype, device):
    """The submitted VLM: CLIP tower, trained projector, LoRA-adapted LLM.

    ``vlm_ckpt`` is a directory holding ``projector.bin`` and a PEFT adapter,
    which is the layout ``train.py`` writes.
    """
    from peft import PeftModel
    from model import VLMConfig, build_model

    cfg = VLMConfig(
        vision_model_name=vision_model,
        llm_model_name=llm,
        projector_type=projector_type,
        vision_feature_layer=-2,
        vision_feature_select_strategy="patch",
        freeze_vision=True,
        freeze_llm=True,
    )
    model = build_model(cfg, torch_dtype=dtype)

    proj = os.path.join(vlm_ckpt, "projector.bin")
    if not os.path.exists(proj):
        raise FileNotFoundError(f"projector.bin missing under {vlm_ckpt}")
    model.projector.load_weights(
        torch.load(proj, map_location="cpu", weights_only=True))

    if not os.path.exists(os.path.join(vlm_ckpt, "adapter_config.json")):
        raise FileNotFoundError(f"no PEFT adapter under {vlm_ckpt}")
    model.language_model = PeftModel.from_pretrained(
        model.language_model, vlm_ckpt, is_trainable=False)

    return model.to(device).eval()


def swap_encoder(model, ckpt: str | None) -> dict:
    """Overlay vision-encoder weights on the frozen tower.

    Returns a load report.  ``None`` or ``"none"`` keeps whatever encoder the
    checkpoint directory already loaded, which is the stock-encoder condition.
    """
    if not ckpt or ckpt.lower() in ("none", ""):
        return {"ckpt": None, "matched": 0, "total": 0}
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"encoder checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    missing, unexpected = model.vision_encoder.model.load_state_dict(
        state, strict=False)
    matched = len(state) - len(unexpected)
    if matched == 0:
        raise RuntimeError(
            f"no vision weights matched from {ckpt}; the run would report a "
            f"swapped encoder while measuring the stock one")
    return {"ckpt": ckpt, "matched": matched, "total": len(state),
            "missing": len(missing), "unexpected": len(unexpected)}


def uniform_prompt(tokenizer, question: str, with_image_token: bool):
    """Token ids for one question, identical for every image in a batch.

    Batching generation needs equal-length sequences.  Rather than pad, the
    callers group probes so that one prompt serves a whole batch and only the
    image varies, so no padding semantics enter the measurement at all.

    ``with_image_token`` controls whether the prompt carries ``<image>``.  The
    model splices the encoder's output at that token; without it the features
    never reach the language model.  See ``halluc_probe`` for why this is a
    switch rather than a constant.
    """
    q = f"<image>\n{question}" if with_image_token else question
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False,
                     return_tensors="pt").input_ids

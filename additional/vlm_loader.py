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


def snapshot_encoder(model) -> dict:
    """A CPU copy of the tower's weights, to restore the stock condition.

    ``build_vlm`` is called once and every ``--encoder`` pass mutates that one
    model in place, so the "stock" condition is only stock if the weights are
    put back.  Without this, ``--encoder gaa=... --encoder stock=none`` would
    measure the GAA tower twice and label one of the rows stock, and the two
    rows would agree perfectly -- which a reader would take as evidence that
    the encoder cannot influence the answer at all.
    """
    return {k: v.detach().cpu().clone()
            for k, v in model.vision_encoder.model.state_dict().items()}


def swap_encoder(model, ckpt: str | None, stock_state: dict | None = None,
                 min_match_ratio: float = 0.99) -> dict:
    """Set the frozen tower's weights for one measurement condition.

    ``None`` or ``"none"`` restores ``stock_state``, which is the stock
    condition.  Passing no snapshot for that case is refused rather than
    silently leaving a previous swap in place.

    A checkpoint that matches only part of the tower is refused too.  Checking
    only for zero matches would let a truncated or partial-layer save through,
    and the run would then report a number for an encoder that is mostly stock
    weights under a label saying otherwise.
    """
    target = model.vision_encoder.model

    if not ckpt or ckpt.lower() in ("none", ""):
        if stock_state is None:
            raise ValueError(
                "the stock condition needs a snapshot taken before any swap; "
                "call snapshot_encoder(model) once and pass it in")
        target.load_state_dict(stock_state, strict=True)
        return {"ckpt": None, "restored_stock": True,
                "matched": len(stock_state), "total": len(stock_state)}

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"encoder checkpoint not found: {ckpt}")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    missing, unexpected = target.load_state_dict(state, strict=False)
    matched = len(state) - len(unexpected)
    n_target = len(target.state_dict())
    if matched < min_match_ratio * n_target:
        raise RuntimeError(
            f"{ckpt} set only {matched} of the tower's {n_target} tensors "
            f"({matched / max(n_target, 1):.1%}); the run would report this as "
            f"a tuned encoder while most weights stayed stock. "
            f"missing={len(missing)} unexpected={len(unexpected)}")
    return {"ckpt": ckpt, "restored_stock": False, "matched": matched,
            "total": len(state), "target_tensors": n_target,
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

"""
Reasoning-preservation check for the decoupled transfer recipe.

Hypothesis: train the CLIP vision encoder with geo loss only (decoupled from
synthetic SFT) on mixed sim+real data for attention transfer, while keeping the
existing baseline LoRA + projector for nautical reasoning.  This script verifies
that swapping in the geo-trained vision encoder does NOT destroy the VLM's
text reasoning, by generating navigation advice on held-out SDS samples with
the stock encoder vs. a supplied geo-trained vision_encoder.bin.

Usage:
  python gaa/eval_reasoning.py \
      --baseline_ckpt /home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en \
      --vision_encoder_ckpt checkpoints/geo_encoders/inbox_ce_vision_encoder.bin \
      --data_path data/sds_gaa_en_test20.json \
      --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
      --num_samples 3
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import textwrap

import torch
from PIL import Image
from peft import PeftModel

from model import VLMConfig, build_model


def parse_args():
    p = argparse.ArgumentParser("GAA reasoning-preservation check")
    p.add_argument("--baseline_ckpt", required=True, help="LoRA+projector checkpoint dir")
    p.add_argument("--vision_encoder_ckpt", default=None,
                   help="geo-trained vision_encoder.bin to swap in (None = stock CLIP)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--llm_model", default="/home/ywlee/SSD/Llama-3.1-8B-Instruct")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_model(args, dtype, device, swap_encoder):
    config = VLMConfig(
        vision_model_name=args.vision_model, llm_model_name=args.llm_model,
        projector_type="mlp2x_gelu", vision_feature_layer=-2,
        vision_feature_select_strategy="patch", freeze_vision=True, freeze_llm=True,
    )
    model = build_model(config, torch_dtype=dtype)
    proj = os.path.join(args.baseline_ckpt, "projector.bin")
    if os.path.exists(proj):
        model.projector.load_weights(torch.load(proj, map_location="cpu", weights_only=True))
    model.language_model = PeftModel.from_pretrained(
        model.language_model, args.baseline_ckpt, is_trainable=False)
    if swap_encoder and args.vision_encoder_ckpt:
        state = torch.load(args.vision_encoder_ckpt, map_location="cpu", weights_only=True)
        if state and all(k.startswith("model.") for k in state):
            state = {k[len("model."):]: v for k, v in state.items()}
        missing, unexpected = model.vision_encoder.model.load_state_dict(state, strict=False)
        print(f"  [swap] loaded geo encoder (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


@torch.no_grad()
def generate(model, human_msg, image_path, max_new_tokens, device):
    tok = model.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": human_msg}],
                                   tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (336, 336), 0)
    pv = model.vision_encoder.image_processor(images=image, return_tensors="pt").pixel_values
    pv = pv.to(device=device, dtype=model.vision_encoder.dtype)
    out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         pixel_values=pv, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def wrap(t, w=90):
    return "\n".join(textwrap.fill(l, w, initial_indent="    ", subsequent_indent="    ")
                     for l in t.splitlines())


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = json.load(open(args.data_path))[:args.num_samples]

    outputs = {}
    for label, swap in [("baseline-encoder", False), ("geo-encoder", True)]:
        if swap and not args.vision_encoder_ckpt:
            continue
        print(f"\nLoading {label} ...")
        model = load_model(args, dtype, device, swap)
        outs = []
        for s in samples:
            outs.append(generate(model, s["conversations"][0]["value"],
                                 os.path.join(args.image_dir, s["image"]),
                                 args.max_new_tokens, device))
        outputs[label] = outs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*100}\n  REASONING PRESERVATION CHECK ({len(samples)} held-out SDS samples)\n{'='*100}")
    for i, s in enumerate(samples):
        print(f"\n[{i+1}] {s['id']}")
        print("  [Ground Truth]"); print(wrap(s["conversations"][1]["value"][:400]))
        for label in outputs:
            print(f"  [{label}]"); print(wrap(outputs[label][i][:400]))
        print("-" * 100)


if __name__ == "__main__":
    main()

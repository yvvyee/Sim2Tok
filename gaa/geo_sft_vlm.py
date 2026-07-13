"""
geo_sft_vlm.py — 2-stage recipe: geo 인코더 동결 + SFT projector/LLM 재정렬

각 VLM에 대해:
  1. 전체 VLM 로드
  2. geo 학습된 비전 인코더 swap + 동결
  3. LLM에 LoRA 적용
  4. SDS SFT 데이터(80샘플)로 projector/LoRA 재정렬
  5. 저장 → COLREG 추론 정확도 평가

Usage:
  python gaa/geo_sft_vlm.py --model smolvlm --num_epochs 3
  python gaa/geo_sft_vlm.py --all --num_epochs 3
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import gc
import argparse
import glob

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, TaskType

from transformers import AutoModelForImageTextToText, AutoProcessor
from gaa.geo_validate_vlms import REGISTRY, get_vision_tower
from gaa.vlm_behavior_eval import gt_situation, gt_vessel_count, yes, parse_sit
from gaa import vlm_zoo

GEO_CKPT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
SFT_DATA   = "/home/ywlee/dev/gaa/data/sds_gaa_en_train80.json"
SFT_IMGDIR = "/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227"
SDS_EVAL   = "/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227"
HF_HOME    = "/home/ywlee/SSD/hf_cache"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_sft_samples(data_path=SFT_DATA):
    return json.load(open(data_path))


def build_input(proc, sample, imgdir, device, dtype, model_name):
    """Build model inputs for one SFT sample.
    Returns dict with pixel_values, input_ids, attention_mask, labels.
    Labels mask the user/system prompt (only train on assistant tokens).
    """
    img_path = os.path.join(imgdir, sample["image"])
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    user_msg  = sample["conversations"][0]["value"]
    asst_msg  = sample["conversations"][1]["value"]

    # Remove <image> token from user message for template (some processors add it)
    user_clean = user_msg.replace("<image>", "").strip()

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_clean},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": asst_msg}]},
    ]

    # Try chat template tokenization
    try:
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt",
            images=[image])
    except TypeError:
        try:
            text = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=False)
            inputs = proc(text=text, images=[image], return_tensors="pt")
        except Exception:
            return None
    except Exception:
        return None

    if "input_ids" not in inputs:
        return None

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None
    if hasattr(pixel_values, "to"):
        pixel_values = pixel_values.to(device=device, dtype=dtype)

    # Build labels: -100 for prompt, token ids for response
    # Find assistant response start by re-tokenizing just the prompt
    messages_prompt = messages[:1]
    try:
        prompt_inputs = proc.apply_chat_template(
            messages_prompt, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            images=[image])
        prompt_len = prompt_inputs["input_ids"].shape[1]
    except Exception:
        # Fallback: train on last 30% of tokens
        prompt_len = int(input_ids.shape[1] * 0.7)

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    # Also forward other processor outputs (image_grid_thw, etc.)
    extra = {k: v.to(device) if hasattr(v, "to") else v
             for k, v in inputs.items()
             if k not in ("input_ids", "attention_mask", "pixel_values")}

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
        **extra,
    }


# ── Vision encoder swap + freeze ─────────────────────────────────────────────

def load_and_freeze_geo_encoder(model, model_name: str) -> bool:
    geo_ckpt = os.path.join(GEO_CKPT_BASE, f"geo_{model_name}", "vision_encoder.bin")
    family = REGISTRY[model_name][1]
    try:
        enc = get_vision_tower(model, family)
    except Exception as e:
        print(f"  [WARN] vision tower not found: {e}")
        return False
    if not os.path.exists(geo_ckpt):
        print(f"  [WARN] geo ckpt missing: {geo_ckpt}")
        return False
    state = torch.load(geo_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    n = len(state) - len(unexpected)
    print(f"  [geo swap] {n}/{len(state)} tensors loaded")
    # Freeze vision encoder
    for p in enc.parameters():
        p.requires_grad = False
    print(f"  [freeze] vision encoder frozen ({sum(p.numel() for p in enc.parameters()):,} params)")
    return True


# ── LoRA setup ────────────────────────────────────────────────────────────────

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # some VLMs use these names
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
]


def apply_lora(model, lora_r=32, lora_alpha=64):
    # Freeze entire model first
    for p in model.parameters():
        p.requires_grad = False

    # Find which LoRA targets exist
    named = {n for n, _ in model.named_modules()}
    targets = [t for t in LORA_TARGET_MODULES
               if any(t in n for n in named)]
    if not targets:
        targets = ["q_proj", "v_proj"]

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=targets, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train_sft(model, proc, model_name, samples, num_epochs, lr, device, dtype):
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.01)
    total_steps = num_epochs * len(samples)
    sched = CosineAnnealingLR(opt, T_max=total_steps)

    model.train()
    for epoch in range(num_epochs):
        total_loss, n_ok = 0.0, 0
        for s in samples:
            inputs = build_input(proc, s, SFT_IMGDIR, device, dtype, model_name)
            if inputs is None:
                continue
            try:
                out = model(**inputs)
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                total_loss += loss.item()
                n_ok += 1
            except Exception as e:
                continue
        avg = total_loss / max(n_ok, 1)
        print(f"  [epoch {epoch+1}/{num_epochs}]  loss={avg:.4f}  n={n_ok}")
    model.eval()


# ── Reasoning evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_colreg(model, proc, num_eps=100) -> dict:
    eps = sorted(d for d in glob.glob(os.path.join(SDS_EVAL, "*"))
                 if os.path.isdir(d))[:num_eps]
    rec, hal, sit_ok, sit_tot = [], [], 0, 0
    for ep in eps:
        img_p = os.path.join(ep, "input_image.png")
        if not os.path.exists(img_p):
            continue
        img = Image.open(img_p).convert("RGB")
        gt_sit = gt_situation(ep)
        gt_n   = gt_vessel_count(ep)

        a_v = vlm_zoo.generate(model, proc, img,
            "Is there a vessel or ship in this image? Answer yes or no.", 16)
        if gt_n is None or gt_n > 0:
            rec.append(1.0 if yes(a_v) else 0.0)
        for obj in ("airplane", "car", "building"):
            a = vlm_zoo.generate(model, proc, img,
                f"Is there a {obj} in this image? Answer yes or no.", 16)
            hal.append(1.0 if yes(a) else 0.0)
        a_s = vlm_zoo.generate(model, proc, img,
            "Look at the vessels. Classify the COLREG encounter situation as exactly "
            "one word: head-on, crossing, overtaking, or safe.", 24)
        ps = parse_sit(a_s)
        sit_tot += 1
        sit_ok  += int(ps == gt_sit)

    def m(lst): return round(sum(lst) / max(len(lst), 1), 4)
    return {
        "vessel_recall": m(rec),
        "halluc_rate":   m(hal),
        "colreg_acc":    round(sit_ok / max(sit_tot, 1), 4),
        "n": sit_tot,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, num_epochs, num_eval, out_dir, lora_r):
    print(f"\n{'='*60}")
    print(f"  2-stage SFT: [{model_name}]")
    print(f"  epochs={num_epochs}  lora_r={lora_r}  eval_eps={num_eval}")
    print(f"{'='*60}")

    os.environ.setdefault("HF_HOME", HF_HOME)
    repo = REGISTRY[model_name][0]
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    # Load model + processor
    print("  모델 로딩 ...")
    for local_only in [False, True]:
        try:
            proc  = AutoProcessor.from_pretrained(repo, local_files_only=local_only)
            model = AutoModelForImageTextToText.from_pretrained(
                repo, torch_dtype=dtype, device_map="cuda",
                local_files_only=local_only).eval()
            break
        except Exception as e:
            if local_only:
                print(f"  [ERROR] {e}"); return None

    # Stage 1: swap geo encoder + freeze
    swapped = load_and_freeze_geo_encoder(model, model_name)
    if not swapped:
        print("  [WARN] geo swap 실패 — stock encoder 사용")

    # Baseline eval (before SFT, after geo swap)
    print("\n  [eval] Before SFT (geo encoder, no projector align) ...")
    before = eval_colreg(model, proc, num_eval)
    print(f"    colreg={before['colreg_acc']:.4f}  recall={before['vessel_recall']:.4f}")

    # Stage 2: LoRA + SFT
    print("\n  [SFT] projector/LoRA 재정렬 ...")
    model = apply_lora(model, lora_r=lora_r, lora_alpha=lora_r * 2)
    samples = load_sft_samples()
    train_sft(model, proc, model_name, samples, num_epochs, lr=2e-4, device=device, dtype=dtype)

    # After eval
    print("\n  [eval] After SFT ...")
    after = eval_colreg(model, proc, num_eval)
    print(f"    colreg={after['colreg_acc']:.4f}  recall={after['vessel_recall']:.4f}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    adapter_dir = os.path.join(out_dir, f"sft_{model_name}")
    try:
        model.save_pretrained(adapter_dir)
        proc.save_pretrained(adapter_dir)
        print(f"  adapter 저장: {adapter_dir}")
    except Exception as e:
        print(f"  [WARN] save failed: {e}")

    result = {
        "model": model_name, "swapped": swapped,
        "before_sft": before, "after_sft": after,
        "delta_colreg": round(after["colreg_acc"] - before["colreg_acc"], 4),
        "delta_from_baseline": None,  # filled after reading geo_reason results
    }
    # Compare with original baseline from geo_reason_eval
    geo_reason_path = os.path.join(
        os.path.dirname(out_dir), "geo_reason", f"{model_name}.json")
    if os.path.exists(geo_reason_path):
        orig = json.load(open(geo_reason_path))
        baseline_colreg = orig["baseline"]["colreg_acc"]
        result["original_baseline_colreg"] = baseline_colreg
        result["delta_from_baseline"] = round(
            after["colreg_acc"] - baseline_colreg, 4)

    result_path = os.path.join(out_dir, f"{model_name}.json")
    json.dump(result, open(result_path, "w"), indent=2, ensure_ascii=False)
    print(f"  결과 저장: {result_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser("2-stage SFT for geo-trained VLMs")
    p.add_argument("--model", default=None, choices=list(REGISTRY.keys()))
    p.add_argument("--all",   action="store_true")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lora_r",     type=int, default=32)
    p.add_argument("--num_eval",   type=int, default=100)
    p.add_argument("--out_dir", default="results/geo_sft")
    args = p.parse_args()

    SKIP = {"llama-3.2"}
    if args.all:
        models = [m for m in REGISTRY if m not in SKIP]
    elif args.model:
        models = [args.model]
    else:
        p.error("--model 또는 --all 지정 필요")

    all_results = {}
    for name in models:
        r = run_model(name, args.num_epochs, args.num_eval, args.out_dir, args.lora_r)
        if r:
            all_results[name] = r

    # Summary
    print(f"\n{'='*72}")
    print("  2-stage SFT 결과 요약")
    print(f"{'='*72}")
    hdr = f"{'model':<14}{'orig_base':>10}{'before_sft':>12}{'after_sft':>11}{'Δ(sft)':>8}{'Δ(vs_orig)':>11}"
    print(hdr); print("-"*len(hdr))
    for name, r in all_results.items():
        ob = r.get("original_baseline_colreg", float("nan"))
        bf = r["before_sft"]["colreg_acc"]
        af = r["after_sft"]["colreg_acc"]
        ds = r["delta_colreg"]
        do = r.get("delta_from_baseline") or float("nan")
        print(f"{name:<14}{ob:>10.3f}{bf:>12.3f}{af:>11.3f}{ds:>+8.3f}{do:>+11.3f}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    json.dump(all_results, open(summary_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n전체 결과: {summary_path}")


if __name__ == "__main__":
    main()

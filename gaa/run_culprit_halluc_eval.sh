#!/bin/bash
# Visual Culprit Hypothesis 검증
# Low/High Attn-IoU 그룹으로 학습한 geo 인코더 → halluc_rate 비교
set -e

PY=/home/ywlee/miniforge3/envs/eva/bin/python
WORKDIR=/home/ywlee/share/PaperWork/GAA/GAA
LLM=/home/ywlee/SSD/Llama-3.1-8B-Instruct
CTRL=/home/ywlee/SSD/checkpoints/ctrl_baseline_fix
SDS_EVAL=/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227
LOG=$WORKDIR/logs/culprit_halluc_eval.log
CKPT_BASE=$WORKDIR/checkpoints/culprit_exp

mkdir -p $WORKDIR/logs
cd $WORKDIR

echo "=== Visual Culprit Hallucination Evaluation ===" | tee $LOG
echo "시작: $(date)" | tee -a $LOG

$PY - << 'PYEOF' 2>&1 | tee -a $LOG
import sys, os, json, glob, torch
sys.path.insert(0, '.')
from PIL import Image
from peft import PeftModel
from model import VLMConfig, build_model

SDS_EVAL  = '/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227'
CTRL_CKPT = '/home/ywlee/SSD/checkpoints/ctrl_baseline_fix'
LLM_MODEL = '/home/ywlee/SSD/Llama-3.1-8B-Instruct'
CKPT_BASE = '/home/ywlee/share/PaperWork/GAA/GAA/checkpoints/culprit_exp'
NUM_EPS   = 100
ABSENT    = ['airplane', 'car', 'building']

ENCODERS = {
    'stock_CLIP':    None,
    'geo_full':      '/home/ywlee/SSD/checkpoints/geo_encoders_fix/inbox_ce_vision_encoder.bin',
    'geo_high_attn': f'{CKPT_BASE}/geo_encoder_high.bin',
    'geo_low_attn':  f'{CKPT_BASE}/geo_encoder_low.bin',
}

device = torch.device('cuda')
dtype  = torch.bfloat16

def load_model(enc_ckpt):
    cfg = VLMConfig(
        vision_model_name='openai/clip-vit-large-patch14-336',
        llm_model_name=LLM_MODEL,
        projector_type='mlp2x_gelu',
        vision_feature_layer=-2,
        vision_feature_select_strategy='patch',
        freeze_vision=True, freeze_llm=True,
    )
    model = build_model(cfg, torch_dtype=dtype)
    proj = os.path.join(CTRL_CKPT, 'projector.bin')
    if os.path.exists(proj):
        model.projector.load_weights(torch.load(proj, map_location='cpu', weights_only=True))
    model.language_model = PeftModel.from_pretrained(
        model.language_model, CTRL_CKPT, is_trainable=False)
    if enc_ckpt and os.path.exists(enc_ckpt):
        state = torch.load(enc_ckpt, map_location='cpu', weights_only=True)
        if state and all(k.startswith('model.') for k in state):
            state = {k[6:]: v for k, v in state.items()}
        model.vision_encoder.model.load_state_dict(state, strict=False)
        print(f"  encoder loaded: {os.path.basename(enc_ckpt)}")
    return model.to(device).eval()

@torch.no_grad()
def ask(model, image, question, max_new=16):
    tok = model.tokenizer
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': question}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    pv  = model.vision_encoder.image_processor(images=image, return_tensors='pt').pixel_values
    pv  = pv.to(device=device, dtype=dtype)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         pixel_values=pv, max_new_tokens=max_new,
                         do_sample=False, temperature=None, top_p=None)
    resp = tok.decode(out[0], skip_special_tokens=True).strip().lower()
    return resp.startswith('yes') or ' yes' in resp[:12]

eps = sorted(d for d in glob.glob(f'{SDS_EVAL}/*') if os.path.isdir(d))[:NUM_EPS]
imgs = [Image.open(os.path.join(ep, 'input_image.png')).convert('RGB')
        for ep in eps if os.path.exists(os.path.join(ep, 'input_image.png'))]
print(f"평가 이미지: {len(imgs)}장")

print(f"\n{'encoder':<20} {'halluc_rate':>12} {'n_halluc':>10}")
print('-' * 45)
results = {}
for name, ckpt in ENCODERS.items():
    if ckpt and not os.path.exists(ckpt):
        print(f"{name:<20}  MISSING: {ckpt}"); continue
    model = load_model(ckpt)
    halluc = []
    for img in imgs:
        for obj in ABSENT:
            halluc.append(1.0 if ask(model, img, f'Is there a {obj} in this image? Answer yes or no.') else 0.0)
    rate = sum(halluc) / max(len(halluc), 1)
    n_h  = int(sum(halluc))
    print(f"{name:<20} {rate:>12.4f} {n_h:>10d}")
    results[name] = {'halluc_rate': round(rate, 4), 'n_halluc': n_h, 'n_total': len(halluc)}
    del model; torch.cuda.empty_cache()

import json
out_path = '/home/ywlee/share/PaperWork/GAA/GAA/results/culprit_halluc.json'
json.dump(results, open(out_path, 'w'), indent=2)
print(f"\n저장: {out_path}")

# Delta analysis
if 'geo_low_attn' in results and 'geo_high_attn' in results:
    low  = results['geo_low_attn']['halluc_rate']
    high = results['geo_high_attn']['halluc_rate']
    stock = results.get('stock_CLIP', {}).get('halluc_rate', float('nan'))
    print(f"\n[핵심 비교]")
    print(f"  stock CLIP:      {stock:.4f}")
    print(f"  geo_high_attn:   {high:.4f}  (Attn-IoU 높은 예시로 학습)")
    print(f"  geo_low_attn:    {low:.4f}  (Attn-IoU 낮은 예시로 학습)")
    print(f"  Delta (low-high): {low-high:+.4f}")
    if low > high:
        print("  → 가설 지지: Low Attn-IoU 예시가 더 높은 환각률을 유발")
    else:
        print("  → 가설 미지지")
PYEOF

echo "완료: $(date)" | tee -a $LOG

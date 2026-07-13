#!/bin/bash
# 2단계 레시피: geo 선학습 인코더 동결 → SFT 재정렬
#
# Stage 1 (완료): vision-only geo 학습 → inbox_ce_vision_encoder.bin (Attn-IoU 0.44)
# Stage 2 (본 스크립트): 인코더 동결 + SFT로 projector/LoRA 재정렬
#
# 실험 조건:
#   stage2_syn   : 동결된 geo 인코더 + SFT (합성 80샘플) ← 핵심 2단계
#   [비교] ctrl_baseline_train80 : 기존 stock CLIP + SFT (이미 완료)
#
# 평가:
#   A) Attn-IoU on SMD onshore 288 (정렬 유지 여부)
#   B) eval_reasoning: test20 질적 추론 비교
set -e

WORKDIR=/home/ywlee/share/PaperWork/GAA/GAA
PY=/home/ywlee/miniforge3/envs/eva/bin/python
LOG=$WORKDIR/logs/stage2.log

# 경로 상수
GEO_ENCODER=/home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel/checkpoints/geo_encoders_fix/inbox_ce_vision_encoder.bin
SFT_DATA=/home/ywlee/dev/gaa/data/sds_gaa_en_train80.json
SFT_IMGDIR=/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227
TEST_DATA=/home/ywlee/dev/gaa/data/sds_gaa_en_test20.json
SMD_DATA=/home/ywlee/dev/gaa/data/smd_onshore_big.json
SMD_IMGDIR=/home/ywlee/SSD/smd_onshore_big
CTRL_CKPT=/home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel/checkpoints/ctrl_baseline_train80
OUT=$WORKDIR/checkpoints/stage2_syn
LLM_MODEL=/home/ywlee/SSD/Llama-3.1-8B-Instruct

mkdir -p $WORKDIR/logs $WORKDIR/checkpoints

cd $WORKDIR

echo "========================================" | tee -a $LOG
echo "STAGE 2: geo 인코더 동결 + SFT 재정렬" | tee -a $LOG
echo "시작: $(date)" | tee -a $LOG
echo "GEO_ENCODER: $GEO_ENCODER" | tee -a $LOG
echo "SFT_DATA: $SFT_DATA ($(python3 -c "import json; print(len(json.load(open('$SFT_DATA'))))")샘플)" | tee -a $LOG
echo "========================================" | tee -a $LOG

# ── Stage 2A: 동결된 geo 인코더 + SFT ──────────────────────────────────────
echo "" | tee -a $LOG
echo "===== [2A] SFT 재정렬 (frozen geo encoder) =====" | tee -a $LOG
$PY gaa/train_gaa.py \
    --llm_model         $LLM_MODEL \
    --init_vision_ckpt  $GEO_ENCODER \
    --freeze_vision \
    --data_path         $SFT_DATA \
    --image_dir         $SFT_IMGDIR \
    --output_dir        $OUT \
    --geo_variant       inbox_ce \
    --num_epochs        5.0 \
    --batch_size        4 \
    --grad_accum        4 \
    --learning_rate     2e-4 \
    --warmup_ratio      0.05 \
    --lora_r            128 \
    --lora_alpha        256 \
    --gradient_checkpointing \
    2>&1 | tee -a $LOG

echo "학습 완료: $(date)" | tee -a $LOG

# ── Eval A: Attn-IoU (SMD onshore) ─────────────────────────────────────────
echo "" | tee -a $LOG
echo "===== [EVAL-A] Attn-IoU on SMD onshore (정렬 유지 여부) =====" | tee -a $LOG

echo "## stock CLIP (기준)" | tee -a $LOG
$PY gaa/evaluate_attn.py \
    --data_path  $SMD_DATA \
    --image_dir  $SMD_IMGDIR \
    2>&1 | tee -a $LOG

echo "## ctrl_baseline_train80 (stock CLIP + SFT)" | tee -a $LOG
$PY gaa/evaluate_attn.py \
    --data_path     $SMD_DATA \
    --image_dir     $SMD_IMGDIR \
    --gaa_ckpt      $CTRL_CKPT/vision_encoder.bin \
    2>&1 | tee -a $LOG

echo "## stage2_syn (geo 인코더 동결 + SFT)" | tee -a $LOG
$PY gaa/evaluate_attn.py \
    --data_path     $SMD_DATA \
    --image_dir     $SMD_IMGDIR \
    --gaa_ckpt      $OUT/vision_encoder.bin \
    2>&1 | tee -a $LOG

# ── Eval B: 추론 품질 (test20 질적 비교) ────────────────────────────────────
echo "" | tee -a $LOG
echo "===== [EVAL-B] 추론 품질 비교 (test20, 5샘플) =====" | tee -a $LOG

echo "## ctrl_baseline_train80 (stock CLIP + SFT)" | tee -a $LOG
$PY gaa/eval_reasoning.py \
    --llm_model     $LLM_MODEL \
    --baseline_ckpt $CTRL_CKPT \
    --data_path     $TEST_DATA \
    --image_dir     $SFT_IMGDIR \
    --num_samples   5 \
    2>&1 | tee -a $LOG

echo "## stage2_syn (geo 인코더 동결 + SFT) + geo 인코더 적용" | tee -a $LOG
$PY gaa/eval_reasoning.py \
    --llm_model           $LLM_MODEL \
    --baseline_ckpt       $OUT \
    --vision_encoder_ckpt $OUT/vision_encoder.bin \
    --data_path           $TEST_DATA \
    --image_dir           $SFT_IMGDIR \
    --num_samples         5 \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "STAGE2DONE $(date)" | tee -a $LOG

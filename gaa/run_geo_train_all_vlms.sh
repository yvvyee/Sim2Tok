#!/bin/bash
# 11개 VLM 비전 인코더 geo 학습

PY=/home/ywlee/miniforge3/envs/eva/bin/python
WORKDIR=/home/ywlee/share/PaperWork/GAA/GAA
CKPT_BASE=$WORKDIR/checkpoints
LOG_DIR=$WORKDIR/logs/geo_train_vlms
DATA=/home/ywlee/dev/gaa/data/sds_geopool.json
EVAL_DATA=/home/ywlee/dev/gaa/data/smd_onshore_big.json
EVAL_IMGDIR=/home/ywlee/SSD/smd_onshore_big

mkdir -p "$LOG_DIR" "$CKPT_BASE"
cd "$WORKDIR"

echo "11-VLM Geo Training 시작: $(date)" | tee "$LOG_DIR/summary.log"

MODELS="llava-1.5 llava-next idefics2 smolvlm paligemma aya-vision gemma-3 qwen2-vl qwen2.5-vl internvl3 llama-3.2"

for MODEL in $MODELS; do
  echo "" | tee -a "$LOG_DIR/summary.log"
  echo "===== $MODEL  시작: $(date) =====" | tee -a "$LOG_DIR/summary.log"

  HF_HOME=/home/ywlee/SSD/hf_cache \
  $PY gaa/geo_train_vlm.py \
    --model       "$MODEL" \
    --data_path   "$DATA" \
    --eval_data   "$EVAL_DATA" \
    --eval_imgdir "$EVAL_IMGDIR" \
    --output_dir  "$CKPT_BASE/geo_$MODEL" \
    --num_epochs  5 \
    --lr          2e-5 \
    --dtype       bfloat16 \
    2>&1 | tee "$LOG_DIR/${MODEL}.log"

  # 결과 요약
  grep -E "Before:|After:|Delta:" "$LOG_DIR/${MODEL}.log" 2>/dev/null | tee -a "$LOG_DIR/summary.log"
  echo "완료: $(date)" | tee -a "$LOG_DIR/summary.log"
done

echo "" | tee -a "$LOG_DIR/summary.log"
echo "전체 완료: $(date)" | tee -a "$LOG_DIR/summary.log"

#!/bin/bash
# 11개 VLM geo 검증 — SMD onshore 288샘플 baseline Attn-IoU 측정
# 목적: 모든 VLM이 해상 선박에 대한 주의집중 정렬 문제를 가짐을 증명 (GAA 동기화)
set -e

WORKDIR=/home/ywlee/share/PaperWork/GAA/GAA
PY=/home/ywlee/miniforge3/envs/eva/bin/python
LOG=$WORKDIR/logs/geo_validate_vlms.log
RESULTS=$WORKDIR/results/geo_vlm_baseline.json

SMD_DATA=/home/ywlee/dev/gaa/data/smd_onshore_big.json
SMD_IMGDIR=/home/ywlee/SSD/smd_onshore_big

mkdir -p $WORKDIR/logs $WORKDIR/results
cd $WORKDIR

echo "========================================" | tee -a $LOG
echo "11-VLM Geo Validation 시작: $(date)" | tee -a $LOG
echo "데이터: $SMD_DATA (288 real SMD samples)" | tee -a $LOG
echo "========================================" | tee -a $LOG

$PY gaa/geo_validate_vlms.py \
    --data_path   $SMD_DATA \
    --image_dir   $SMD_IMGDIR \
    --num_samples -1 \
    --dtype       float32 \
    --output_json $RESULTS \
    2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "완료: $(date)" | tee -a $LOG
echo "결과: $RESULTS" | tee -a $LOG

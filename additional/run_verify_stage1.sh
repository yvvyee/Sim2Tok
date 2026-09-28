#!/usr/bin/env bash
# Verification stage 1 (representation, hypothesis H1a).
#
# Every job uses GPUs 2,3,4,5 through torchrun and jobs run one after another.
# Training data is the AIVN-SDS 10k simulator corpus only; every evaluation set
# is real imagery and is never trained on.
#
#   1. GAA inbox_ce trained with crop-frame masks, seed 0, otherwise identical
#      to ok_n10000 (raw-frame masks).
#   2. SMD full, cropped geometry: stock, ok_n10000, crop_n10000.
#   3. Real corpora (MCAD RGB/IR, LaRS, SeaDronesSee), cropped geometry,
#      tau 0.5 and tau 0.01: stock, ok_n10000, crop_n10000.
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
export CUDA_VISIBLE_DEVICES=2,3,4,5
LOG="$ROOT/logs/verify_stage1.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

CK_RAW="$ROOT/checkpoints/ok_n10000/vision_encoder.bin"
CK_CROP_DIR="$ROOT/checkpoints/crop_n10000_s0"

if [ ! -f "$CK_CROP_DIR/vision_encoder.bin" ]; then
  say "train crop_n10000_s0"
  $BIN/torchrun --nproc_per_node 4 --master_port 29621 -m additional.train_geo_ddp \
    --data_path "$ROOT/data/ok_aivn_10000.json" \
    --image_dir "$ROOT/aivn_correct/png" \
    --output_dir "$CK_CROP_DIR" \
    --variant inbox_ce --lam 1.0 --epochs 5 --lr 1e-5 --weight_decay 0.01 \
    --per_device_batch 8 --geo_tau 0.01 --seed 0 --dtype float32 \
    --box_geometry cropped >> "$LOG" 2>&1 || { say "TRAIN FAILED"; exit 1; }
fi
CK_CROP="$CK_CROP_DIR/vision_encoder.bin"

say "eval smd_full cropped tau0.5"
$BIN/torchrun --nproc_per_node 4 --master_port 29622 -m additional.eval_attn_mp \
  --data_path "$ROOT/data/smd_full.json" --image_dir "$ROOT/frames" \
  --ckpt stock=none --ckpt raw_n10000="$CK_RAW" --ckpt crop_n10000="$CK_CROP" \
  --geo_tau 0.5 --box_geometry cropped --image_w 1920 --image_h 1080 \
  --out_dir "$ROOT/results/verify_s1/smd_full_t05" >> "$LOG" 2>&1 || say "EVAL FAILED smd"

for DS in mcad_rgb mcad_ir lars seadronessee; do
  for TAU in 0.5 0.01; do
    say "eval $DS cropped tau$TAU"
    $BIN/torchrun --nproc_per_node 4 --master_port 29623 -m additional.eval_attn_mp \
      --data_path "$ROOT/data/real/$DS.json" --image_dir /mnt/ssd1/yvvyee/realeval \
      --ckpt stock=none --ckpt raw_n10000="$CK_RAW" --ckpt crop_n10000="$CK_CROP" \
      --geo_tau "$TAU" --box_geometry cropped \
      --out_dir "$ROOT/results/verify_s1/${DS}_t${TAU/./}" >> "$LOG" 2>&1 \
      || say "EVAL FAILED $DS tau$TAU"
  done
done
say "VERIFY_STAGE1_DONE"

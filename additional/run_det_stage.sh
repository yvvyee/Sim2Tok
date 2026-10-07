#!/usr/bin/env bash
# Frozen-encoder detection probe (CenterNet head trained on AIVN-SDS train
# split only) for every encoder of the token stage.  GPUs 2,3,4,5.
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
REAL=/mnt/ssd1/yvvyee/realeval
export CUDA_VISIBLE_DEVICES=2,3,4,5
LOG="$ROOT/logs/det_stage.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
twice () { "$@" && return 0; say "retrying after failure: ${*: -2}"; "$@"; }
det () {  # $1=label $2=encoder path or none $3=seed
  local OUT="$ROOT/results/det/$1_s$3"
  [ -f "$OUT/meta.json" ] && return 0
  say "det $1 seed $3"
  twice $BIN/torchrun --nproc_per_node 4 --master_port 29663 -m additional.eval_det_probe \
    --encoder "$1=$2" --sim_frames_json "$ROOT/data/ok_aivn_10000.json" \
    --split_json "$ROOT/data/aivn_split.json" --crops_json "$ROOT/aivn_crops/crops.json" \
    --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops" \
    --real "smd=$ROOT/data/smd_full.json:$ROOT/frames" \
    --real "mcad_rgb=$ROOT/data/real/mcad_rgb.json:$REAL" \
    --real "mcad_ir=$ROOT/data/real/mcad_ir.json:$REAL" \
    --real "lars=$ROOT/data/real/lars.json:$REAL" \
    --real "seadronessee=$ROOT/data/real/seadronessee.json:$REAL" \
    --seed "$3" --out_dir "$OUT" >> "$LOG" 2>&1 || say "DET FAILED $1 s$3"
}
for S in 0 1 2; do
  det stock none "$S"
  det gaa "$ROOT/checkpoints/crop_n10000_s$S/vision_encoder.bin" "$S"
  for N in tok_occ tok_occ_state tok_occ_stshuf tok_occ_state_abs; do
    det "$N" "$ROOT/checkpoints/${N}_s$S/vision_encoder.bin" "$S"
  done
  say "DET_SEED_DONE s$S"
done
say "DET_STAGE_DONE"

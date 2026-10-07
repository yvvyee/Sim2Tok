#!/usr/bin/env bash
# After the detection queue finishes, re-evaluate (no retraining) every run
# whose records lack the center-hit flag, using its saved det_head.bin.
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
REAL=/mnt/ssd1/yvvyee/realeval
export CUDA_VISIBLE_DEVICES=2,3,4,5
LOG="$ROOT/logs/det_stage.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
until grep -q DET_STAGE_DONE "$LOG"; do sleep 60; done
enc_path () { case $1 in stock) echo none;; gaa) echo "$ROOT/checkpoints/crop_n10000_s$2/vision_encoder.bin";; *) echo "$ROOT/checkpoints/$1_s$2/vision_encoder.bin";; esac; }
for S in 0 1 2; do for N in stock gaa tok_occ tok_occ_state tok_occ_stshuf tok_occ_state_abs; do
  OUT="$ROOT/results/det/${N}_s$S"
  [ -f "$OUT/aivn_val.jsonl" ] || continue
  head -c 100000 "$OUT/aivn_val.jsonl" | grep -q '"tpc"' && continue
  say "reeval $N seed $S"
  $BIN/torchrun --nproc_per_node 4 --master_port 29664 -m additional.eval_det_probe --eval_only \
    --encoder "$N=$(enc_path $N $S)" --sim_frames_json "$ROOT/data/ok_aivn_10000.json" \
    --split_json "$ROOT/data/aivn_split.json" --crops_json "$ROOT/aivn_crops/crops.json" \
    --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops" \
    --real "smd=$ROOT/data/smd_full.json:$ROOT/frames" \
    --real "mcad_rgb=$ROOT/data/real/mcad_rgb.json:$REAL" \
    --real "mcad_ir=$ROOT/data/real/mcad_ir.json:$REAL" \
    --real "lars=$ROOT/data/real/lars.json:$REAL" \
    --real "seadronessee=$ROOT/data/real/seadronessee.json:$REAL" \
    --seed "$S" --out_dir "$OUT" >> "$LOG" 2>&1 || say "REEVAL FAILED $N s$S"
done; done
say "DET_REEVAL_DONE"

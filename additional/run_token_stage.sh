#!/usr/bin/env bash
# Token-level supervision (proposal A, components 1-3) and the frozen-encoder
# sim-to-real probe.  Training uses AIVN-SDS 10k simulator images only; the
# probe head is trained on the simulator training split and applied unchanged
# to the AIVN validation split and the real datasets.  GPUs 2,3,4,5 via
# torchrun, one job at a time.
#
#   tok_occ         occupancy + centre on layer-23 tokens
#   tok_occ_state   + navigation state (distance, relative course, CPA, TCPA)
#   tok_occ_stshuf  + navigation state permuted across vessels (control)
#   tok_occ_state_abs  + vessel-free images with all-zero targets
#
# Probed encoders per seed: stock, GAA (crop_n10000_s$S) and the four above.
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
REAL=/mnt/ssd1/yvvyee/realeval
export CUDA_VISIBLE_DEVICES=2,3,4,5
SEEDS=${SEEDS:-"0 1 2"}
LOG="$ROOT/logs/token_stage.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
twice () { "$@" && return 0; say "retrying after failure: ${*: -2}"; "$@"; }

COMMON_DATA=(--frames_json "$ROOT/data/ok_aivn_10000.json"
             --crops_json "$ROOT/aivn_crops/crops.json"
             --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops")

tok_train () {  # $1=name $2=seed, rest=flags
  local name=$1 s=$2; shift 2
  local OUT="$ROOT/checkpoints/${name}_s$s"
  [ -f "$OUT/vision_encoder.bin" ] && return 0
  say "train $name seed $s"
  twice $BIN/torchrun --nproc_per_node 4 --master_port 29661 -m additional.train_token_ddp \
    "${COMMON_DATA[@]}" --nav_json "$ROOT/data/aivn_nav_state.json" \
    --output_dir "$OUT" --total_steps 3000 --seed "$s" "$@" >> "$LOG" 2>&1 \
    || { say "TRAIN FAILED $name s$s"; return 1; }
}

probe () {  # $1=label $2=encoder path or none $3=seed
  local OUT="$ROOT/results/probe/$1_s$3"
  [ -f "$OUT/meta.json" ] && return 0
  say "probe $1 seed $3"
  twice $BIN/torchrun --nproc_per_node 4 --master_port 29662 -m additional.eval_token_probe \
    --encoder "$1=$2" --sim_frames_json "$ROOT/data/ok_aivn_10000.json" \
    --split_json "$ROOT/data/aivn_split.json" --crops_json "$ROOT/aivn_crops/crops.json" \
    --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops" \
    --real "smd=$ROOT/data/smd_full.json:$ROOT/frames" \
    --real "mcad_rgb=$ROOT/data/real/mcad_rgb.json:$REAL" \
    --real "mcad_ir=$ROOT/data/real/mcad_ir.json:$REAL" \
    --real "lars=$ROOT/data/real/lars.json:$REAL" \
    --real "seadronessee=$ROOT/data/real/seadronessee.json:$REAL" \
    --seed "$3" --out_dir "$OUT" >> "$LOG" 2>&1 || say "PROBE FAILED $1 s$3"
}

for S in $SEEDS; do
  tok_train tok_occ "$S" --channels occ
  tok_train tok_occ_state "$S" --channels occ,state
  tok_train tok_occ_stshuf "$S" --channels occ,state --shuffle_state
  tok_train tok_occ_state_abs "$S" --channels occ,state --absent
  probe stock none "$S"
  probe gaa "$ROOT/checkpoints/crop_n10000_s$S/vision_encoder.bin" "$S"
  for N in tok_occ tok_occ_state tok_occ_stshuf tok_occ_state_abs; do
    [ -f "$ROOT/checkpoints/${N}_s$S/vision_encoder.bin" ] \
      && probe "$N" "$ROOT/checkpoints/${N}_s$S/vision_encoder.bin" "$S"
  done
  say "TOKEN_SEED_DONE s$S"
done
say "TOKEN_STAGE_DONE seeds=$SEEDS"

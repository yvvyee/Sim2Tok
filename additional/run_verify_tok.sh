#!/usr/bin/env bash
# VLM behaviour with the token-supervised encoder (tok_occ, layer-23 occupancy
# and centre).  Same recipe as run_verify_stage2.sh: AIVN-SDS 10k SFT set
# only, start from the CC3M projector and LoRA, encoder frozen, evaluation on
# the AIVN validation frames and the real datasets.  GPUs 2,3,4,5.
#
#   M4  tok_occ_s$S encoder, frozen
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
LLM=/home/yvvyee/data/Models/Llama-3.1-8B-Instruct
CC3M=$ROOT/weights/tango2_cc3m/clip_llama31_proj_lora
REAL=/mnt/ssd1/yvvyee/realeval
export CUDA_VISIBLE_DEVICES=2,3,4,5
SEEDS=${SEEDS:-"0 1 2"}
LOG="$ROOT/logs/verify_tok.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
twice () { "$@" && return 0; say "retrying after failure: ${*: -2}"; "$@"; }

train_vlm () {  # $1=name $2=seed $3=encoder path or empty
  local OUT="$ROOT/checkpoints/vlm_v2_$1_s$2"
  if [ -f "$OUT/vlm_config.json" ]; then return 0; fi
  local EXTRA=""
  [ -n "$3" ] && EXTRA="--vision_encoder_path $3"
  say "train $1 seed $2 (encoder: ${3:-stock})"
  local T0=$(date +%s)
  $BIN/torchrun --nproc_per_node 4 --master_port 29641 train.py \
    --vision_model openai/clip-vit-large-patch14-336 --llm_model "$LLM" \
    --projector_type mlp2x_gelu --train_type lora \
    --projector_path "$CC3M/projector.bin" --resume_lora_path "$CC3M" \
    $EXTRA \
    --data_path "$ROOT/data/vlm_sft_v2_train.json" --image_dir / \
    --output_dir "$OUT" --num_epochs 1 --batch_size 4 --grad_accum 4 \
    --learning_rate 2e-4 --warmup_ratio 0.03 --lr_scheduler cosine \
    --save_steps 100000 --save_total_limit 1 --seed "$2" --dtype bfloat16 \
    --no_tensorboard >> "$LOG" 2>&1 || { say "TRAIN FAILED $1 s$2"; return 1; }
  echo "${3:-none}" > "$OUT/encoder_used.txt"
  say "trained $1 seed $2 in $(( $(date +%s) - T0 )) s"
}

eval_vlm () {  # $1=name $2=seed
  local CK="$ROOT/checkpoints/vlm_v2_$1_s$2"
  local ENC; ENC=$(cat "$CK/encoder_used.txt")
  for DS in aivn_val smd mcad_rgb mcad_ir lars seadronessee buoy_distance; do
    local DATA="$ROOT/data/real/$DS.json" IMG="$REAL"
    if [ "$DS" = smd ]; then DATA="$ROOT/data/smd_full.json"; IMG="$ROOT/frames"; fi
    [ "$DS" = aivn_val ] && IMG=/
    local OUTD="$ROOT/results/verify_s2/$1_s$2/$DS"
    say "eval $1 s$2 on $DS"
    local T0=$(date +%s)
    twice $BIN/torchrun --nproc_per_node 4 --master_port 29642 -m additional.yesno_eval \
      --data_path "$DATA" --image_dir "$IMG" --vlm_ckpt "$CK" --llm "$LLM" \
      --encoder "$1=$ENC" --box_geometry cropped \
      --vessel_words ship --sides left,right --counts 1,3 \
      --absent airplane,car,building --present water \
      --keep_text 1 --batch_size 32 --out_dir "$OUTD" >> "$LOG" 2>&1 \
      || say "EVAL FAILED $1 s$2 $DS"
    say "evaluated $1 s$2 on $DS in $(( $(date +%s) - T0 )) s"
  done
}

for S in $SEEDS; do
  E="$ROOT/checkpoints/tok_occ_s$S/vision_encoder.bin"
  [ -f "$E" ] || { say "ENCODER MISSING $E"; continue; }
  train_vlm M4 "$S" "$E" && eval_vlm M4 "$S"
done
say "VERIFY_TOK_DONE seeds=$SEEDS"

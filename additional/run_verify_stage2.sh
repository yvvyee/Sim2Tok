#!/usr/bin/env bash
# Verification stage 2: VLM-level hypotheses H1b, H1c, H2a, H2b, H2c.
#
# Training uses the AIVN-SDS 10k simulator corpus only (frames and sub-crops).
# Every VLM starts from ETRI-TANGO/tango2-sds-vlm-eva clip_llama31_proj_lora
# (projector and LoRA aligned on CC3M).  Evaluation uses real corpora only,
# plus the held-out AIVN validation frames for the sim-side score of H1c.
# GPUs 2,3,4,5 via torchrun, one job at a time.
#
#   M0  stock CLIP, frozen
#   M1  GAA inbox_ce encoder trained with crop-frame masks, frozen
#   M3  inbox_ce encoder trained on shuffled boxes (crop frame), frozen
#
# SEEDS selects the seeds to run (default "0").  For every seed the GAA and
# shuffled-box encoders are trained with that seed (seed 0 reuses the stage-1
# encoders), and seeds other than 0 also get the stage-1 alignment evaluation.
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
LLM=/home/yvvyee/data/Models/Llama-3.1-8B-Instruct
CC3M=$ROOT/weights/tango2_cc3m/clip_llama31_proj_lora
REAL=/mnt/ssd1/yvvyee/realeval
export CUDA_VISIBLE_DEVICES=2,3,4,5
SEEDS=${SEEDS:-0}
LOG="$ROOT/logs/verify_stage2.log"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

enc_train () {  # $1=out dir $2=data json $3=seed
  [ -f "$1/vision_encoder.bin" ] && return 0
  say "train encoder $(basename $1)"
  $BIN/torchrun --nproc_per_node 4 --master_port 29651 -m additional.train_geo_ddp \
    --data_path "$2" --image_dir "$ROOT/aivn_correct/png" --output_dir "$1" \
    --variant inbox_ce --lam 1.0 --epochs 5 --lr 1e-5 --weight_decay 0.01 \
    --per_device_batch 8 --geo_tau 0.01 --seed "$3" --dtype float32 \
    --box_geometry cropped >> "$LOG" 2>&1 || { say "ENCODER FAILED $1"; return 1; }
}
enc_h1a () {  # $1=tag $2=label $3=encoder path
  local run
  run () {  # $1=outdir $2=data $3=image dir $4=tau
    [ -f "$1/summary.json" ] && return 0
    $BIN/torchrun --nproc_per_node 4 --master_port 29652 -m additional.eval_attn_mp \
      --data_path "$2" --image_dir "$3" --ckpt stock=none --ckpt "$ENC_LABEL=$ENC_PATH" \
      --geo_tau "$4" --box_geometry cropped --image_w 1920 --image_h 1080 \
      --out_dir "$1" >> "$LOG" 2>&1 || say "H1a EVAL FAILED $1"
  }
  ENC_LABEL=$2; ENC_PATH=$3
  say "H1a eval $1"
  run "$ROOT/results/verify_s1/$1_smd_full_t05" "$ROOT/data/smd_full.json" "$ROOT/frames" 0.5
  for DS in mcad_rgb mcad_ir lars seadronessee; do
    for TAU in 0.5 0.01; do
      run "$ROOT/results/verify_s1/$1_${DS}_t${TAU/./}" "$ROOT/data/real/$DS.json" "$REAL" "$TAU"
    done
  done
}

# ---------------------------------------------------------------- shuffled data
if [ ! -f "$ROOT/data/aivn_10k_shuffled.json" ]; then
  $BIN/python -m additional.make_shuffled_boxes \
    --in_json "$ROOT/data/ok_aivn_10000.json" \
    --out_json "$ROOT/data/aivn_10k_shuffled.json" --seed 0 >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- SFT data
until [ -f "$ROOT/aivn_crops/crops.json" ]; do sleep 60; done
if [ ! -f "$ROOT/data/vlm_sft_v2_train.json" ]; then
  say "build SFT set"
  $BIN/python -m additional.build_vlm_sft_v2 \
    --describe_json "$ROOT/data/ok_aivn_10000.json" \
    --advice_json "$ROOT/data/aivn_10k_advice.json" \
    --crops_json "$ROOT/aivn_crops/crops.json" \
    --split_json "$ROOT/data/aivn_split.json" \
    --frame_image_dir "$ROOT/aivn_correct/png" \
    --crop_image_dir "$ROOT/aivn_crops" \
    --out_train "$ROOT/data/vlm_sft_v2_train.json" \
    --out_val "$ROOT/data/vlm_sft_v2_val.json" \
    --stats_json "$ROOT/results/vlm_sft_v2_stats.json" >> "$LOG" 2>&1 \
    || { say "SFT BUILD FAILED"; exit 1; }
  # Held-out simulator frames for the sim-side score (H1c).
  $BIN/python - "$ROOT" <<'PY' >> "$LOG" 2>&1
import json, sys, os
R = sys.argv[1]
split = json.load(open(f"{R}/data/aivn_split.json"))["split"]
frames = [dict(r, width=1920, height=1080, dataset="aivn_val", video=r["id"],
               vessel_labelled=True, image=os.path.join(f"{R}/aivn_correct/png", r["image"]))
          for r in json.load(open(f"{R}/data/ok_aivn_10000.json")) if split[r["id"]] == "val"]
json.dump(frames, open(f"{R}/data/real/aivn_val.json", "w"), ensure_ascii=False)
print("[aivn_val]", len(frames), "frames")
PY
fi

# ---------------------------------------------------------------- train + eval
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
    $BIN/torchrun --nproc_per_node 4 --master_port 29642 -m additional.yesno_eval \
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
  E1="$ROOT/checkpoints/crop_n10000_s$S"; E3="$ROOT/checkpoints/shuf_crop_n10000_s$S"
  enc_train "$E1" "$ROOT/data/ok_aivn_10000.json" "$S" || continue
  enc_train "$E3" "$ROOT/data/aivn_10k_shuffled.json" "$S" || continue
  [ "$S" != 0 ] && enc_h1a "s${S}" "crop_n10000" "$E1/vision_encoder.bin"
  [ "$S" != 0 ] && enc_h1a "shuf_s${S}" "shuf_crop_n10000" "$E3/vision_encoder.bin"
  train_vlm M0 "$S" ""                         && eval_vlm M0 "$S"
  train_vlm M1 "$S" "$E1/vision_encoder.bin"   && eval_vlm M1 "$S"
  train_vlm M3 "$S" "$E3/vision_encoder.bin"   && eval_vlm M3 "$S"
done
say "VERIFY_STAGE2_DONE seeds=$SEEDS"

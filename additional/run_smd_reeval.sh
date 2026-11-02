#!/usr/bin/env bash
# SMD re-evaluation with vessel classes only (2026-11-02).
#
# The first SMD conversion used every ObjectGT box, including buoys, birds or
# planes and "Other" (about 15% of the boxes).  smd_full_to_gaa.py now keeps
# ObjectGT class ids 1, 3, 4, 5, 6 and 7 (Ferry, Vessel/ship, Speed boat, Boat,
# Kayak, Sail boat) as vessels.  The image set is unchanged.  No encoder or
# head is retrained: every saved encoder is evaluated with its saved probe,
# detection or YOLO head on the new SMD labels.
#
# Host tg-a80g, GPUs 0 and 1 (config.HOST_POLICY), one job at a time.
#
#   step 0  frames and smd_vessel.json from the SMD videos; smd_allcls.json with
#           every class, which must reproduce the old data/smd_full.json
#   step 1  equivalence: rev_tok seed 0 on the old labels must reproduce the
#           saved results/rev_probe/rev_tok_s0/smd.jsonl
#   step 2  frozen-encoder probes on the vessel labels
#   step 3  attention overlap on the vessel labels
#   step 4  detection probes on the vessel labels
#   step 5  YOLOv8s reference on the vessel labels
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_smd
cd "$ROOT/src"
BIN=/mnt/ssd1/yvvyee/miniconda3/envs/eva/bin
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_CACHE=/mnt/ssd1/yvvyee/hf/hub HF_HUB_OFFLINE=1
NP=2
STEPS=${STEPS:-"0 1 2 3 4 5"}
SEEDS="0 1 2"
LOG="$ROOT/logs/smd_reeval.log"
mkdir -p "$ROOT/logs" "$ROOT/results"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
twice () { "$@" && return 0; say "retrying after failure: ${*: -2}"; "$@"; }
L14=openai/clip-vit-large-patch14-336
B16=openai/clip-vit-base-patch16
SIG=google/siglip-so400m-patch14-384
SIM=(--sim_frames_json "$ROOT/data/ok_aivn_10000.json" --split_json "$ROOT/data/aivn_split.json"
     --crops_json "$ROOT/aivn_crops/crops.json" --frame_dir "$ROOT/aivn_correct/png"
     --crop_dir "$ROOT/aivn_crops")
SMDV="smd=$ROOT/data/smd_vessel.json:$ROOT/frames"
enc () { [ "$2" = none ] && echo none || echo "$ROOT/checkpoints/rev/$1_s$2/vision_encoder.bin"; }

step0 () {
  if [ ! -f "$ROOT/data/smd_vessel.json" ]; then
    say "convert SMD (vessel classes, frames)"
    $BIN/python -m additional.smd_full_to_gaa --smd_root "$ROOT/SMD" --split VIS_Onshore NIR \
      --out_json "$ROOT/data/smd_vessel.json" --out_images "$ROOT/frames" --workers 32 \
      --stats_json "$ROOT/results/smd_vessel_stats.json" >> "$LOG" 2>&1 || { say "CONVERT FAILED"; return 1; }
  fi
  if [ ! -f "$ROOT/data/smd_allcls.json" ]; then
    say "convert SMD (all classes, json only)"
    $BIN/python -m additional.smd_full_to_gaa --smd_root "$ROOT/SMD" --split VIS_Onshore NIR \
      --out_json "$ROOT/data/smd_allcls.json" --out_images "$ROOT/frames" --workers 32 --no_images \
      --vessel_ids 0,1,2,3,4,5,6,7,8,9,10 >> "$LOG" 2>&1 || { say "CONVERT FAILED"; return 1; }
  fi
  say "STEP0_DONE"
}

probe () {  # $1=run $2=seed $3=vision model $4=smd spec $5=out root
  local OUT="$5/$1_s$2"
  [ -f "$OUT/smd.jsonl" ] && return 0
  say "probe $1 s$2 -> $OUT"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29713 -m additional.eval_token_probe \
    --encoder "$1=$(enc $1 $2)" --vision_model "$3" "${SIM[@]}" --real "$4" \
    --init_head "$ROOT/results/rev_probe/$1_s$2/probe_head.bin" --skip_sim_val \
    --seed "$2" --out_dir "$OUT" >> "$LOG" 2>&1 || say "PROBE FAILED $1 s$2"
}

step1 () {
  probe rev_tok 0 $L14 "smd=$ROOT/data/smd_full.json:$ROOT/frames" "$ROOT/results/eqcheck"
  say "STEP1_DONE"
}

step2 () {
  local P="$ROOT/results/rev_probe_smdv"
  for S in $SEEDS; do
    for N in rev_stock rev_tok rev_gaa rev_gaam rev_gaa23 rev_tokdr rev_gaadr rev_img \
             rev_state rev_stshuf rev_state_abs; do
      [ $N = rev_stock ] && E=none || E=$S
      probe_e $N $S $L14 $E "$P"
    done
    for N in b16_stock b16_tok b16_gaam; do [ $N = b16_stock ] && E=none || E=$S; probe_e $N $S $B16 $E "$P"; done
    for N in sig_stock sig_tok sig_gaam; do [ $N = sig_stock ] && E=none || E=$S; probe_e $N $S $SIG $E "$P"; done
  done
  say "STEP2_DONE"
}
probe_e () {  # $1=run $2=seed $3=vision model $4=seed or none for the encoder $5=out root
  local OUT="$5/$1_s$2"
  [ -f "$OUT/smd.jsonl" ] && return 0
  say "probe $1 s$2"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29713 -m additional.eval_token_probe \
    --encoder "$1=$(enc $1 $4)" --vision_model "$3" "${SIM[@]}" --real "$SMDV" \
    --init_head "$ROOT/results/rev_probe/$1_s$2/probe_head.bin" --skip_sim_val \
    --seed "$2" --out_dir "$OUT" >> "$LOG" 2>&1 || say "PROBE FAILED $1 s$2"
}

attn () {  # $1=tag $2=label $3=encoder path $4=attn layer
  local O="$ROOT/results/rev_attn_smdv/$1_smd"
  [ -f "$O/summary.json" ] && return 0
  say "attn $1"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29718 -m additional.eval_attn_mp \
    --data_path "$ROOT/data/smd_vessel.json" --image_dir "$ROOT/frames" --ckpt stock=none --ckpt "$2=$3" \
    --geo_tau 0.5 --box_geometry cropped --image_w 1920 --image_h 1080 \
    --attn_layer "$4" --out_dir "$O" >> "$LOG" 2>&1 || say "ATTN FAILED $1"
}
step3 () {
  for S in $SEEDS; do
    attn "gaa_s$S" rev_gaa "$(enc rev_gaa $S)" -1
    attn "gaam_s$S" rev_gaam "$(enc rev_gaam $S)" -1
    attn "gaashuf_s$S" rev_gaashuf "$(enc rev_gaashuf $S)" -1
    attn "gaa23_l23_s$S" rev_gaa23 "$(enc rev_gaa23 $S)" -2
    attn "gaa23_l24_s$S" rev_gaa23 "$(enc rev_gaa23 $S)" -1
  done
  say "STEP3_DONE"
}

step4 () {
  for S in $SEEDS; do
    for N in rev_stock rev_tok rev_gaa rev_gaam; do
      local OUT="$ROOT/results/rev_det_smdv/${N}_s$S"
      [ -f "$OUT/smd.jsonl" ] && continue
      mkdir -p "$OUT" && cp "$ROOT/results/rev_det/${N}_s$S/det_head.bin" "$OUT/"
      [ $N = rev_stock ] && E=none || E=$S
      say "det $N s$S"
      twice $BIN/torchrun --nproc_per_node $NP --master_port 29717 -m additional.eval_det_probe \
        --encoder "$N=$(enc $N $E)" "${SIM[@]}" --real "$SMDV" --eval_only --skip_sim_val \
        --per_device_batch 24 --seed "$S" --out_dir "$OUT" >> "$LOG" 2>&1 || say "DET FAILED $N s$S"
    done
  done
  say "STEP4_DONE"
}

step5 () {
  local OUT="$ROOT/results/yolo_ref_smdv"
  if [ ! -f "$OUT/smd.jsonl" ]; then
    say "yolo eval"
    twice $BIN/torchrun --nproc_per_node $NP --master_port 29714 -m additional.yolo_ref eval \
      --frames_json "$ROOT/data/ok_aivn_10000.json" --split_json "$ROOT/data/aivn_split.json" \
      --crops_json "$ROOT/aivn_crops/crops.json" --frame_dir "$ROOT/aivn_correct/png" \
      --crop_dir "$ROOT/aivn_crops" --weights "$ROOT/results/yolo_ref/train/weights/best.pt" \
      --real "$SMDV" --skip_sim_val --out_dir "$OUT" >> "$LOG" 2>&1 || say "YOLO FAILED"
  fi
  say "STEP5_DONE"
}

say "SMD_REEVAL start steps=[$STEPS]"
for K in $STEPS; do "step$K"; done
say "SMD_REEVAL_DONE"

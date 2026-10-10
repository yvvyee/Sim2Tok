#!/usr/bin/env bash
# Revision experiments for the MaCVi submission (items 1-8 of the 2026-10-08
# review).  Simulator data only: every encoder is trained on the AIVN-SDS
# training split (--train_only), so the AIVN validation split is unseen by the
# encoders as well as the probe heads.  Real datasets are used for evaluation
# only.  GPUs 2,3,4,5 through torchrun (4 ranks), one job at a time.  The
# first encoder (rev_tok seed 0) ran on GPUs 2,3,4 with 3 ranks; the global
# batch sizes below are the same for 3 and 4 ranks, so only the sharding of
# the data order differs.
#
#   token / GAA          global batch 24, 4,000 steps        (96,000 samples)
#   probe head           global batch 48, 2,000 steps        (96,000 samples)
#   detection probe      global batch 48, 4,000 steps        (192,000 samples)
#   VLM                  global batch 48 (per-device x accumulation 4)
#
# Blocks, in order (each block runs all seeds before the next starts):
#   A  core: token occupancy, GAA as before (frames, 5 epochs), GAA matched to
#      the token data and steps (frames + vessel crops, 4,000 steps); probes
#   B  GAA on the layer-23 class token (matched data and steps)
#   C  domain randomization (token occ., matched GAA) and image-level control
#   D  CLIP ViT-B/16: stock, matched GAA, token occ.
#   E  YOLOv8s simulator-only reference
#   F  VLMs on the train-only encoders: M0 stock, M1 matched GAA, M4 token
#   G  SigLIP so400m/14-384: stock, matched GAA, token occ.
#   H  detection probes, attention alignment, permuted-box GAA control
#   I  navigation-state and absence variants
set -u
ROOT=/mnt/ssd1/yvvyee/gaa_additional
cd "$ROOT/src"
BIN=/home/yvvyee/miniconda3/envs/eva/bin
REAL=/mnt/ssd1/yvvyee/realeval
LLM=/home/yvvyee/data/Models/Llama-3.1-8B-Instruct
CC3M=$ROOT/weights/tango2_cc3m/clip_llama31_proj_lora
CUE=$ROOT/cue_sets
export CUDA_VISIBLE_DEVICES=2,3,4,5
# 2026-10-10: unattended-upgrade replaced the NVIDIA user-space libraries
# (595.99.02) while the loaded kernel module stayed at 595.91.07, so NVML and
# hence NCCL fail.  The matching 595.91.07 user-space libraries, extracted from
# NVIDIA's runfile into the home directory, are put first on the library path
# for these jobs only.  No system file is changed.
NVLIB=$HOME/nvidia_595.91.07/lib
if [ -d "$NVLIB" ] && ! grep -q "595.99" /proc/driver/nvidia/version; then
  export LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
NP=4
PD_ENC=$((24 / NP)); PD_PROBE=$((48 / NP)); PD_VLM=$((12 / NP))
SEEDS=${SEEDS:-"0 1 2"}
BLOCKS=${BLOCKS:-"A B C D E F G H I"}
LOG="$ROOT/logs/rev_stage.log"
CK=$ROOT/checkpoints/rev
PR=$ROOT/results/rev_probe
mkdir -p "$CK" "$PR"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
twice () { "$@" && return 0; say "retrying after failure: ${*: -2}"; "$@"; }
L14=openai/clip-vit-large-patch14-336
B16=openai/clip-vit-base-patch16
SIG=google/siglip-so400m-patch14-384

SIMD=(--frames_json "$ROOT/data/ok_aivn_10000.json" --crops_json "$ROOT/aivn_crops/crops.json"
      --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops")
SPLIT=(--split_json "$ROOT/data/aivn_split.json" --train_only)
REALS=(--real "smd=$ROOT/data/smd_full.json:$ROOT/frames"
       --real "mcad_rgb=$ROOT/data/real/mcad_rgb.json:$REAL"
       --real "mcad_ir=$ROOT/data/real/mcad_ir.json:$REAL"
       --real "lars=$ROOT/data/real/lars.json:$REAL"
       --real "seadronessee=$ROOT/data/real/seadronessee.json:$REAL")
CUES=(--real "aivn_val_tex=$CUE/aivn_val_tex.json:$CUE"
      --real "aivn_val_shape=$CUE/aivn_val_shape.json:$CUE")

tok () {  # $1=name $2=seed $3=vision model, rest=flags
  local name=$1 s=$2 vm=$3; shift 3
  local OUT="$CK/${name}_s$s"
  [ -f "$OUT/vision_encoder.bin" ] && return 0
  say "train $name s$s"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29711 -m additional.train_token_ddp \
    "${SIMD[@]}" "${SPLIT[@]}" --nav_json "$ROOT/data/aivn_nav_state.json" \
    --vision_model "$vm" --output_dir "$OUT" --total_steps 4000 --per_device_batch $PD_ENC \
    --seed "$s" "$@" >> "$LOG" 2>&1 || { say "TRAIN FAILED $name s$s"; return 1; }
}

gaa () {  # $1=name $2=seed $3=vision model $4=data json, rest=flags
  local name=$1 s=$2 vm=$3 data=$4; shift 4
  local OUT="$CK/${name}_s$s"
  [ -f "$OUT/vision_encoder.bin" ] && return 0
  say "train $name s$s"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29712 -m additional.train_geo_ddp \
    --data_path "$data" --image_dir "$ROOT/aivn_correct/png" "${SPLIT[@]}" \
    --vision_model "$vm" --output_dir "$OUT" --variant inbox_ce --lam 1.0 --lr 1e-5 \
    --weight_decay 0.01 --per_device_batch $PD_ENC --geo_tau 0.01 --box_geometry cropped \
    --dtype float32 --seed "$s" "$@" >> "$LOG" 2>&1 || { say "TRAIN FAILED $name s$s"; return 1; }
}
MATCHED=(--crops_json "$ROOT/aivn_crops/crops.json" --crop_dir "$ROOT/aivn_crops"
         --epochs 1000 --max_steps 4000)

probe () {  # $1=label $2=seed $3=vision model $4=encoder path or none
  local OUT="$PR/$1_s$2"
  [ -f "$OUT/meta.json" ] && return 0
  [ "$4" != none ] && [ ! -f "$4" ] && { say "PROBE SKIPPED $1 s$2 (no encoder)"; return 1; }
  say "probe $1 s$2"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29713 -m additional.eval_token_probe \
    --encoder "$1=$4" --vision_model "$3" --sim_frames_json "$ROOT/data/ok_aivn_10000.json" \
    --split_json "$ROOT/data/aivn_split.json" --crops_json "$ROOT/aivn_crops/crops.json" \
    --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops" \
    "${REALS[@]}" "${CUES[@]}" --head_steps 2000 --per_device_batch $PD_PROBE \
    --seed "$2" --out_dir "$OUT" >> "$LOG" 2>&1 || say "PROBE FAILED $1 s$2"
}
enc () { echo "$CK/$1_s$2/vision_encoder.bin"; }

block_A () {
  for S in $SEEDS; do
    tok rev_tok "$S" $L14 --channels occ
    gaa rev_gaa "$S" $L14 "$ROOT/data/ok_aivn_10000.json" --epochs 5
    gaa rev_gaam "$S" $L14 "$ROOT/data/ok_aivn_10000.json" "${MATCHED[@]}"
    probe rev_stock "$S" $L14 none
    for N in rev_tok rev_gaa rev_gaam; do probe $N "$S" $L14 "$(enc $N $S)"; done
  done
  say "BLOCK_A_DONE"
}
block_B () {
  for S in $SEEDS; do
    gaa rev_gaa23 "$S" $L14 "$ROOT/data/ok_aivn_10000.json" "${MATCHED[@]}" --attn_layer -2
    probe rev_gaa23 "$S" $L14 "$(enc rev_gaa23 $S)"
  done
  say "BLOCK_B_DONE"
}
block_C () {
  for S in $SEEDS; do
    tok rev_tokdr "$S" $L14 --channels occ --dr
    gaa rev_gaadr "$S" $L14 "$ROOT/data/ok_aivn_10000.json" "${MATCHED[@]}" --dr
    tok rev_img "$S" $L14 --target image
    for N in rev_tokdr rev_gaadr rev_img; do probe $N "$S" $L14 "$(enc $N $S)"; done
  done
  say "BLOCK_C_DONE"
}
block_D () {
  for S in $SEEDS; do
    tok b16_tok "$S" $B16 --channels occ
    gaa b16_gaam "$S" $B16 "$ROOT/data/ok_aivn_10000.json" "${MATCHED[@]}"
    probe b16_stock "$S" $B16 none
    for N in b16_tok b16_gaam; do probe $N "$S" $B16 "$(enc $N $S)"; done
  done
  say "BLOCK_D_DONE"
}
block_E () {
  local Y=$ROOT/results/yolo_ref
  if [ ! -f "$Y/train/weights/best.pt" ]; then
    say "train YOLOv8s (sim only)"
    $BIN/python -m additional.assert_gpu_policy --select-device 0,1,2,3 >> "$LOG" 2>&1 \
      || { say "YOLO GPU POLICY CHECK FAILED"; return 1; }
    (cd "$ROOT" && $BIN/yolo detect train model="$ROOT/yolov8s.pt" data="$ROOT/yolo_sim/data.yaml" \
      epochs=40 imgsz=1280 batch=48 device=0,1,2,3 workers=12 seed=0 \
      project="$Y" name=train exist_ok=True val=True plots=False) >> "$LOG" 2>&1 \
      || { say "YOLO TRAIN FAILED"; return 1; }
  fi
  if [ ! -f "$Y/eval/meta.json" ]; then
    say "eval YOLOv8s"
    twice $BIN/torchrun --nproc_per_node $NP --master_port 29714 -m additional.yolo_ref eval \
      "${SIMD[@]}" --split_json "$ROOT/data/aivn_split.json" \
      --weights "$Y/train/weights/best.pt" "${REALS[@]}" --out_dir "$Y/eval" >> "$LOG" 2>&1 \
      || say "YOLO EVAL FAILED"
  fi
  say "BLOCK_E_DONE"
}

train_vlm () {  # $1=name $2=seed $3=encoder path or empty
  local OUT="$CK/vlm_$1_s$2"
  [ -f "$OUT/vlm_config.json" ] && return 0
  local EXTRA=""
  [ -n "$3" ] && EXTRA="--vision_encoder_path $3"
  say "train VLM $1 s$2 (encoder: ${3:-stock})"
  $BIN/torchrun --nproc_per_node $NP --master_port 29715 train.py \
    --vision_model $L14 --llm_model "$LLM" \
    --projector_type mlp2x_gelu --train_type lora \
    --projector_path "$CC3M/projector.bin" --resume_lora_path "$CC3M" \
    $EXTRA \
    --data_path "$ROOT/data/vlm_sft_v2_train.json" --image_dir / \
    --output_dir "$OUT" --num_epochs 1 --batch_size $PD_VLM --grad_accum 4 \
    --learning_rate 2e-4 --warmup_ratio 0.03 --lr_scheduler cosine \
    --save_steps 100000 --save_total_limit 1 --seed "$2" --dtype bfloat16 \
    --no_tensorboard >> "$LOG" 2>&1 || { say "VLM TRAIN FAILED $1 s$2"; return 1; }
  echo "${3:-none}" > "$OUT/encoder_used.txt"
}
eval_vlm () {  # $1=name $2=seed
  local C="$CK/vlm_$1_s$2" E
  E=$(cat "$C/encoder_used.txt")
  for DS in aivn_val smd mcad_rgb mcad_ir lars seadronessee buoy_distance; do
    local DATA="$ROOT/data/real/$DS.json" IMG="$REAL"
    [ "$DS" = smd ] && { DATA="$ROOT/data/smd_full.json"; IMG="$ROOT/frames"; }
    [ "$DS" = aivn_val ] && IMG=/
    local O="$ROOT/results/rev_vlm/$1_s$2/$DS"
    [ -f "$O/summary.json" ] && continue
    say "eval VLM $1 s$2 $DS"
    twice $BIN/torchrun --nproc_per_node $NP --master_port 29716 -m additional.yesno_eval \
      --data_path "$DATA" --image_dir "$IMG" --vlm_ckpt "$C" --llm "$LLM" \
      --encoder "$1=$E" --box_geometry cropped \
      --vessel_words ship --sides left,right --counts 1,3 \
      --absent airplane,car,building --present water \
      --keep_text 1 --batch_size 32 --out_dir "$O" >> "$LOG" 2>&1 \
      || say "VLM EVAL FAILED $1 s$2 $DS"
  done
}
block_F () {
  for S in $SEEDS; do
    train_vlm M0 "$S" ""                     && eval_vlm M0 "$S"
    train_vlm M1 "$S" "$(enc rev_gaam $S)"   && eval_vlm M1 "$S"
    train_vlm M4 "$S" "$(enc rev_tok $S)"    && eval_vlm M4 "$S"
  done
  say "BLOCK_F_DONE"
}
block_G () {
  for S in $SEEDS; do
    tok sig_tok "$S" $SIG --channels occ
    gaa sig_gaam "$S" $SIG "$ROOT/data/ok_aivn_10000.json" "${MATCHED[@]}"
    probe sig_stock "$S" $SIG none
    for N in sig_tok sig_gaam; do probe $N "$S" $SIG "$(enc $N $S)"; done
  done
  say "BLOCK_G_DONE"
}

det () {  # $1=label $2=seed $3=encoder path or none
  local OUT="$ROOT/results/rev_det/$1_s$2"
  [ -f "$OUT/meta.json" ] && return 0
  say "det probe $1 s$2"
  twice $BIN/torchrun --nproc_per_node $NP --master_port 29717 -m additional.eval_det_probe \
    --encoder "$1=$3" --sim_frames_json "$ROOT/data/ok_aivn_10000.json" \
    --split_json "$ROOT/data/aivn_split.json" --crops_json "$ROOT/aivn_crops/crops.json" \
    --frame_dir "$ROOT/aivn_correct/png" --crop_dir "$ROOT/aivn_crops" \
    "${REALS[@]}" --head_steps 4000 --per_device_batch $PD_PROBE \
    --seed "$2" --out_dir "$OUT" >> "$LOG" 2>&1 || say "DET FAILED $1 s$2"
}
attn () {  # $1=tag $2=label $3=encoder path $4=attn layer
  for DS in smd mcad_rgb mcad_ir lars seadronessee; do
    local DATA="$ROOT/data/real/$DS.json" IMG="$REAL"
    [ "$DS" = smd ] && { DATA="$ROOT/data/smd_full.json"; IMG="$ROOT/frames"; }
    local O="$ROOT/results/rev_attn/$1_$DS"
    [ -f "$O/summary.json" ] && continue
    twice $BIN/torchrun --nproc_per_node $NP --master_port 29718 -m additional.eval_attn_mp \
      --data_path "$DATA" --image_dir "$IMG" --ckpt stock=none --ckpt "$2=$3" \
      --geo_tau 0.5 --box_geometry cropped --image_w 1920 --image_h 1080 \
      --attn_layer "$4" --out_dir "$O" >> "$LOG" 2>&1 || say "ATTN FAILED $1 $DS"
  done
}
block_H () {
  for S in $SEEDS; do
    det rev_stock "$S" none
    for N in rev_tok rev_gaa rev_gaam; do det $N "$S" "$(enc $N $S)"; done
    gaa rev_gaashuf "$S" $L14 "$ROOT/data/aivn_10k_shuffled.json" --epochs 5
    say "attention s$S"
    attn "gaa_s$S" rev_gaa "$(enc rev_gaa $S)" -1
    attn "gaam_s$S" rev_gaam "$(enc rev_gaam $S)" -1
    attn "gaashuf_s$S" rev_gaashuf "$(enc rev_gaashuf $S)" -1
    attn "gaa23_l23_s$S" rev_gaa23 "$(enc rev_gaa23 $S)" -2
    attn "gaa23_l24_s$S" rev_gaa23 "$(enc rev_gaa23 $S)" -1
  done
  say "BLOCK_H_DONE"
}
block_I () {
  for S in $SEEDS; do
    tok rev_state "$S" $L14 --channels occ,state
    tok rev_stshuf "$S" $L14 --channels occ,state --shuffle_state
    tok rev_state_abs "$S" $L14 --channels occ,state --absent
    for N in rev_state rev_stshuf rev_state_abs; do probe $N "$S" $L14 "$(enc $N $S)"; done
  done
  say "BLOCK_I_DONE"
}

say "REV_STAGE start blocks=[$BLOCKS] seeds=[$SEEDS]"
until [ -f "$CUE/stats.json" ] && [ -f "$ROOT/yolo_sim/data.yaml" ]; do sleep 60; done
for B in $BLOCKS; do "block_$B"; done
say "REV_STAGE_DONE"

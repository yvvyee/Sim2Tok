#!/usr/bin/env bash
# Corrected-bbox re-run master (2026-06-24): items 2 (full-GAA + synthetic viz) & 3 (VLM eval).
# All full-GAA outputs use *_fix dirs to avoid resuming pre-fix checkpoints.
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
VM=openai/clip-vit-large-patch14-336
LLM=/home/ywlee/SSD/Llama-3.1-8B-Instruct
PROJ=/home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en/projector.bin
LORA=/home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en
TRAIN=data/sds_gaa_en_train80.json
IMGDIR=/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227
ONSHORE="--data_path data/smd_onshore_big.json --image_dir /home/ywlee/SSD/smd_onshore_big"

train_gaa () {  # $1=outdir extra args $2..
  local OUT="checkpoints/$1"; shift
  $PY gaa/train_gaa.py --vision_model $VM --llm_model $LLM --projector_path $PROJ \
    --resume_lora_path $LORA --data_path $TRAIN --image_dir $IMGDIR --output_dir "$OUT" \
    --num_epochs 5 --batch_size 1 --grad_accum 2 --dtype bfloat16 --logging_steps 50 --save_steps 1000 \
    "$@" 2>&1 | grep -oE "{.loss/[^}]*}|Training complete" | tail -2
  $PY gaa/evaluate_attn.py $ONSHORE --gaa_ckpt "$OUT/vision_encoder.bin" --dtype float32 \
    2>&1 | grep -E "stock-CLIP|GAA "
}

echo "===== PHASE 1: full-GAA corrected re-train + onshore eval ====="
echo "## ctrl_baseline_fix (SFT-only)";  train_gaa ctrl_baseline_fix --geo_loss_weight 0.0
echo "## ctrl_crop_fix (box-crop)";      train_gaa ctrl_crop_fix --privileged_mode input_crop
for LAM in 0.05 0.1 0.3 0.7; do
  echo "## gaa_lam${LAM}_fix (inbox_ce)"; train_gaa gaa_lam${LAM}_fix --geo_variant inbox_ce --geo_loss_weight $LAM
done

echo "===== PHASE 2: regen geo_encoders (mixed, corrected) ====="
$PY gaa/loss_study.py --data_path data/geopool_plus_real.json --image_dir / \
  --test_data_path data/smd_onshore_big.json --test_image_dir /home/ywlee/SSD/smd_onshore_big \
  --variants inbox_ce --steps 200 --lr 1e-5 --batch_size 8 --dtype float32 \
  --save_dir checkpoints/geo_encoders_fix 2>&1 | grep -E "stock|inbox_ce|save"

echo "===== PHASE 3: synthetic-set attention viz (corrected boxes) ====="
CK="stock=stock,noGAA=checkpoints/ctrl_baseline_fix/vision_encoder.bin,GAA=checkpoints/gaa_lam0.3_fix/vision_encoder.bin,GAAmixed=checkpoints/geo_encoders_fix/inbox_ce_vision_encoder.bin"
$PY gaa/attn_visualize.py --data_path data/sds_gaa_en.json --image_dir $IMGDIR \
  --out_dir /home/ywlee/SSD/gaa_attn_viz_fix/sds_synth --tag sds_synth_fix --ckpts "$CK" --dtype float32 \
  2>&1 | grep -E "mean attn_in_box|saved"

echo "===== PHASE 4: VLM 11-model behavioral eval (fixed parser/paligemma) ====="
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)" HF_HOME=/home/ywlee/SSD/hf_cache
mkdir -p results/vlm_behavior_prefix_backup
mv results/vlm_behavior/*.json results/vlm_behavior_prefix_backup/ 2>/dev/null || true
for M in smolvlm qwen2-vl qwen2.5-vl llava-1.5 llava-next internvl3 idefics2 paligemma aya-vision gemma-3; do
  echo "##### VLM=$M $(date '+%H:%M:%S') #####"
  $PY gaa/vlm_behavior_eval.py --model "$M" --num 100 2>&1 | grep -E "behavior\]|Error|Traceback|CUDA out|Gated" | tail -4 || echo "FAILED $M"
done
$PY gaa/vlm_behavior_eval.py --model llama-3.2 --num 100 --four_bit 2>&1 | grep -E "behavior\]|Error|FAILED" | tail -4 || echo "FAILED llama-3.2"
echo CORRECTEDMASTERDONE

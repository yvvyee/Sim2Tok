#!/usr/bin/env bash
# Realign projector/LoRA to a FROZEN geo-trained vision encoder, then check that
# reasoning recovers (perception stays at the encoder's 0.40 since it is frozen).
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
GEO=checkpoints/geo_encoders/inbox_ce_vision_encoder.bin

echo "### STEP1: realign projector+LoRA to frozen geo encoder (SFT on SDS) ###"
$PY gaa/train_gaa.py \
  --vision_model openai/clip-vit-large-patch14-336 \
  --llm_model /home/ywlee/SSD/Llama-3.1-8B-Instruct \
  --projector_path /home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en/projector.bin \
  --resume_lora_path /home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en \
  --data_path data/sds_gaa_en_train80.json \
  --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
  --output_dir checkpoints/realign_geo_train80 \
  --init_vision_ckpt $GEO --freeze_vision \
  --num_epochs 5 --batch_size 1 --grad_accum 2 --dtype bfloat16 \
  --logging_steps 50 --save_steps 1000 2>&1 | grep -oE "{.loss/[^}]*}|freeze_vision|vision init:" | tail -4

echo "### STEP2: reasoning after realign (geo encoder + realigned projector/LoRA) ###"
$PY gaa/eval_reasoning.py \
  --baseline_ckpt checkpoints/realign_geo_train80 \
  --vision_encoder_ckpt $GEO \
  --data_path data/sds_gaa_en_test20.json \
  --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
  --num_samples 3 --dtype bfloat16 2>&1 | grep -vE "UNEXPECTED|Loading weights|LOAD REPORT|^Key|^---|position_ids|Warning|tensorboard|not sharded|mean_resizing|pad_token"

echo "### STEP3: confirm perception unchanged (frozen geo encoder) on onshore ###"
$PY gaa/evaluate_attn.py \
  --data_path data/smd_vis_onshore_eval.json --image_dir /home/ywlee/SSD/smd_eval_frames \
  --gaa_ckpt $GEO --dtype float32 2>&1 | grep -E "stock-CLIP|GAA "
echo REALIGNDONE

#!/usr/bin/env bash
# Transfer-maximization experiment:
#  STEP1: train inbox_ce vision encoder (geo-only, decoupled from SFT) on mixed
#         sim+real data; eval Attn-IoU on SMD VIS_Onshore held-out; save encoder.
#  STEP2: assemble full VLM (geo encoder + baseline LoRA/projector); verify
#         reasoning preserved on held-out SDS samples vs the stock encoder.
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0

echo "### STEP1: train mixed geo encoder + onshore eval ###"
$PY gaa/loss_study.py \
  --data_path data/mixed_train_abs.json --image_dir / \
  --test_data_path data/smd_vis_onshore_eval.json --test_image_dir /home/ywlee/SSD/smd_eval_frames \
  --variants inbox_ce --steps 200 --lr 1e-5 --batch_size 8 --dtype float32 \
  --save_dir checkpoints/geo_encoders 2>&1 | grep -E "stock|inbox_ce|save|train="

echo "### STEP2: reasoning preservation (baseline vs geo encoder) ###"
$PY gaa/eval_reasoning.py \
  --baseline_ckpt /home/ywlee/SSD/checkpoints/clip_llama31_lora_marine_sds_lora_en \
  --vision_encoder_ckpt checkpoints/geo_encoders/inbox_ce_vision_encoder.bin \
  --data_path data/sds_gaa_en_test20.json \
  --image_dir /home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227 \
  --num_samples 3 --dtype bfloat16 2>&1 | grep -vE "UNEXPECTED|Loading weights|LOAD REPORT|^Key|^---|position_ids|Warning|tensorboard"
echo MAXDONE

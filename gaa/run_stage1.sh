#!/usr/bin/env bash
# Stage-1 (single GPU): rigorous multi-seed H1 (loss spec) + H4 (transfer) on the
# enlarged SMD VIS_Onshore held-out benchmark (288 frames), validated metrics
# (Attn-IoU area-matched, attn_in_box). 3 sources x {bg_l2, inbox_ce} x 3 seeds.
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
TEST="--test_data_path data/smd_onshore_big.json --test_image_dir /home/ywlee/SSD/smd_onshore_big"

run () {  # $1=tag $2=data_path $3=image_dir $4=variant $5=seed
  echo "@@@ TAG=$1 variant=$4 seed=$5"
  $PY gaa/loss_study.py --data_path "$2" --image_dir "$3" $TEST \
    --variants "$4" --seed "$5" --steps 200 --lr 1e-5 --batch_size 8 --dtype float32 \
    2>&1 | grep -E "^stock|^bg_l2|^inbox_ce"
}

for SEED in 0 1 2; do
  for V in bg_l2 inbox_ce; do
    run "sim"   data/sim_train_abs.json   /                                 "$V" "$SEED"
    run "real"  data/smd_realtrain.json   /home/ywlee/SSD/smd_realtrain_frames "$V" "$SEED"
    run "mixed" data/mixed_train_abs.json /                                 "$V" "$SEED"
  done
done
echo STAGE1DONE

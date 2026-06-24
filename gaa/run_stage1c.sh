#!/usr/bin/env bash
# RE-VERIFICATION after 20260227 bbox correction (2026-06-24).
# Sim/geopool geo-training masks are now CORRECT; eval set (SMD onshore) GT was
# always independent/correct. inbox_ce x {sim100, geopool264, geopool+real} x 3 seeds
# + bg_l2 control. Compare to prior (pre-fix) numbers.
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
TEST="--test_data_path data/smd_onshore_big.json --test_image_dir /home/ywlee/SSD/smd_onshore_big"

run () { echo "@@@ TAG=$1 variant=$3 seed=$4"
  $PY gaa/loss_study.py --data_path "$2" --image_dir / $TEST \
    --variants "$3" --seed "$4" --steps 200 --lr 1e-5 --batch_size 8 --dtype float32 \
    2>&1 | grep -E "^stock|^bg_l2|^inbox_ce"; }

for SEED in 0 1 2; do
  run sim       data/sim_train_abs.json     inbox_ce "$SEED"
  run geopool   data/sds_geopool.json       inbox_ce "$SEED"
  run gp_real   data/geopool_plus_real.json inbox_ce "$SEED"
  run geopool   data/sds_geopool.json       bg_l2    "$SEED"
done
echo STAGE1CDONE

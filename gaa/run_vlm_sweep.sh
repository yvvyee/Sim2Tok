#!/usr/bin/env bash
# Stage-1 multi-VLM zero-shot maritime behavioral sweep (11 families).
# Sequential (single GPU), resumable (skips models with existing json),
# failure-isolated (one model crash does not stop the sweep).
set -u
cd /home/ywlee/dev/TANGO2_main/Field_Test/SDS/VisionLanguageModel
PY=/home/ywlee/miniforge3/envs/eva/bin/python
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export HF_HOME=/home/ywlee/SSD/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
NUM=100

# ungated small->large first, then gated. llama-3.2 (11B) uses 4bit.
run () { echo "##### MODEL=$1 $(date '+%H:%M:%S') #####"
  $PY gaa/vlm_behavior_eval.py --model "$1" --num "$NUM" ${2:-} 2>&1 \
    | grep -E "behavior\]|Error|Traceback|CUDA out|OSError|GatedRepo" | tail -8 \
    || echo "FAILED $1"; }

for M in smolvlm qwen2-vl qwen2.5-vl llava-1.5 llava-next internvl3 idefics2 paligemma aya-vision gemma-3; do
  run "$M"
done
run llama-3.2 --four_bit
echo VLMSWEEPDONE

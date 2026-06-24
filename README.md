# GAA — Geometric Attention Alignment for Maritime VLMs

**English** · [한국어](README.ko.md)

Train-time **privileged-information attention alignment** for maritime
Vision-Language Models. GAA adds an auxiliary loss that uses ground-truth
vessel bounding boxes (available only during training) to steer the visual
encoder's attention onto vessel geometry instead of sea/background texture,
addressing the sim-to-real gap in maritime perception. The boxes are used
**only during training** (LUPI paradigm); inference is a standard VLM with no
extra cost.

This repository bundles the GAA implementation (`gaa/`) together with the
minimal base VLM framework it depends on (`model/`, `data/dataset.py`,
`train.py`) so it runs standalone.

---

## Highlights

- **Geometric attention loss with multiple variants** (`gaa/geo_losses.py`):
  - `bg_l2` — original background-suppression L2 (baseline; see note below).
  - `inbox_ce` — **recommended**: cross-entropy matching CLS attention to the
    foreground (vessel-mask) distribution. Actively concentrates attention on
    vessels rather than merely suppressing background.
  - `inbox_mass`, `fg_bg_margin` — alternative formulations.
- **Evaluation suite**: attention alignment (Attn-IoU), attention-grounded
  detection, texture-bias decomposition (vessel/sea/sky), loss-variant study,
  reasoning preservation.
- **Multi-VLM benchmark** (`gaa/vlm_zoo.py`, `gaa/vlm_behavior_eval.py`): a
  unified zero-shot maritime benchmark (vessel recall / hallucination / COLREG
  situation accuracy) across 11 open VLM families.

> **Note on `bg_l2`.** The original background-L2 loss is minimized by
> *spreading* attention thinly rather than concentrating it on vessels, and in
> our experiments does not improve (and can degrade) attention alignment. Use
> `--geo_variant inbox_ce`. The variants are kept for ablation/reproducibility.

---

## Repository structure

```
gaa/
  geometric_loss.py     # geometric mask (bbox -> patch grid) + original bg_l2 loss
  geo_losses.py         # loss variants: bg_l2 / inbox_mass / inbox_ce / fg_bg_margin
  gaa_dataset.py        # GAADataset / GAADataCollator (+ box-guided input crop)
  gaa_trainer.py        # GAATrainer: L_total = L_SFT + lambda * L_geo
  train_gaa.py          # training entry point (--geo_variant, --privileged_mode, ...)
  evaluate_attn.py      # Attn-IoU / attn_in_box / bg_ratio
  detect_eval.py        # attention-grounded detection: mAP@50 + FP/image
  texture_bias_eval.py  # H0: vessel%/sea%/sky% attention decomposition (uses SMD HorizonGT)
  loss_study.py         # vision-only loss-variant comparison (cross-domain via --test_data_path)
  eval_reasoning.py     # reasoning-preservation check (swap encoder, keep LoRA)
  attn_visualize.py     # per-sample attention heatmap panels
  vlm_zoo.py            # unified multi-VLM registry + load()/generate()
  vlm_behavior_eval.py  # zero-shot maritime behavioral benchmark
  sds_reformat.py       # SDS (20260227 format) -> GAA chat.json
  sds_old_to_gaa.py     # older SDS format (frame_*.png + input.csv) -> GAA json
  smd_to_gaa.py         # Singapore Maritime Dataset (videos + ObjectGT) -> GAA json
  compare_models.py     # qualitative baseline-vs-GAA generation
  run_*.sh              # experiment drivers
model/                  # base VLM framework (CLIP/SigLIP encoder, projector, LLaVA-style model)
data/dataset.py         # LLaVADataset + collator (GAA dataset extends these)
train.py                # base VLMTrainer (GAATrainer subclasses this)
scripts/                # DeepSpeed configs + training shell scripts + SDS prep
requirements.txt
```

---

## Installation

```bash
conda create -n gaa python=3.11 -y && conda activate gaa
# install a CUDA-matched PyTorch first (do NOT install torch via requirements.txt):
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
# optional, for 4-bit multi-VLM eval:
pip install bitsandbytes
```

---

## Data preparation

GAA training needs image–text pairs with **normalized `[x1,y1,x2,y2]` vessel
boxes** (`bboxes` field). Converters produce this from raw sources:

```bash
# SDS (current format): episode dirs with input_image.png + input_data.csv + advice text
python gaa/sds_reformat.py --dataset_root /path/to/SDS/20260227 --output data/sds_gaa_en.json --lang en

# Older SDS format (dataset*/frame_N.png + input.csv)
python gaa/sds_old_to_gaa.py --dataset_root /path/to/SDS/20250922 --out_json data/_old_20250922.json

# Singapore Maritime Dataset (real target): videos + ObjectGT(.mat) -> frames + boxes
python gaa/smd_to_gaa.py --smd_root /path/to/SMD --split VIS_Onshore --frames_per_video 8 \
    --out_json data/smd_onshore.json --out_images /path/to/smd_frames
```

JSON entry: `{"id", "image", "conversations"?, "bboxes": [[x1,y1,x2,y2], ...]}`
(normalized coords). Boxes are train-only privileged info.

---

## Training

```bash
python gaa/train_gaa.py \
    --vision_model openai/clip-vit-large-patch14-336 \
    --llm_model /path/to/Llama-3.1-8B-Instruct \
    --projector_path /path/to/projector.bin \
    --resume_lora_path /path/to/baseline_lora \
    --data_path data/sds_gaa_en.json \
    --image_dir /path/to/images \
    --output_dir checkpoints/gaa_inbox_ce \
    --geo_variant inbox_ce --geo_loss_weight 0.5 --geo_tau 0.5
```

Key flags:

| flag | meaning |
|------|---------|
| `--geo_variant` | `inbox_ce` (recommended) / `bg_l2` / `inbox_mass` / `fg_bg_margin` |
| `--geo_loss_weight` | lambda for `L_total = L_SFT + lambda * L_geo` |
| `--privileged_mode` | `attn_loss` (GAA) or `input_crop` (box-guided crop control) |
| `--init_vision_ckpt` / `--freeze_vision` | load a geo-trained encoder and realign projector/LoRA only |

The GAA-trained CLIP weights are saved to `vision_encoder.bin` in the output
dir (required so the attention alignment is preserved at evaluation time).

Multi-GPU: `bash scripts/train_lora_gaa.sh` (DeepSpeed; see `scripts/zero*.json`).

---

## Evaluation

```bash
# Attention alignment (encoder-only; stock vs trained encoders)
python gaa/evaluate_attn.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --gaa_ckpt checkpoints/gaa_inbox_ce/vision_encoder.bin

# Loss-variant study (vision-only; cross-domain transfer via --test_data_path)
python gaa/loss_study.py --data_path data/train.json --image_dir / \
    --test_data_path data/smd_onshore.json --test_image_dir /path/to/frames \
    --variants bg_l2,inbox_ce --steps 200

# Texture-bias decomposition (H0): vessel% vs sea% vs sky% (needs SMD HorizonGT)
python gaa/texture_bias_eval.py --smd_root /path/to/SMD --split VIS_Onshore \
    --ckpts stock=stock,GAA=checkpoints/gaa_inbox_ce/vision_encoder.bin

# Attention-grounded detection (mAP@50 + FP/image)
python gaa/detect_eval.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --ckpts stock,checkpoints/gaa_inbox_ce/vision_encoder.bin

# Per-sample attention heatmaps
python gaa/attn_visualize.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --out_dir viz/ --ckpts "stock=stock,GAA=checkpoints/gaa_inbox_ce/vision_encoder.bin"
```

---

## Multi-VLM maritime benchmark

Zero-shot comparison across 11 open VLM families (Qwen2-VL, Qwen2.5-VL,
LLaVA-1.5, LLaVA-NeXT, InternVL3, Idefics2, SmolVLM, PaliGemma, Llama-3.2-Vision,
Aya-Vision, Gemma-3). Gated models require an accepted HF license + token.

```bash
python gaa/vlm_behavior_eval.py --model qwen2.5-vl --num 100   # one model
bash gaa/run_vlm_sweep.sh                                      # full sweep (sequential, resumable)
```

Metrics: `vessel_recall` (detection), `halluc_rate` (POPE-style false-positive
on absent objects), `colreg_acc` (head-on / crossing / overtaking / safe).

---

## License

See [LICENSE](LICENSE).

# GAA — 해양 VLM을 위한 기하 어텐션 정렬 (Geometric Attention Alignment)

[English](README.md) · **한국어**

해양 Vision-Language Model을 위한 **학습 시 특권 정보(privileged information) 기반 어텐션 정렬** 기법입니다.
GAA는 학습 시점에만 존재하는 선박 바운딩 박스를 이용해, 시각 인코더의 어텐션이
바다/배경 텍스처가 아니라 **선박 기하**에 향하도록 유도하는 보조 손실을 추가합니다.
이는 해양 인식의 sim-to-real 격차를 다룹니다. 박스는 **학습에만** 쓰이며(LUPI 패러다임),
추론은 추가 비용 없는 표준 VLM입니다.

이 저장소는 GAA 구현(`gaa/`)과 함께, 의존하는 최소 베이스 VLM 프레임워크
(`model/`, `data/dataset.py`, `train.py`)를 포함해 단독 실행이 가능합니다.

---

## 핵심 기능

- **기하 어텐션 손실의 다중 변형** (`gaa/geo_losses.py`):
  - `bg_l2` — 원본 배경억제 L2 (기준선; 아래 주의 참고).
  - `inbox_ce` — **권장**: CLS 어텐션을 전경(선박 마스크) 분포에 매칭하는
    cross-entropy. 배경을 단순 억제하는 대신 선박에 어텐션을 **적극 집중**시킵니다.
  - `inbox_mass`, `fg_bg_margin` — 대안 정식화.
- **평가 도구 모음**: 어텐션 정렬(Attn-IoU), 어텐션 기반 검출, 텍스처 편향 분해
  (선박/바다/하늘), 손실 변형 비교 연구, reasoning 보존 검증.
- **다중 VLM 벤치마크** (`gaa/vlm_zoo.py`, `gaa/vlm_behavior_eval.py`): 11개 공개
  VLM 패밀리에 대한 통합 zero-shot 해양 벤치마크(선박 recall / 환각율 / COLREG
  상황분류 정확도).

> **`bg_l2`에 대한 주의.** 원본 배경 L2 손실은 어텐션을 선박에 집중시키기보다
> *고르게 분산*시킬 때 최소화되며, 실험상 어텐션 정렬을 개선하지 못하거나 오히려
> 악화시킵니다. `--geo_variant inbox_ce`를 사용하세요. 변형들은 ablation/재현을
> 위해 유지됩니다.

---

## 저장소 구조

```
gaa/
  geometric_loss.py     # 기하 마스크(bbox -> 패치 그리드) + 원본 bg_l2 손실
  geo_losses.py         # 손실 변형: bg_l2 / inbox_mass / inbox_ce / fg_bg_margin
  gaa_dataset.py        # GAADataset / GAADataCollator (+ 박스 기반 입력 crop)
  gaa_trainer.py        # GAATrainer: L_total = L_SFT + lambda * L_geo
  train_gaa.py          # 학습 진입점 (--geo_variant, --privileged_mode, ...)
  evaluate_attn.py      # Attn-IoU / attn_in_box / bg_ratio
  detect_eval.py        # 어텐션 기반 검출: mAP@50 + FP/이미지
  texture_bias_eval.py  # H0: 선박%/바다%/하늘% 어텐션 분해 (SMD HorizonGT 사용)
  loss_study.py         # vision-only 손실 변형 비교 (--test_data_path로 교차도메인)
  eval_reasoning.py     # reasoning 보존 검증 (인코더 교체, LoRA 유지)
  attn_visualize.py     # 샘플별 어텐션 히트맵 패널
  vlm_zoo.py            # 통합 다중 VLM 레지스트리 + load()/generate()
  vlm_behavior_eval.py  # zero-shot 해양 행동 벤치마크
  sds_reformat.py       # SDS(20260227 형식) -> GAA chat.json
  sds_old_to_gaa.py     # 구 SDS 형식(frame_*.png + input.csv) -> GAA json
  smd_to_gaa.py         # Singapore Maritime Dataset(영상 + ObjectGT) -> GAA json
  compare_models.py     # baseline vs GAA 정성 생성 비교
  run_*.sh              # 실험 구동 스크립트
model/                  # 베이스 VLM 프레임워크 (CLIP/SigLIP 인코더, projector, LLaVA형 모델)
data/dataset.py         # LLaVADataset + collator (GAA 데이터셋이 상속)
train.py                # 베이스 VLMTrainer (GAATrainer가 상속)
scripts/                # DeepSpeed 설정 + 학습 셸 스크립트 + SDS 준비
requirements.txt
```

---

## 설치

```bash
conda create -n gaa python=3.11 -y && conda activate gaa
# CUDA에 맞는 PyTorch를 먼저 설치하세요 (requirements.txt로 torch 설치 금지):
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
# 4-bit 다중 VLM 평가용 (선택):
pip install bitsandbytes
```

---

## 데이터 준비

GAA 학습에는 **정규화된 `[x1,y1,x2,y2]` 선박 박스**(`bboxes` 필드)가 있는
이미지-텍스트 쌍이 필요합니다. 변환기로 원천 데이터에서 생성합니다:

```bash
# SDS (현재 형식): input_image.png + input_data.csv + 조언 텍스트가 있는 에피소드 디렉토리
python gaa/sds_reformat.py --dataset_root /path/to/SDS/20260227 --output data/sds_gaa_en.json --lang en

# 구 SDS 형식 (dataset*/frame_N.png + input.csv)
python gaa/sds_old_to_gaa.py --dataset_root /path/to/SDS/20250922 --out_json data/_old_20250922.json

# Singapore Maritime Dataset (실제 타깃): 영상 + ObjectGT(.mat) -> 프레임 + 박스
python gaa/smd_to_gaa.py --smd_root /path/to/SMD --split VIS_Onshore --frames_per_video 8 \
    --out_json data/smd_onshore.json --out_images /path/to/smd_frames
```

JSON 항목: `{"id", "image", "conversations"?, "bboxes": [[x1,y1,x2,y2], ...]}`
(정규화 좌표). 박스는 학습 전용 특권 정보입니다.

---

## 학습

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

주요 인수:

| 인수 | 의미 |
|------|------|
| `--geo_variant` | `inbox_ce`(권장) / `bg_l2` / `inbox_mass` / `fg_bg_margin` |
| `--geo_loss_weight` | `L_total = L_SFT + lambda * L_geo`의 lambda |
| `--privileged_mode` | `attn_loss`(GAA) 또는 `input_crop`(박스 기반 crop 통제군) |
| `--init_vision_ckpt` / `--freeze_vision` | geo-학습 인코더를 로드하고 projector/LoRA만 재정렬 |

GAA로 학습된 CLIP 가중치는 출력 디렉토리의 `vision_encoder.bin`에 저장됩니다
(평가 시점에 어텐션 정렬이 보존되도록 필수).

멀티 GPU: `bash scripts/train_lora_gaa.sh` (DeepSpeed; `scripts/zero*.json` 참고).

---

## 평가

```bash
# 어텐션 정렬 (인코더 단독; stock vs 학습 인코더)
python gaa/evaluate_attn.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --gaa_ckpt checkpoints/gaa_inbox_ce/vision_encoder.bin

# 손실 변형 연구 (vision-only; --test_data_path로 교차도메인 전이)
python gaa/loss_study.py --data_path data/train.json --image_dir / \
    --test_data_path data/smd_onshore.json --test_image_dir /path/to/frames \
    --variants bg_l2,inbox_ce --steps 200

# 텍스처 편향 분해 (H0): 선박% vs 바다% vs 하늘% (SMD HorizonGT 필요)
python gaa/texture_bias_eval.py --smd_root /path/to/SMD --split VIS_Onshore \
    --ckpts stock=stock,GAA=checkpoints/gaa_inbox_ce/vision_encoder.bin

# 어텐션 기반 검출 (mAP@50 + FP/이미지)
python gaa/detect_eval.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --ckpts stock,checkpoints/gaa_inbox_ce/vision_encoder.bin

# 샘플별 어텐션 히트맵
python gaa/attn_visualize.py --data_path data/smd_onshore.json --image_dir /path/to/frames \
    --out_dir viz/ --ckpts "stock=stock,GAA=checkpoints/gaa_inbox_ce/vision_encoder.bin"
```

---

## 다중 VLM 해양 벤치마크

11개 공개 VLM 패밀리(Qwen2-VL, Qwen2.5-VL, LLaVA-1.5, LLaVA-NeXT, InternVL3,
Idefics2, SmolVLM, PaliGemma, Llama-3.2-Vision, Aya-Vision, Gemma-3)에 대한
zero-shot 비교. 게이트 모델은 HF 라이선스 동의 + 토큰이 필요합니다.

```bash
python gaa/vlm_behavior_eval.py --model qwen2.5-vl --num 100   # 단일 모델
bash gaa/run_vlm_sweep.sh                                      # 전체 sweep (순차, resumable)
```

지표: `vessel_recall`(검출), `halluc_rate`(absent object에 대한 POPE식 오탐),
`colreg_acc`(head-on / crossing / overtaking / safe).

---

## 라이선스

[LICENSE](LICENSE) 참고.

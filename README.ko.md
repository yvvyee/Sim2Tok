# Sim2Tok

[English](README.md) · 한국어

논문 "Where Simulator Geometry Should Enter a Vision-Language Encoder: Token Supervision Narrows the Maritime Sim-to-Real Gap"(이영운, 조창식, 김병규, WACV 2027 Maritime Computer Vision (MaCVi) 워크숍)의 코드입니다.

이 논문은 실제 이미지를 학습, 모델 선택, 임계값 조정에 전혀 쓰지 않을 때, 시뮬레이터의 선박 기하 정보가 LLaVA 계열 시각 언어 모델(Vision-Language Model, VLM)의 CLIP 시각 인코더 어디를 감독해야 하는지 묻습니다. 두 감독 방식을 비교하며, 둘 다 시뮬레이터가 제공하는 선박 박스만 사용합니다.

- 기하 어텐션 정렬(Geometric Attention Alignment, GAA)은 24층 클래스 토큰 어텐션을 선박 박스에 맞춥니다. VLM은 클래스 토큰을 버리므로 이 값은 언어 모델에 전달되지 않습니다.
- 토큰 점유율(token occupancy)은 VLM 프로젝터의 입력인 23층 패치 토큰마다 헤드를 붙여, 토큰 셀에서 선박이 차지하는 비율과 선박 중심이 셀 안에 있는지를 예측합니다. 헤드는 학습 후 버리므로 인코더의 인터페이스와 추론 비용은 그대로입니다.

모든 인코더는 AIVN-SDS 시뮬레이터 데이터셋의 학습 분할로만 학습합니다. 다섯 개의 실제 데이터셋은 평가에만 사용합니다.

## 저장소 구성

```
additional/            논문의 파이프라인 (학습, 프로브, 평가, 데이터 준비)
  run_rev_stage.sh     논문의 모든 실험을 실행하는 단일 진입점 (블록 A~I)
  tables/              저장된 출력에서 논문의 표와 그림을 만드는 스크립트
gaa/                   GAA 손실 (geo_losses.py, geometric_loss.py)과 이전 GAA 실험
model/, data/, train.py  LLaVA 계열 VLM (CLIP 인코더, 프로젝터, LoRA를 적용한 Llama-3.1-8B)
scripts/               DeepSpeed 설정과 이전 학습 스크립트
```

`gaa/README.md`는 이전 GAA 프레임워크를 설명합니다. `additional/run_token_stage.sh`, `run_det_stage.sh`, `run_det_reeval.sh`, `run_verify_*.sh`는 개정 전 프로토콜(3,000 스텝, 학습 전용 분할 없음)의 스크립트이며 논문의 수치를 만들지 않습니다.

## 환경

실험은 Ubuntu 24.04.5 LTS에서 작업마다 NVIDIA A100-PCIE 40 GB GPU 4장을 사용했고, `torchrun`과 분산 데이터 병렬로 실행했습니다. SMD 재평가는 NVIDIA A100 80 GB PCIe GPU 2장이 있는 두 번째 서버에서 같은 소프트웨어로 실행했습니다.

- Python 3.11.15
- CUDA 12.8과 cuDNN 9.10.2용 PyTorch 2.10.0, torchvision 0.25.0
- Transformers 5.3.0, PEFT 0.18.1, Accelerate 1.13.0, NumPy 2.4.3 및 `requirements.txt`의 나머지 고정 버전
- 일부 스크립트는 `requirements.txt`에 없는 패키지를 사용합니다. `ultralytics` 8.4.142(YOLOv8s 기준 검출기), `opencv-python`(`make_cue_sets.py`), `scipy`, `matplotlib`, `tensorboard`입니다.

```bash
conda create -n sim2tok python=3.11 -y && conda activate sim2tok
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install ultralytics==8.4.142 opencv-python scipy matplotlib tensorboard
```

경로, conda 환경, GPU 구성은 `additional/run_rev_stage.sh`(`ROOT`, `BIN`, `REAL`, `LLM`, `CC3M`, `CUDA_VISIBLE_DEVICES`)와 `additional/config.py`에 적혀 있습니다. `config.HOST_POLICY`는 호스트별로 허용된 GPU를 적어 둡니다(`tg-a40g`는 2, 3, 4, 5, `tg-a80g`는 0, 1). `config.assert_policy_or_die`는 그 밖의 GPU에서 실행을 중단합니다. 다른 장비에서 실행하려면 이 값들을 먼저 수정해야 합니다.

## 데이터

학습 데이터

- Unity3D 시뮬레이터로 렌더링한 선박 조우 이미지 데이터셋 AIVN-SDS는 https://huggingface.co/datasets/ETRI-TANGO/AIVN-SDS 에서 받을 수 있습니다. 실험은 `data/ok_aivn_10000.json`에 있는 프레임 10,000장을 사용합니다. 이 목록은 이 저장소의 스크립트로 생성되지 않습니다.

실제 평가 데이터 (각 원 출처에서 내려받음)

| 데이터셋 | 선박으로 세는 클래스 | 변환기 |
|---|---|---|
| Singapore Maritime Dataset (SMD), VIS_Onshore와 NIR | ObjectGT 클래스 1, 3, 4, 5, 6, 7 (Ferry, Vessel/ship, Speed boat, Boat, Kayak, Sail boat). 부표, 새 또는 비행기, "Other"는 `other_boxes`로 따로 저장하며 선박으로 세지 않음 | `additional/smd_full_to_gaa.py` |
| Maritime Collision Avoidance Dataset (MCAD), 가시광과 열화상 | power-driven vessel, sailboat | `additional/real_eval_to_gaa.py` |
| Lakes, Rivers and Seas (LaRS), 학습과 검증 분할 | Boat/ship, Row boats | `additional/real_eval_to_gaa.py` |
| SeaDronesSee ODv2, 학습과 검증 분할 | boat, jetski | `additional/real_eval_to_gaa.py` |

변환기는 이미지마다 bootstrap 재표집 단위도 기록합니다. SMD는 비디오, LaRS는 시퀀스, SeaDronesSee는 원본 비디오, MCAD는 이미지 자체가 단위입니다.

준비 단계 (한 번만, 이 순서로 실행)

1. `additional/aivn_to_gaa.py`는 AIVN-SDS의 프레임, 상황 설명, 항해 조언을 JSON으로 변환합니다.
2. `additional/make_sim_crops.py`는 선박 크롭과 선박 없는 크롭, 그리고 프레임 기준 90/10 분할(학습 25,529장, 검증 2,842장)을 만듭니다.
3. `additional/build_nav_state.py`는 `train_token_ddp.py`가 읽는 항해 상태 타깃을 만듭니다.
4. `additional/make_shuffled_boxes.py`는 박스 순열 대조군용으로 이미지 간에 섞은 박스를 만듭니다.
5. `additional/real_eval_to_gaa.py`와 `additional/smd_full_to_gaa.py`는 실제 평가용 JSON을 만듭니다.
6. `additional/make_cue_sets.py`는 시뮬레이터 검증 이미지의 텍스처 억제본과 블록 순열본을 만듭니다.
7. `additional/build_vlm_sft_v2.py`는 VLM 실험용 시뮬레이터 전용 지시 데이터를 만듭니다.
8. `python -m additional.yolo_ref export`는 기준 검출기용 YOLO 데이터셋을 만듭니다.

## 학습과 평가

```bash
BLOCKS="A B C D E F G H I" SEEDS="0 1 2" bash additional/run_rev_stage.sh
```

각 블록은 모든 시드를 마친 뒤 다음 블록으로 넘어갑니다.

| 블록 | 내용 |
|---|---|
| A | 토큰 점유율, 원래 설정의 GAA, 이미지와 스텝을 토큰 점유율에 맞춘 GAA, 그리고 이 인코더들과 원본 CLIP 인코더의 고정 인코더 프로브 |
| B | 23층 클래스 토큰 어텐션에 적용한 GAA |
| C | 토큰 점유율과 맞춘 GAA의 도메인 무작위화, 이미지 수준 대조군 |
| D | 224 픽셀 CLIP ViT-B/16 |
| E | 시뮬레이터 이미지로 학습한 YOLOv8s 기준 검출기 |
| F | 원본, 맞춘 GAA, 토큰 점유율 인코더 기반 VLM |
| G | 384 픽셀 SigLIP so400m/14 |
| H | 검출 프로브, 어텐션 겹침, 박스 순열 GAA 대조군 |
| I | 항해 상태 타깃과 부재 타깃 |

주요 설정

| 구성 요소 | 설정 | 코드 |
|---|---|---|
| 토큰 점유율 | 23층 토큰, 4,000 스텝, 전역 배치 24, 코사인 스케줄 AdamW, 인코더 lr 1e-5, 헤드 lr 1e-3, weight decay 0.01, 중심 손실 양성 가중치 50, 그래디언트 클리핑 1.0 | `train_token_ddp.py` |
| GAA | 헤드 평균 24층 CLS 어텐션에 대한 `inbox_ce` 손실, 커버리지 임계값 0.01, lr 1e-5. 원래 설정은 프레임 9,000장 5 에폭(1,875 스텝). 맞춘 설정은 프레임과 선박 크롭으로 4,000 스텝 | `train_geo_ddp.py`, `gaa/geo_losses.py` |
| 고정 인코더 프로브 | 토큰별 헤드, 2,000 스텝, 전역 배치 48, 헤드 lr 1e-3, 시뮬레이터 학습 이미지 25,529장으로 학습, 개수 임계값은 시뮬레이터 검증으로 선택 | `eval_token_probe.py`, `probe_compare.py` |
| 어텐션 겹침 | 헤드 평균 24층 CLS 어텐션의 top-k 셀과 박스가 절반 넘게 덮는 셀의 겹침 | `eval_attn_mp.py` |

## 선박 클래스만 쓴 SMD 재평가

처음 SMD 변환은 ObjectGT의 모든 박스를 선박으로 셌습니다. 논문의 SMD 결과는 선박 클래스만 쓴 값입니다. 이를 위해 다시 학습한 모델은 없습니다. `additional/run_smd_reeval.sh`는 SMD를 다시 변환하고, 이전 라벨로 저장된 SMD 출력이 재현되는지 확인한 다음, 저장된 모든 인코더를 저장된 프로브, 검출 또는 YOLO 헤드로 새 라벨에서 평가합니다(`--init_head`, `--eval_only`, `--skip_sim_val`). `additional/yesno_rescore.py`는 저장된 VLM 답변을 새 라벨로 다시 채점합니다. 이미지와 질문은 라벨에 따라 달라지지 않기 때문입니다. 선박 클래스 기준으로 SMD 중앙 크롭에는 선박 122,917척이 있고, 선박이 없는 크롭은 124장입니다.

## 출력에서 논문 결과까지

`additional/tables/`의 스크립트는 `$GAA_ROOT/results` 아래에 저장된 출력을 읽습니다. 이때 SMD 출력은 `run_smd_reeval.sh`의 출력입니다. `GAA_ROOT`의 기본값은 논문에 쓴 디렉터리 `/mnt/ssd1/yvvyee/gaa_additional`입니다. 저장소 최상위에서 실행합니다. 예를 들면 `python -m additional.tables.pooled_boot main`입니다.

| 논문 결과 | 스크립트 |
|---|---|
| Table 1, 데이터셋 통계 | `real_eval_to_gaa.py`와 `smd_full_to_gaa.py`가 쓰는 요약 파일 |
| Table 2, 프로브 결과 (세 시드의 평균과 표준편차) | `probe_compare.py` 출력의 `per_encoder` 항목 |
| Figure 2, 시뮬레이터-실제 격차 | `tables/fig_gap_rev.py` |
| Table 3과 5, 95% 구간을 포함한 쌍별 차이 | `tables/pooled_boot.py main` |
| 원본 CLIP 인코더와의 비교 | `tables/pooled_boot.py stock` |
| Table 4, 어텐션 겹침 | `eval_attn_mp.py`, `tables/attagg.py` |
| 보충자료, 시드별 쌍별 차이 | `probe_compare.py`, `tables/pagg.py` |
| 보충자료, 검출 프로브 | `eval_det_probe.py`, `det_compare.py`, `tables/dagg.py` |
| 보충자료, YOLOv8s 기준 검출기 | `yolo_ref.py`, `tables/yolo_ap.py` |
| 보충자료, 텍스처와 형태 단서 실험 | `make_cue_sets.py`, `tables/cue_sub.py` |
| 보충자료, VLM 실험 | `build_vlm_sft_v2.py`, `train.py`, `yesno_eval.py`, `yesno_rescore.py` (SMD), `verify_s2_compare.py`, `tables/vlmagg.py`, `tables/vlmtab.py` |
| 보충자료, 예측 점유율 지도 | `fig_occupancy.py --smd_json` |

`tables/pooled_boot.py`는 쌍별 차이를 세 시드 차이의 평균으로 계산합니다. 95% 구간은 재표집 단위에 대한 bootstrap 1,000회로 구하며, 매 회차의 재표집을 세 시드에 똑같이 적용합니다.

## 인용

```bibtex
@inproceedings{Lee2027Sim2Tok,
  author    = {Lee, Young-Woon and Cho, Changsik and Kim, Byung-Gyu},
  title     = {Where Simulator Geometry Should Enter a Vision-Language Encoder: Token Supervision Narrows the Maritime Sim-to-Real Gap},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## 라이선스

[LICENSE](LICENSE)를 참고하세요.

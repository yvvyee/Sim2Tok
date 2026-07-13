"""
geo_reason_eval.py — Reasoning accuracy before/after geo encoder swap

각 VLM에 대해:
  (1) 원본 인코더로 COLREG 추론 정확도 측정 (baseline)
  (2) geo 학습된 인코더로 교체 후 재측정 (+GAA)
  (3) vessel_recall, halluc_rate, colreg_acc 비교 출력

Usage:
  python gaa/geo_reason_eval.py --model smolvlm --num 100
  python gaa/geo_reason_eval.py --all --num 100
"""
import sys
import os
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from gaa import vlm_zoo
from gaa.vlm_behavior_eval import gt_situation, gt_vessel_count, yes, parse_sit
from gaa.geo_validate_vlms import get_vision_tower, REGISTRY


GEO_CKPT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints"
)
SDS_DATA = "/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227"


# ── Vision encoder swap ───────────────────────────────────────────────────────

def swap_vision_encoder(model, model_name: str, geo_ckpt: str) -> bool:
    """Load geo-trained weights into the VLM's vision tower in-place.
    Returns True if successful."""
    family = REGISTRY[model_name][1]
    try:
        enc = get_vision_tower(model, family)
    except Exception as e:
        print(f"  [WARN] vision tower extraction failed: {e}")
        return False

    if not os.path.exists(geo_ckpt):
        print(f"  [WARN] geo ckpt not found: {geo_ckpt}")
        return False

    state = torch.load(geo_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    n_matched = len(state) - len(unexpected)
    print(f"  [geo swap] {n_matched}/{len(state)} tensors loaded "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if n_matched == 0:
        print("  [WARN] no tensors matched — swap had no effect")
        return False
    return True


def restore_vision_encoder(model, model_name: str, orig_state: dict) -> None:
    """Restore original vision encoder weights."""
    family = REGISTRY[model_name][1]
    try:
        enc = get_vision_tower(model, family)
        enc.load_state_dict(orig_state, strict=False)
    except Exception as e:
        print(f"  [WARN] restore failed: {e}")


# ── Single-run evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, proc, episodes: list) -> dict:
    rec, hal, sit_ok, sit_tot, per = [], [], 0, 0, []
    for ep in episodes:
        img_p = os.path.join(ep, "input_image.png")
        if not os.path.exists(img_p):
            continue
        img = Image.open(img_p).convert("RGB")
        gt_sit = gt_situation(ep)
        gt_n   = gt_vessel_count(ep)

        a_v = vlm_zoo.generate(model, proc, img,
            "Is there a vessel or ship in this image? Answer yes or no.", 16)
        if gt_n is None or gt_n > 0:
            rec.append(1.0 if yes(a_v) else 0.0)

        for obj in ("airplane", "car", "building"):
            a = vlm_zoo.generate(model, proc, img,
                f"Is there a {obj} in this image? Answer yes or no.", 16)
            hal.append(1.0 if yes(a) else 0.0)

        a_s = vlm_zoo.generate(model, proc, img,
            "Look at the vessels. Classify the COLREG encounter situation as exactly "
            "one word: head-on, crossing, overtaking, or safe. Answer with only that "
            "one word.", 24)
        ps = parse_sit(a_s)
        sit_tot += 1
        sit_ok  += int(ps == gt_sit)
        per.append({"ep": os.path.basename(ep), "gt_sit": gt_sit,
                    "pred_sit": ps, "raw_sit": a_s[:80]})

    def mean(lst): return round(sum(lst) / max(len(lst), 1), 4)
    return {
        "vessel_recall": mean(rec),
        "halluc_rate":   mean(hal),
        "colreg_acc":    round(sit_ok / max(sit_tot, 1), 4),
        "n": sit_tot,
        "per_sample": per,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, num: int, out_dir: str,
                   four_bit: bool = False) -> dict:
    geo_ckpt = os.path.join(GEO_CKPT_BASE, f"geo_{model_name}", "vision_encoder.bin")
    out_path = os.path.join(out_dir, f"{model_name}.json")

    print(f"\n{'='*60}")
    print(f"  [{model_name}]")
    print(f"  geo_ckpt: {geo_ckpt}")
    print(f"{'='*60}")

    # Episodes
    eps = sorted(d for d in glob.glob(os.path.join(SDS_DATA, "*"))
                 if os.path.isdir(d))[:num]
    print(f"  에피소드: {len(eps)}")

    # Load model (local_files_only fallback for gated models)
    print("  모델 로딩 ...")
    os.environ.setdefault("HF_HOME", "/home/ywlee/SSD/hf_cache")
    repo = vlm_zoo.REGISTRY[model_name][0]
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model, proc = None, None
    for local_only in [False, True]:
        try:
            proc  = AutoProcessor.from_pretrained(repo, local_files_only=local_only)
            model = AutoModelForImageTextToText.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map="cuda",
                local_files_only=local_only).eval()
            break
        except Exception as e:
            if local_only:
                print(f"  [ERROR] load failed: {e}")
                return {"model": model_name, "error": str(e)}
    if model is None:
        return {"model": model_name, "error": "load failed"}

    # ── Baseline eval ─────────────────────────────────────────────────────────
    print("\n  [1/2] Baseline (원본 인코더) 평가 ...")
    baseline = run_eval(model, proc, eps)
    print(f"    vessel_recall={baseline['vessel_recall']:.4f}  "
          f"halluc_rate={baseline['halluc_rate']:.4f}  "
          f"colreg_acc={baseline['colreg_acc']:.4f}")

    # Save original vision encoder weights for restore
    family = REGISTRY[model_name][1]
    try:
        enc = get_vision_tower(model, family)
        orig_state = {k: v.clone() for k, v in enc.state_dict().items()}
    except Exception:
        orig_state = None

    # ── Geo encoder swap ──────────────────────────────────────────────────────
    print("\n  [2/2] +GAA (geo 인코더 교체) 평가 ...")
    swapped = swap_vision_encoder(model, model_name, geo_ckpt)
    if not swapped:
        print("  [SKIP] geo swap 실패 — baseline만 기록")
        geo_result = None
    else:
        geo_result = run_eval(model, proc, eps)
        print(f"    vessel_recall={geo_result['vessel_recall']:.4f}  "
              f"halluc_rate={geo_result['halluc_rate']:.4f}  "
              f"colreg_acc={geo_result['colreg_acc']:.4f}")
        if orig_state:
            restore_vision_encoder(model, model_name, orig_state)

    # ── Summary ───────────────────────────────────────────────────────────────
    result = {
        "model": model_name,
        "n": baseline["n"],
        "baseline": baseline,
        "geo": geo_result,
    }
    if geo_result:
        delta = geo_result["colreg_acc"] - baseline["colreg_acc"]
        result["colreg_delta"] = round(delta, 4)
        print(f"\n  colreg_acc: {baseline['colreg_acc']:.4f} → "
              f"{geo_result['colreg_acc']:.4f}  (Δ{delta:+.4f})")

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    p = argparse.ArgumentParser("GAA Reasoning Accuracy Eval")
    p.add_argument("--model", default=None, choices=list(vlm_zoo.REGISTRY))
    p.add_argument("--all",   action="store_true", help="10개 모델 전부 순차 실행")
    p.add_argument("--num",   type=int, default=100, help="평가 에피소드 수")
    p.add_argument("--out_dir", default="results/geo_reason")
    p.add_argument("--four_bit", action="store_true")
    args = p.parse_args()

    # llama-3.2는 geo swap 불가 → 원본 평가만 가능하므로 제외
    SKIP = {"llama-3.2"}
    if args.all:
        models = [m for m in vlm_zoo.REGISTRY if m not in SKIP]
    elif args.model:
        models = [args.model]
    else:
        p.error("--model 또는 --all 중 하나를 지정하세요.")
        return

    all_results = {}
    for name in models:
        r = evaluate_model(name, args.num, args.out_dir, args.four_bit)
        all_results[name] = r

    # ── 전체 요약 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  10-VLM 추론 정확도 Before/After GAA")
    print(f"{'='*72}")
    hdr = f"{'model':<14}{'colreg(base)':>14}{'colreg(+geo)':>14}{'delta':>8}{'recall(+geo)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in all_results.items():
        base_acc = r["baseline"]["colreg_acc"]
        if r["geo"]:
            geo_acc  = r["geo"]["colreg_acc"]
            delta    = r.get("colreg_delta", 0)
            geo_rec  = r["geo"]["vessel_recall"]
            print(f"{name:<14}{base_acc:>14.4f}{geo_acc:>14.4f}{delta:>+8.4f}{geo_rec:>14.4f}")
        else:
            print(f"{name:<14}{base_acc:>14.4f}{'N/A':>14}{'N/A':>8}{'N/A':>14}")

    # 전체 JSON 저장
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n전체 요약: {summary_path}")


if __name__ == "__main__":
    main()

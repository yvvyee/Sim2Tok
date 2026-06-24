"""
Zero-shot maritime behavioral benchmark across VLM families (Stage 1).

Per model, over the 20260227 SDS set (has GT navigation text + bboxes), measures:
  - vessel_recall : P(model says "yes" to "is there a vessel?") on vessel-present imgs (higher=better)
  - halluc_rate   : P(model says "yes" to absent objects: airplane/car/building) (lower=better)
  - colreg_acc    : situation classification accuracy vs GT (head-on/crossing/overtaking/safe)
Addresses reviewers' "missing mainstream-VLM comparison".

Writes results/vlm_behavior/<model>.json. Resumable (skips if output exists).

Usage:
  HF_TOKEN=... HF_HOME=/home/ywlee/SSD/hf_cache \
  python gaa/vlm_behavior_eval.py --model smolvlm --num 100
"""
import sys, os, json, re, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
import pandas as pd
from gaa import vlm_zoo

DATA = "/home/ywlee/dev/TANGO2_main/Field_Test/SDS/dataset/20260227"
ABSENT = ["airplane", "car", "building"]
SIT_KEYS = ["head-on", "crossing", "overtaking"]  # else -> "safe"


def gt_situation(ep_dir):
    txt = ""
    for f in ("output_describe_en.txt", "output_advice_en.txt"):
        p = os.path.join(ep_dir, f)
        if os.path.exists(p):
            txt += open(p, encoding="utf-8").read().lower() + " "
    for k in SIT_KEYS:
        if k in txt:
            return k
    return "safe"


def gt_vessel_count(ep_dir):
    csv = os.path.join(ep_dir, "input_data.csv")
    try:
        df = pd.read_csv(csv)
        c_my = next((c for c in df.columns if c.lower() == "my_ship"), None)
        c_bw = next((c for c in df.columns if c.lower() == "bbox_width"), None)
        n = 0
        for _, r in df.iterrows():
            if c_my is not None and int(float(r[c_my])) == 1:
                continue
            if c_bw is None or float(r[c_bw]) > 0:
                n += 1
        return n
    except Exception:
        return None


def yes(ans):
    a = ans.strip().lower()
    return a.startswith("yes") or a.startswith("y ") or " yes" in a[:12]


def parse_sit(ans):
    a = ans.strip().lower()
    # tolerant keyword matching (hyphen/space/verb variants)
    if "head-on" in a or "head on" in a or "headon" in a:
        return "head-on"
    if "overtak" in a:           # overtaking/overtake/overtaken
        return "overtaking"
    if "crossing" in a or "cross " in a or a.startswith("cross"):
        return "crossing"
    if any(s in a for s in ("no risk", "safe", "none", "no collision", "no encounter", "clear")):
        return "safe"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(vlm_zoo.REGISTRY))
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--four_bit", action="store_true")
    ap.add_argument("--out_dir", default="results/vlm_behavior")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists"); return

    eps = sorted(d for d in glob.glob(os.path.join(DATA, "*")) if os.path.isdir(d))[:args.num]
    print(f"[behavior] {args.model}: {len(eps)} episodes", flush=True)
    model, proc = vlm_zoo.load(args.model, four_bit=args.four_bit)

    rec, hal, sit_ok, sit_tot, per = [], [], 0, 0, []
    for ep in eps:
        img_p = os.path.join(ep, "input_image.png")
        if not os.path.exists(img_p):
            continue
        img = Image.open(img_p).convert("RGB")
        gt_sit = gt_situation(ep); gt_n = gt_vessel_count(ep)

        a_v = vlm_zoo.generate(model, proc, img, "Is there a vessel or ship in this image? Answer yes or no.", 16)
        if gt_n is None or gt_n > 0:
            rec.append(1.0 if yes(a_v) else 0.0)
        for obj in ABSENT:
            a = vlm_zoo.generate(model, proc, img, f"Is there a {obj} in this image? Answer yes or no.", 16)
            hal.append(1.0 if yes(a) else 0.0)
        a_s = vlm_zoo.generate(model, proc, img,
            "Look at the vessels. Classify the COLREG encounter situation as exactly one word: "
            "head-on, crossing, overtaking, or safe. Answer with only that one word.", 24)
        ps = parse_sit(a_s); sit_tot += 1; sit_ok += int(ps == gt_sit)
        per.append({"ep": os.path.basename(ep), "gt_sit": gt_sit, "pred_sit": ps,
                    "raw_sit": a_s[:120], "vessel_yes": yes(a_v), "gt_n": gt_n})

    res = {
        "model": args.model, "repo": vlm_zoo.REGISTRY[args.model][0], "n": len(per),
        "vessel_recall": round(sum(rec)/max(len(rec), 1), 4),
        "halluc_rate": round(sum(hal)/max(len(hal), 1), 4),
        "colreg_acc": round(sit_ok/max(sit_tot, 1), 4),
        "per_sample": per,
    }
    json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[behavior] {args.model}: vessel_recall={res['vessel_recall']} "
          f"halluc_rate={res['halluc_rate']} colreg_acc={res['colreg_acc']} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

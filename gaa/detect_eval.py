"""
Attention-grounded DETECTION evaluation (real mAP@50 + FP/image), single class.

Promotes the Attn-IoU proxy to a standard detection metric WITHOUT training a
head: the CLS-token attention map is turned into scored vessel detections
(multi-threshold connected components -> boxes, score = mean attention in the
component, NMS), then scored against GT boxes with all-point AP@IoU0.5.
This isolates the vision encoder's attention quality as detection performance,
so it directly tests whether a geo-trained encoder detects vessels better.

Metrics:
  mAP@50      : all-point average precision at IoU>=0.5 (single class "vessel"), higher better.
  FP_per_img  : mean false positives per image at the F1-optimal score threshold (safety proxy), lower better.
  recall@F1   : recall at that threshold.

Compares stock CLIP vs one or more vision_encoder.bin checkpoints.

Usage:
  python gaa/detect_eval.py --data_path data/smd_onshore_big.json \
      --image_dir /home/ywlee/SSD/smd_onshore_big \
      --ckpts stock,checkpoints/geo_encoders/inbox_ce_vision_encoder.bin
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np
import torch
from PIL import Image
from scipy import ndimage


def parse_args():
    p = argparse.ArgumentParser("attention-grounded detection eval")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--ckpts", default="stock",
                   help="comma list; 'stock' or path to vision_encoder.bin")
    p.add_argument("--feature_size", type=int, default=-1)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args()


def load_clip(vision_model, ckpt, dtype, device):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    m = CLIPVisionModel.from_pretrained(vision_model, torch_dtype=dtype,
                                        attn_implementation="eager")
    proc = CLIPImageProcessor.from_pretrained(vision_model)
    if ckpt and ckpt != "stock":
        if os.path.exists(ckpt):
            st = torch.load(ckpt, map_location="cpu", weights_only=True)
            if st and all(k.startswith("model.") for k in st):
                st = {k[len("model."):]: v for k, v in st.items()}
            miss, unexp = m.load_state_dict(st, strict=False)
            print(f"  [load] {os.path.basename(ckpt)}: matched {len(st)-len(unexp)}/{len(st)}")
        else:
            print(f"  [WARN] missing {ckpt}, using stock")
    return m.to(device).eval(), proc


@torch.no_grad()
def attn_map(model, pv, F):
    a = model(pixel_values=pv, output_attentions=True).attentions[-1]  # (1,H,N+1,N+1)
    cls = a[0, :, 0, 1:].mean(0).float()        # (N,)
    return cls.reshape(F, F).cpu().numpy()


def attn_to_dets(amap):
    """Multi-threshold connected components -> list of (box[x1,y1,x2,y2] norm, score)."""
    F = amap.shape[0]
    a = amap / (amap.max() + 1e-8)
    dets = []
    for t in np.linspace(0.3, 0.9, 7):
        lab, n = ndimage.label(a > t)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) == 0:
                continue
            score = float(a[lab == i].mean())
            box = [xs.min()/F, ys.min()/F, (xs.max()+1)/F, (ys.max()+1)/F]
            dets.append((box, score))
    return dets


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for box, sc in dets:
        if all(iou(box, k[0]) < thr for k in keep):
            keep.append((box, sc))
    return keep


def ap50(all_preds, all_gts):
    """all-point AP@0.5, single class. all_preds: list per image of (box,score); all_gts: list per image of boxes."""
    # flatten predictions with image idx
    P = [(i, b, s) for i, ps in enumerate(all_preds) for (b, s) in ps]
    P.sort(key=lambda x: -x[2])
    n_gt = sum(len(g) for g in all_gts)
    if n_gt == 0:
        return 0.0, []
    matched = [set() for _ in all_gts]
    tp = np.zeros(len(P)); fp = np.zeros(len(P))
    for k, (i, box, s) in enumerate(P):
        gts = all_gts[i]
        best, bj = 0.5, -1
        for j, g in enumerate(gts):
            if j in matched[i]:
                continue
            v = iou(box, g)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            matched[i].add(bj); tp[k] = 1
        else:
            fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-8)
    # all-point AP
    mrec = np.concatenate(([0], rec, [1])); mpre = np.concatenate(([0], prec, [0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1])
    return float(ap), list(zip([p[2] for p in P], tp, fp))


def fp_per_image_at_bestF1(scored, n_images, n_gt):
    """sweep score threshold, find F1-optimal, return (FP/img, recall) there."""
    if not scored:
        return 0.0, 0.0
    best = (0.0, 0.0, 0.0)  # f1, fp_img, rec
    for thr in np.linspace(0.05, 0.95, 19):
        tp = sum(t for s, t, f in scored if s >= thr)
        fp = sum(f for s, t, f in scored if s >= thr)
        rec = tp / max(n_gt, 1); prec = tp / max(tp+fp, 1e-8)
        f1 = 2*prec*rec/max(prec+rec, 1e-8)
        if f1 > best[0]:
            best = (f1, fp/max(n_images, 1), rec)
    return best[1], best[2]


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [d for d in json.load(open(args.data_path)) if d.get("bboxes")]
    if args.num_samples > 0:
        data = data[:args.num_samples]
    F = args.feature_size
    if F == -1:
        from transformers import CLIPVisionConfig
        c = CLIPVisionConfig.from_pretrained(args.vision_model)
        F = c.image_size // c.patch_size

    results = {}
    for ckpt in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        label = "stock" if ckpt == "stock" else os.path.basename(os.path.dirname(ckpt)) or ckpt
        model, proc = load_clip(args.vision_model, ckpt, dtype, device)
        preds, gts = [], []
        for s in data:
            try:
                img = Image.open(os.path.join(args.image_dir, s["image"])).convert("RGB")
            except Exception:
                continue
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            am = attn_map(model, pv, F)
            preds.append(nms(attn_to_dets(am)))
            gts.append([b for b in s["bboxes"]])
        ap, scored = ap50(preds, gts)
        n_gt = sum(len(g) for g in gts)
        fpi, rec = fp_per_image_at_bestF1(scored, len(preds), n_gt)
        results[label] = dict(mAP50=ap, FP_per_img=fpi, recall=rec, n=len(preds), n_gt=n_gt)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  Attention-grounded detection on {os.path.basename(args.data_path)} "
          f"(n={len(data)})\n{'='*70}")
    print(f"{'encoder':<22}{'mAP@50↑':>10}{'FP/img↓':>10}{'recall':>9}")
    print("-"*51)
    for k, m in results.items():
        print(f"{k:<22}{m['mAP50']:>10.4f}{m['FP_per_img']:>10.3f}{m['recall']:>9.3f}")


if __name__ == "__main__":
    main()

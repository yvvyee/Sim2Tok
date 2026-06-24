"""
H0 verification: does texture bias occur WITHOUT GAA?

Decomposes CLS attention into {vessel, sea, sky} regions using SMD ObjectGT
(boxes) + HorizonGT (horizon line per frame), and measures the correlation
between attention and high-frequency image energy (waves/glint texture).

Texture-bias evidence for a no-GAA model:
  - high SEA% (attention sits on sea/wave texture, not the vessel),
  - low VESSEL%,
  - high attn<->high-frequency correlation.
GAA should shift mass to VESSEL% and lower the high-freq correlation.

Compares encoders (stock / no-GAA SFT / GAA) on the SAME frames.

Usage:
  python gaa/texture_bias_eval.py --smd_root /tmp/smd --split VIS_Onshore \
    --frames_per_video 6 \
    --ckpts stock=stock,noGAA=checkpoints/ctrl_baseline_train80/vision_encoder.bin,GAA=checkpoints/gaa_inboxce_train80/vision_encoder.bin
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob
import numpy as np
import scipy.io as sio
from scipy import ndimage
import cv2
import torch
from PIL import Image

from gaa.geometric_loss import compute_geometric_mask


def parse_args():
    p = argparse.ArgumentParser("texture-bias (H0) verification")
    p.add_argument("--smd_root", required=True)
    p.add_argument("--split", default="VIS_Onshore")
    p.add_argument("--frames_per_video", type=int, default=6)
    p.add_argument("--max_videos", type=int, default=40)
    p.add_argument("--ckpts", required=True, help="label=path,... (path='stock' for pretrained)")
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--feature_size", type=int, default=-1)
    p.add_argument("--geo_tau", type=float, default=0.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args()


def load_clip(vm, ckpt, dtype, device):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    m = CLIPVisionModel.from_pretrained(vm, torch_dtype=dtype, attn_implementation="eager")
    proc = CLIPImageProcessor.from_pretrained(vm)
    if ckpt and ckpt != "stock" and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location="cpu", weights_only=True)
        if st and all(k.startswith("model.") for k in st):
            st = {k[len("model."):]: v for k, v in st.items()}
        m.load_state_dict(st, strict=False)
    return m.to(device).eval(), proc


@torch.no_grad()
def cls_attn(model, pv, F):
    a = model(pixel_values=pv, output_attentions=True).attentions[-1]
    return a[0, :, 0, 1:].mean(0).float().reshape(F, F).cpu().numpy()


def load_struct(mat, field):
    m = sio.loadmat(mat, struct_as_record=False, squeeze_me=True)
    return np.atleast_1d(m["structXML"])


def horizon_masks(F, W, H, hz):
    """Return (sea_mask, sky_mask) on FxF grid using horizon point+normal."""
    X, Y, Nx, Ny = float(hz.X), float(hz.Y), float(hz.Nx), float(hz.Ny)
    # patch-center pixel coords
    cc = (np.arange(F) + 0.5) / F * W
    rr = (np.arange(F) + 0.5) / F * H
    gx, gy = np.meshgrid(cc, rr)               # (F,F)
    val = Nx * (gx - X) + Ny * (gy - Y)
    # sea = same sign as bottom-center reference (clearly sea)
    ref = Nx * (W/2 - X) + Ny * (H - 1 - Y)
    sea = (np.sign(val) == np.sign(ref))
    return sea.astype(np.float32), (~sea).astype(np.float32)


def highfreq_energy(gray, F):
    """High-pass energy downsampled to FxF (texture proxy)."""
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=2.0)
    hp = np.abs(g - blur)
    # block-reduce to FxF
    Hh, Ww = hp.shape
    yb = np.linspace(0, Hh, F + 1).astype(int); xb = np.linspace(0, Ww, F + 1).astype(int)
    out = np.zeros((F, F), np.float32)
    for r in range(F):
        for c in range(F):
            out[r, c] = hp[yb[r]:yb[r+1], xb[c]:xb[c+1]].mean()
    return out


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = args.feature_size
    if F == -1:
        from transformers import CLIPVisionConfig
        F = (lambda c: c.image_size // c.patch_size)(CLIPVisionConfig.from_pretrained(args.vision_model))
    base = os.path.join(args.smd_root, args.split, args.split)
    gt_dir, vid_dir, hz_dir = (os.path.join(base, d) for d in ("ObjectGT", "Videos", "HorizonGT"))
    mats = sorted(glob.glob(os.path.join(gt_dir, "*ObjectGT.mat")))[:args.max_videos]

    # Pre-extract frames + region masks + highfreq once (shared across encoders)
    samples = []  # (pil_image, fg(FxF), sea(FxF), sky(FxF), hf(FxF))
    for mat in mats:
        stem = os.path.basename(mat).replace("_ObjectGT.mat", "")
        vids = [v for ext in (".avi", ".mp4", ".mov") for v in glob.glob(os.path.join(vid_dir, stem + ext))]
        hzf = os.path.join(hz_dir, stem + "_HorizonGT.mat")
        if not vids or not os.path.exists(hzf):
            continue
        obj = load_struct(mat, "BB"); hz = load_struct(hzf, None)
        cap = cv2.VideoCapture(vids[0])
        n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(obj), len(obj), len(hz))
        if n == 0:
            cap.release(); continue
        for fi in np.linspace(0, n - 1, args.frames_per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            bb = getattr(obj[fi], "BB", None)
            boxes = []
            if bb is not None and np.size(bb):
                for b in np.atleast_2d(np.asarray(bb, float)):
                    x, y, w, h = b[:4]
                    if w > 0 and h > 0:
                        boxes.append([max(0, x/W), max(0, y/H), min(1, (x+w)/W), min(1, (y+h)/H)])
            if not boxes:
                continue
            fg = compute_geometric_mask([boxes], F, torch.device("cpu"), args.geo_tau)[0].numpy()
            sea, sky = horizon_masks(F, W, H, hz[fi])
            sea = sea * (1 - fg); sky = sky * (1 - fg)   # exclude vessel
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hf = highfreq_energy(gray, F)
            samples.append((Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fg, sea, sky, hf))
        cap.release()
    print(f"[H0] {len(samples)} frames with boxes from {args.split}", flush=True)

    encoders = [kv.split("=", 1) for kv in args.ckpts.split(",")]
    results = {}
    for lab, ckpt in encoders:
        model, proc = load_clip(args.vision_model, ckpt, dtype, device)
        ves, see, sk, fc = [], [], [], []
        for img, fg, sea, sky, hf in samples:
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            am = cls_attn(model, pv, F)
            tot = am.sum() + 1e-8
            ves.append(float((am*fg).sum()/tot)); see.append(float((am*sea).sum()/tot)); sk.append(float((am*sky).sum()/tot))
            a = (am - am.mean()).ravel(); h = (hf - hf.mean()).ravel()
            denom = (np.linalg.norm(a)*np.linalg.norm(h) + 1e-8)
            fc.append(float((a*h).sum()/denom))
        mean = lambda v: sum(v)/max(len(v), 1)
        results[lab] = dict(vessel=mean(ves), sea=mean(see), sky=mean(sk), freqcorr=mean(fc))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*68}\n  H0 texture-bias decomposition ({args.split}, n={len(samples)} frames)\n{'='*68}")
    print(f"{'encoder':<10}{'vessel%↑':>10}{'sea%↓':>10}{'sky%':>9}{'freqcorr↓':>12}")
    print("-"*51)
    for lab, m in results.items():
        print(f"{lab:<10}{m['vessel']*100:>9.1f}{m['sea']*100:>10.1f}{m['sky']*100:>9.1f}{m['freqcorr']:>12.3f}")
    print("\nvessel% = attn mass on vessel boxes; sea% = on sea (waves) region;")
    print("freqcorr = corr(attention, high-frequency texture energy). Texture bias => high sea% + high freqcorr, low vessel%.")


if __name__ == "__main__":
    main()

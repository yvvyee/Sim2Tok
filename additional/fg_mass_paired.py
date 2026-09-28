"""
Paired comparison of two alignment measures on per-frame evaluation records.

Reads two per-sample JSONL files written by eval_attn_mp.py (same frames, two
encoders) and reports, for both attn_iou (area-matched top-K overlap) and
attn_in_box (fraction of CLS attention mass inside the boxes):

  * the mean under each encoder,
  * the paired difference with a 95% interval bootstrapped over videos,
  * the Spearman rank correlation between the two measures per encoder,
  * the uniform-attention reference for attn_in_box (mean foreground share).

Videos are the resampling unit because adjacent frames of one video are not
independent draws.

    python additional/fg_mass_paired.py \
        --a additional_exp/raw/full_smd_stock/per_sample_stock.jsonl \
        --b additional_exp/raw/full_smd_paper/per_sample_paper_gaa.jsonl \
        --out additional_exp/raw/fg_mass_full_smd.json
"""
import argparse
import json
import math
import random

FIELDS = ("attn_iou", "attn_in_box")
N_PATCHES = 576


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def ranks(x):
    order = sorted(range(len(x)), key=lambda k: x[k])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        mid = (i + j) / 2.0          # average rank for ties
        for k in range(i, j + 1):
            r[order[k]] = mid
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="reference encoder records")
    p.add_argument("--b", required=True, help="compared encoder records")
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_empty", action="store_true",
                   help="drop frames whose foreground mask is empty in either "
                        "file (fg_patches == 0); with --box_geometry cropped "
                        "these are frames whose every vessel lies outside the "
                        "centre crop, where both measures are undefined")
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    n_before = len(ids)
    if args.exclude_empty:
        ids = [i for i in ids if a[i]["fg_patches"] > 0 and b[i]["fg_patches"] > 0]
    videos = {}
    for i in ids:
        videos.setdefault(a[i]["video"], []).append(i)
    vlist = sorted(videos)

    rng = random.Random(args.seed)
    boots = [[i for v in rng.choices(vlist, k=len(vlist)) for i in videos[v]]
             for _ in range(args.n_boot)]

    res = {"n_frames": len(ids), "n_frames_before_exclusion": n_before,
           "n_videos": len(vlist),
           "a": args.a, "b": args.b, "n_boot": args.n_boot, "seed": args.seed,
           "uniform_attn_in_box": sum(a[i]["fg_patches"] for i in ids) / len(ids) / N_PATCHES}
    for f in FIELDS:
        diff = {i: b[i][f] - a[i][f] for i in ids}
        bs = sorted(sum(diff[i] for i in s) / len(s) for s in boots)
        res[f] = {
            "mean_a": sum(a[i][f] for i in ids) / len(ids),
            "mean_b": sum(b[i][f] for i in ids) / len(ids),
            "paired_diff": sum(diff.values()) / len(ids),
            "ci95": [bs[int(0.025 * args.n_boot)], bs[int(0.975 * args.n_boot) - 1]],
        }
    res["spearman_a"] = spearman([a[i]["attn_iou"] for i in ids], [a[i]["attn_in_box"] for i in ids])
    res["spearman_b"] = spearman([b[i]["attn_iou"] for i in ids], [b[i]["attn_in_box"] for i in ids])

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

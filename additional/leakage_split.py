"""
Split a paired alignment comparison by whether each evaluation video was used
to train the compared encoder.

The paper's GAA encoder (geo_encoders_fix/inbox_ce_vision_encoder.bin) was
trained on data/geopool_plus_real.json: 264 simulator images plus 159 SMD
frames with ground-truth boxes from 27 SMD videos.  The full-SMD evaluation
contains every labelled frame, so some of its videos are training videos of
that encoder.  This script reports the paired difference on the overlapping
videos and on the held-out videos separately, overall and per condition.

    python additional/leakage_split.py \
        --train_json /home/ywlee/SSD/dev/home-dev/gaa/data/geopool_plus_real.json \
        --a additional_exp/raw/full_smd_stock/per_sample_stock.jsonl \
        --b additional_exp/raw/full_smd_paper/per_sample_paper_gaa.jsonl \
        --out additional_exp/raw/leakage_split_full_smd.json
"""
import argparse
import json
import random
import re

VIDEO_RE = re.compile(r"(MVI_\d+_[A-Za-z_]+?)_f\d+")


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def train_videos(path):
    vids = set()
    for r in json.load(open(path)):
        m = VIDEO_RE.search(r.get("image", ""))
        if m:
            vids.add(m.group(1))
    return vids


def paired(a, b, sel, field, n_boot, seed):
    videos = {}
    for i in sel:
        videos.setdefault(a[i]["video"], []).append(i)
    vlist = sorted(videos)
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        s = [i for v in rng.choices(vlist, k=len(vlist)) for i in videos[v]]
        bs.append(sum(b[i][field] - a[i][field] for i in s) / len(s))
    bs.sort()
    ma = sum(a[i][field] for i in sel) / len(sel)
    mb = sum(b[i][field] for i in sel) / len(sel)
    return {"n_videos": len(vlist), "n_frames": len(sel), "mean_a": ma, "mean_b": mb,
            "paired_diff": mb - ma,
            "ci95": [bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--a", required=True, help="reference encoder records")
    p.add_argument("--b", required=True, help="compared encoder records")
    p.add_argument("--out", required=True)
    p.add_argument("--field", default="attn_iou")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tv = train_videos(args.train_json)
    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    eval_videos = {a[i]["video"] for i in ids}

    groups = {"all": ids,
              "train_videos": [i for i in ids if a[i]["video"] in tv],
              "held_out_videos": [i for i in ids if a[i]["video"] not in tv]}
    for part in ("train_videos", "held_out_videos"):
        for cond in sorted({a[i]["condition"] for i in groups[part]}):
            groups[f"{part}/{cond}"] = [i for i in groups[part] if a[i]["condition"] == cond]

    res = {"field": args.field, "train_json": args.train_json,
           "n_train_videos": len(tv),
           "train_videos_in_eval": sorted(tv & eval_videos),
           "train_videos_not_in_eval": sorted(tv - eval_videos),
           "groups": {k: paired(a, b, v, args.field, args.n_boot, args.seed)
                      for k, v in groups.items() if v}}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res["groups"].items():
        print(f"{k:32s} videos {v['n_videos']:3d} frames {v['n_frames']:6d} "
              f"a {v['mean_a']:.4f} b {v['mean_b']:.4f} diff {v['paired_diff']:+.4f} "
              f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()

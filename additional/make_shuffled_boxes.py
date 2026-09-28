"""
Shuffled-box control for the geometric loss (condition M3, hypothesis H2c).

Every image receives the complete box list of another, randomly chosen image.
The loss, optimiser, step count and the distribution of box count, size and
position are unchanged; only the correspondence between a box and the vessel in
the pixels is broken.  If an encoder trained this way changes the downstream
behaviour as much as the real GAA encoder does, the change cannot be credited to
attending to vessels.

The permutation is a derangement: no image keeps its own boxes.

    python -m additional.make_shuffled_boxes \\
        --in_json data/ok_aivn_10000.json --out_json data/aivn_10k_shuffled.json
"""
from __future__ import annotations

import argparse
import json
import random


def derangement(n: int, rng: random.Random) -> list[int]:
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            return perm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    data = json.load(open(args.in_json))
    perm = derangement(len(data), rng)
    out = []
    for i, j in enumerate(perm):
        r = dict(data[i])
        r["bboxes"] = data[j]["bboxes"]
        r["boxes_from"] = data[j]["id"]
        out.append(r)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False)
    same = sum(1 for i, j in enumerate(perm) if data[i]["bboxes"] == data[j]["bboxes"])
    print(f"[shuffle] {len(out)} images, seed {args.seed}, "
          f"identical box lists after shuffle: {same}")


if __name__ == "__main__":
    main()

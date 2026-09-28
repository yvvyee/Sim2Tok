"""
Supervised fine-tuning set for the verification VLMs (M0, M1, M3), built from
the AIVN-SDS 10k simulator corpus only.

Sources, all simulator-derived:

* the 10,000 full frames with their navigator-written situation description
  (``describe``) and navigation advice (``advice``),
* the sub-crops written by ``make_sim_crops``: vessel-free windows (``neg``)
  and windows that contain whole vessels (``pos``) at the same scales.

Every frame and crop inherits one split from ``data/aivn_split.json``, so the
validation set shares no source frame with the training set.  The validation
set is what the sim-side score of hypothesis H1c is measured on.

Yes/no questions are graded against what the model is shown.  The encoder
receives the CLIP centre crop, so on a full frame only the vessels inside that
crop count, in crop coordinates.  A sub-crop is square and is shown whole.
The ``describe`` and ``advice`` texts were written about the whole frame and
can mention vessels the centre crop removes; they are kept as the corpus
provides them.

Question families:

``vessel``   "Is there a {ship|boat|vessel} in this image?"  Yes on frames and
             positive crops, no on negative crops.  The frame-level rate is
             drawn so that yes and no occur equally often overall.
``count``    "Are there more than {n} vessels in this image?"  The threshold is
             drawn on either side of the visible count.
``side``     "Is the largest vessel on the {left|right} side of the image?"
             Left and right of the image, not port and starboard: the real
             evaluation sets include shore cameras, where the ship's sides are
             undefined.
``absent``   A category from ``TRAIN_ABSENT``.  Always no.
``present``  water, sea, sky or horizon.  Always yes.

The evaluation-only absent categories are disjoint from ``TRAIN_ABSENT``; the
script refuses to run otherwise.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

from additional.box_geometry import transform_boxes
from additional.build_yesno_sft import EVAL_ONLY_ABSENT, TRAIN_ABSENT

PRESENT_SCENE = ["water", "sea", "sky", "horizon"]
VESSEL_WORDS = ["ship", "boat", "vessel"]

Q_OBJ = "Is there a {obj} in this image? Answer yes or no."
Q_COUNT = "Are there more than {n} vessels in this image? Answer yes or no."
Q_SIDE = ("Is the largest vessel on the {side} side of the image? "
          "Answer yes or no.")
YES, NO = "Yes.", "No."


def conv(q: str, a: str) -> list[dict]:
    return [{"from": "human", "value": "<image>\n" + q},
            {"from": "gpt", "value": a}]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def count_q(rng, count):
    if count >= 1 and rng.random() < 0.5:
        n = rng.randint(0, count - 1)
    else:
        n = rng.randint(count, count + 2)
    return Q_COUNT.format(n=n), (YES if count > n else NO)


def side_q(rng, boxes):
    big = max(boxes, key=area)
    cx = (big[0] + big[2]) / 2
    true_side = "left" if cx < 0.5 else "right"
    asked = rng.choice(["left", "right"])
    return Q_SIDE.format(side=asked), (YES if asked == true_side else NO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--describe_json", required=True)
    p.add_argument("--advice_json", required=True)
    p.add_argument("--crops_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--frame_image_dir", required=True)
    p.add_argument("--crop_image_dir", required=True)
    p.add_argument("--out_train", required=True)
    p.add_argument("--out_val", required=True)
    p.add_argument("--stats_json", required=True)
    p.add_argument("--image_w", type=int, default=1920)
    p.add_argument("--image_h", type=int, default=1080)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    overlap = set(TRAIN_ABSENT) & set(EVAL_ONLY_ABSENT)
    if overlap:
        raise SystemExit(f"train/eval absent categories overlap: {overlap}")

    rng = random.Random(args.seed)
    split = json.load(open(args.split_json))["split"]
    desc = json.load(open(args.describe_json))
    adv = {r["id"]: r for r in json.load(open(args.advice_json))}
    crops = json.load(open(args.crops_json))

    n_neg = sum(1 for c in crops if c["kind"] == "neg")
    n_pos = sum(1 for c in crops if c["kind"] == "pos")
    # Frames and positive crops both answer yes; draw the frame-level vessel
    # question often enough that yes and no balance overall.
    p_frame_vessel = max(0.0, min(1.0, (n_neg - n_pos) / max(1, len(desc))))

    out = {"train": [], "val": []}
    stats = collections.Counter()

    def add(sp, rid, image, q_or_conv, answer, family, source):
        c = q_or_conv if answer is None else conv(q_or_conv, answer)
        out[sp].append({"id": rid, "image": image, "conversations": c,
                        "family": family, "source_id": source,
                        "answer": answer})
        stats[f"{sp}/{family}/{answer or 'text'}"] += 1

    for r in desc:
        sp = split[r["id"]]
        img = os.path.join(args.frame_image_dir, r["image"])
        add(sp, r["id"] + "__describe", img, r["conversations"], None,
            "describe", r["id"])
        if r["id"] in adv:
            add(sp, r["id"] + "__advice", img, adv[r["id"]]["conversations"],
                None, "advice", r["id"])
        vis = transform_boxes(r["bboxes"], args.image_w, args.image_h)
        if rng.random() < p_frame_vessel:
            add(sp, r["id"] + "__vessel", img,
                Q_OBJ.format(obj=rng.choice(VESSEL_WORDS)), YES, "vessel", r["id"])
        q, a = count_q(rng, len(vis))
        add(sp, r["id"] + "__count", img, q, a, "count", r["id"])
        if vis:
            q, a = side_q(rng, vis)
            add(sp, r["id"] + "__side", img, q, a, "side", r["id"])
        add(sp, r["id"] + "__absent", img,
            Q_OBJ.format(obj=rng.choice(TRAIN_ABSENT)), NO, "absent", r["id"])
        add(sp, r["id"] + "__present", img,
            Q_OBJ.format(obj=rng.choice(PRESENT_SCENE)), YES, "present", r["id"])

    for c in crops:
        sp = c["split"]
        img = os.path.join(args.crop_image_dir, c["image"])
        has = c["kind"] == "pos"
        add(sp, c["id"] + "__vessel", img,
            Q_OBJ.format(obj=rng.choice(VESSEL_WORDS)), YES if has else NO,
            "vessel", c["source_id"])
        q, a = count_q(rng, c["n_vessels"])
        add(sp, c["id"] + "__count", img, q, a, "count", c["source_id"])
        if has:
            q, a = side_q(rng, c["bboxes"])
            add(sp, c["id"] + "__side", img, q, a, "side", c["source_id"])

    # data/dataset.py replaces an unreadable image with a black one and keeps
    # training, so a wrong path would train silently on blank input.  Refuse.
    missing = sorted({r["image"] for sp in out for r in out[sp]
                      if not os.path.exists(r["image"])})
    if missing:
        raise SystemExit(f"{len(missing)} image paths do not exist, "
                         f"e.g. {missing[:3]}")

    for sp in out:
        rng.shuffle(out[sp])
    json.dump(out["train"], open(args.out_train, "w"), ensure_ascii=False)
    json.dump(out["val"], open(args.out_val, "w"), ensure_ascii=False)

    fam = collections.defaultdict(collections.Counter)
    for sp in out:
        for r in out[sp]:
            fam[f"{sp}/{r['family']}"][r["answer"] or "text"] += 1
    summary = {"n_train": len(out["train"]), "n_val": len(out["val"]),
               "p_frame_vessel": p_frame_vessel, "n_neg_crops": n_neg,
               "n_pos_crops": n_pos,
               "by_family": {k: dict(v) for k, v in sorted(fam.items())},
               "train_absent": TRAIN_ABSENT,
               "eval_only_absent": EVAL_ONLY_ABSENT,
               "args": vars(args)}
    json.dump(summary, open(args.stats_json, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

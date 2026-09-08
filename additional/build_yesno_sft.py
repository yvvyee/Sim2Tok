"""
Build a yes/no answering set from the AIVN-SDS corpus.

The submitted VLM cannot be measured with the absence protocol.  Its LoRA was
fine-tuned on navigational narration, and the measurement shows it does not
answer an absence question in any read-out: free generation returns seven
distinct response openings over 70,800 probes, all of them unparsed under the
submitted match rule, and constrained first-token scoring answers "yes" to
``airplane`` on 86.88% of the 29,253 SMD frames against 82.93% for ``water``,
which is present in every one of them.  The affirmative rate is therefore a
yes-bias rather than a hallucination rate.

This module builds training data that gives the model the missing ability, so
the submitted protocol can be run on a model that can actually take it.  Three
constraints shape the set, and each exists to stop the resulting measurement
from being vacuous.

**The evaluation categories never appear in training.**  A model trained on
"airplane -> no" answers "no" to airplanes from memory, on any image, under any
encoder.  The hallucination rate would then be zero for every condition and the
comparison would measure memorisation.  So the submitted work's three
categories, plus five more, are held out entirely; training uses a disjoint set
of twelve.

**Both answers occur about equally often.**  A set that is mostly "no" teaches
the constant "no", which again scores a perfect hallucination rate without
grounding.  Presence questions about things in every maritime frame supply the
"yes" side.

**Some questions cannot be answered without looking.**  Category questions have
an answer that is constant across the corpus, so a model can learn them as a
lookup.  Vessel count and vessel side vary frame by frame and are recoverable
only from the image, so they force the pooled representation to carry scene
content.  These are the questions the mechanism claim is about.

Usage::

    python -m additional.build_yesno_sft \\
        --in_json data/ok_aivn_10000.json \\
        --out_json data/yesno_aivn_10k.json \\
        --stats_json results/yesno_sft_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

#: Held out of training so the evaluation measures grounding, not recall.
#: The first three are the submitted work's categories.
EVAL_ONLY_ABSENT = ["airplane", "car", "building",
                    "elephant", "zebra", "refrigerator", "laptop", "clock"]

#: Absent categories used for training.  Disjoint from EVAL_ONLY_ABSENT.
TRAIN_ABSENT = ["train", "bicycle", "motorcycle", "traffic light",
                "fire hydrant", "stop sign", "bench", "horse", "sheep", "cow",
                "television", "book"]

#: Present in every frame of open-water maritime imagery.
TRAIN_PRESENT = ["ship", "vessel", "water", "sea", "sky", "horizon"]

QUESTION_OBJ = "Is there a {obj} in this image? Answer yes or no."
QUESTION_COUNT = ("Are there more than {n} vessels in this image? "
                  "Answer yes or no.")
QUESTION_SIDE = ("Is the nearest vessel on the {side} side of the image? "
                 "Answer yes or no.")

YES, NO = "Yes.", "No."


def nearest_box(boxes: list[list[float]]) -> list[float] | None:
    """The box with the largest area, taken as the nearest vessel.

    Apparent size is the only depth cue the annotation carries, so "nearest"
    means "largest" here.  The questions built from it are graded against that
    same definition, so the label is exact even though the physical claim is an
    approximation.
    """
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def build_for_sample(s: dict, rng: random.Random) -> list[dict]:
    """Four questions for one image: absent, present, count, side."""
    boxes = s.get("bboxes") or []
    out: list[dict] = []

    def rec(tag: str, question: str, answer: str) -> dict:
        return {
            "id": f"{s['id']}__{tag}",
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
            "source_id": s["id"],
            "question_type": tag,
            "answer": answer,
        }

    obj = rng.choice(TRAIN_ABSENT)
    out.append(rec("absent", QUESTION_OBJ.format(obj=obj), NO))

    obj = rng.choice(TRAIN_PRESENT)
    out.append(rec("present", QUESTION_OBJ.format(obj=obj), YES))

    # The threshold is drawn on either side of the true count so that the count
    # question is not answerable from a fixed threshold.  With n < count the
    # answer is yes and with n >= count it is no, and both are drawn equally
    # often, so the model cannot score by always answering one way.
    count = len(boxes)
    if count >= 1:
        # count >= 1 always admits a threshold below the count, so both answers
        # are reachable for every annotated image and no count value is locked
        # to one answer.
        if rng.random() < 0.5:
            n = rng.randint(0, count - 1)      # yes
        else:
            n = rng.randint(count, count + 2)  # no
        out.append(rec("count", QUESTION_COUNT.format(n=n),
                       YES if count > n else NO))

    nb = nearest_box(boxes)
    if nb is not None:
        cx = (nb[0] + nb[2]) / 2.0
        true_side = "port" if cx < 0.5 else "starboard"
        asked = rng.choice(["port", "starboard"])
        out.append(rec("side", QUESTION_SIDE.format(side=asked),
                       YES if asked == true_side else NO))

    return out


def parse_args():
    p = argparse.ArgumentParser("build a yes/no answering set from AIVN-SDS")
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--stats_json", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    with open(args.in_json) as f:
        samples = json.load(f)
    if args.limit > 0:
        samples = samples[:args.limit]

    overlap = set(TRAIN_ABSENT) & set(EVAL_ONLY_ABSENT)
    if overlap:
        raise SystemExit(f"학습 범주와 평가 범주가 겹칩니다: {sorted(overlap)}. "
                         f"겹치면 평가가 암기를 재게 됩니다.")

    records: list[dict] = []
    for s in samples:
        records.extend(build_for_sample(s, rng))
    rng.shuffle(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(records, f, ensure_ascii=False)

    by_type = Counter(r["question_type"] for r in records)
    by_answer = Counter(r["answer"] for r in records)
    per_type_answer = {
        t: dict(Counter(r["answer"] for r in records
                        if r["question_type"] == t))
        for t in sorted(by_type)
    }
    stats = {
        "in_json": args.in_json, "out_json": args.out_json,
        "seed": args.seed,
        "n_source_images": len(samples),
        "n_records": len(records),
        "by_question_type": dict(by_type),
        "by_answer": dict(by_answer),
        "yes_fraction": round(by_answer[YES] / max(len(records), 1), 6),
        "per_type_answer": per_type_answer,
        "train_absent_categories": TRAIN_ABSENT,
        "train_present_categories": TRAIN_PRESENT,
        "eval_only_absent_categories": EVAL_ONLY_ABSENT,
        "held_out_note": ("평가 범주는 학습에 전혀 등장하지 않는다. 등장하면 "
                          "평가가 접지가 아니라 암기를 재게 된다"),
    }

    print(f"[yesno] 원본 이미지 {len(samples):,} -> 질문 {len(records):,}")
    print(f"[yesno] 유형별 {dict(by_type)}")
    print(f"[yesno] 답 분포 {dict(by_answer)}  yes 비율 "
          f"{stats['yes_fraction']:.4f}")
    for t, d in per_type_answer.items():
        print(f"[yesno]   {t:8s} {d}")
    print(f"[yesno] 학습 부재 범주 {len(TRAIN_ABSENT)}종, "
          f"평가 전용 부재 범주 {len(EVAL_ONLY_ABSENT)}종, 겹침 없음")

    if args.stats_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_json)),
                    exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[yesno] stats -> {args.stats_json}")


if __name__ == "__main__":
    main()

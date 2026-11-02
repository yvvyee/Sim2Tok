"""Grade saved yes/no VLM answers against new labels, without running the VLM.

``yesno_eval.py`` asks every frame the same fixed questions, and neither the
image the model sees (the CLIP centre crop) nor the question text depends on
the labels.  Only the expected answer does.  A change of labels, such as the
SMD vessel-class correction of 2026-11-02, therefore changes the grading but
not the answers, and re-grading the saved ``ans_*.jsonl`` with the same
question objects and the same ``summarise`` gives what a new run would give.

A record is kept only if the new labels define its expected answer, exactly as
``yesno_eval.ask_one`` drops undefined questions.  This requires the saved run
to have asked every question that the new labels define: a frame without a
visible vessel under the old labels has no side records, so it cannot gain one.
The script stops if that happens.

Usage::

    python -m additional.yesno_rescore --answers <old>/ans_M0__with.jsonl \\
        --old_summary <old>/summary.json --data_path data/smd_vessel.json \\
        --vessel_words ship --sides left,right --out_dir <new>
"""
from __future__ import annotations

import argparse
import json
import os

from additional.yesno_eval import (attach_visible, build_questions, part_paths, summarise,
                                   visible_boxes, write_part)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", required=True)
    p.add_argument("--old_summary", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--box_geometry", default="cropped", choices=["raw", "cropped"])
    p.add_argument("--vessel_words", default="")
    p.add_argument("--sides", default="")
    p.add_argument("--min_present_rate", type=float, default=0.5)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    old = json.load(open(args.old_summary))
    absent, present = old["absent_categories"], old["present_categories"]
    counts = old["count_thresholds"]
    words = [w for w in args.vessel_words.split(",") if w]
    sides = [s for s in args.sides.split(",") if s]
    questions = {q.key: q for q in build_questions(absent, present, counts, words, sides)}

    samples = json.load(open(args.data_path))
    attach_visible(samples, args.box_geometry)
    meta_of = {s["id"]: s for s in samples}

    answered = {}
    for line in open(args.answers):
        r = json.loads(line)
        answered[(r["id"], r["question"])] = r
    asked = {k for k in answered}

    records = []
    for (sid, key), r in answered.items():
        meta, q = meta_of[sid], questions[key]
        exp = q.expected(meta)
        if exp is None:
            continue
        r = dict(r)
        r["n_boxes"] = len(meta.get("bboxes") or [])
        r["n_visible"] = len(visible_boxes(meta))
        r["expected_yes"] = exp
        r["correct"] = int(r["said_yes"] == exp)
        r["halluc"] = r["said_yes"] if exp == 0 else 0
        records.append(r)
    missing = [(s["id"], k) for s in samples for k, q in questions.items()
               if q.expected(s) is not None and (s["id"], k) not in asked]
    if missing:
        raise SystemExit(f"{len(missing)} questions are defined by the new labels "
                         f"but were never asked, e.g. {missing[:3]}")

    records.sort(key=lambda r: (r["question"], r["id"]))
    os.makedirs(args.out_dir, exist_ok=True)
    (label, res), = old["results"].items()
    out_path = os.path.join(args.out_dir, os.path.basename(args.answers))
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # verify_s2_compare.py reads the per-question files under parts/, so they are
    # written with the same names and markers as yesno_eval writes them.
    os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)
    by_q = {}
    for r in records:
        by_q.setdefault(r["question"], []).append(r)
    mode = old.get("image_token_mode", "with")
    model = label[:-len("__" + mode)] if label.endswith("__" + mode) else label  # "M0__with" -> "M0"
    for key, rs in by_q.items():
        write_part(*part_paths(args.out_dir, model, key, mode), rs)
    stats = summarise(records, absent[:3], args.min_present_rate)
    stats["encoder"] = res.get("encoder")
    stats["image_token_mode"] = res.get("image_token_mode")
    stats["answers_path"] = out_path
    meta = dict(old)
    meta.update({"data_path": args.data_path, "rescored_from": args.answers,
                 "n_frames": len(samples), "results": {label: stats}})
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[rescore] {label}: {len(records):,} records of {len(answered):,} answers -> {out_path}")


if __name__ == "__main__":
    main()

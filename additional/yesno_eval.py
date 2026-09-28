"""
Run the absence protocol on a VLM that can answer it, and grade three things.

The submitted VLM cannot take the absence protocol at all: it narrates instead
of answering, and a constrained first-token read-out gives it a yes-bias that
runs opposite to presence (86.88% affirmative for ``airplane``, absent from
every frame, against 82.93% for ``water``, present in every frame).  So the
model is first taught to answer yes/no on the AIVN corpus by
``build_yesno_sft.py`` and ``train.py``, and this module measures the result.

Three question families are asked of every frame, and each answers a different
objection.

``absent``
    Categories that occur in no maritime frame.  An affirmative answer is a
    hallucination.  These categories are held out of the fine-tune entirely, so
    a low rate here cannot come from having been trained on the answer.  The
    first three are the submitted work's own categories, which makes the number
    comparable with the reported 5/864.

``present``
    Things in every frame.  This is the discrimination control.  If the
    affirmative rate here is not far above the absent rate, the measurement has
    found an answer bias rather than a hallucination rate, and the absent rate
    carries no claim.  The check is applied before the absent rate is reported,
    not after.

``count``
    "Are there more than N vessels in this image?", with the answer taken from
    the frame's own annotation.  Category questions have an answer that is
    constant over the corpus, so a model can score on them without looking; the
    count answer changes frame by frame and is recoverable only from the image.
    This is the family the mechanism claim is about, since a pooled
    representation dominated by background texture cannot carry the count.

Scoring reads the first generated token, comparing the probability mass on
affirmative surface forms against negative ones.  The fine-tuned model emits
"Yes." or "No." as its whole answer, so this is its actual output rather than a
constrained reading of something else; the generated string is recorded
alongside so the two can be checked against each other.

Each (encoder, question) pair is persisted as it finishes, so an interrupted
run resumes and partial results are readable while the rest is still running.

Usage::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.yesno_eval \\
        --data_path data/smd_full.json --image_dir frames \\
        --vlm_ckpt checkpoints/yesno_vlm \\
        --llm /home/yvvyee/data/Models/Llama-3.1-8B-Instruct \\
        --encoder stock=none \\
        --encoder paper_gaa=paper_ckpt/geo_encoders_fix/inbox_ce_vision_encoder.bin \\
        --out_dir results/yesno_full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from additional.config import assert_policy_or_die
from additional.stats import video_of
from additional.vlm_loader import (build_vlm, snapshot_encoder,
                                   swap_encoder, uniform_prompt)

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}

QUESTION_OBJ = "Is there a {obj} in this image? Answer yes or no."
QUESTION_COUNT = ("Are there more than {n} vessels in this image? "
                  "Answer yes or no.")
QUESTION_SIDE = ("Is the largest vessel on the {side} side of the image? "
                 "Answer yes or no.")


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def visible_boxes(meta: dict) -> list:
    """Vessel boxes graded against, set once per record by ``attach_visible``."""
    return meta.get("_vis", meta.get("bboxes") or [])


def attach_visible(samples: list[dict], geometry: str) -> None:
    """Store the boxes a question is graded against on every record.

    ``raw`` keeps full-frame boxes, as every earlier run did.  ``cropped``
    keeps only what survives the CLIP centre crop, in crop coordinates, which
    is what the model is actually shown; a vessel the crop removes cannot be
    counted, located, or expected to be seen.
    """
    from additional.box_geometry import transform_boxes
    for s_ in samples:
        b = s_.get("bboxes") or []
        s_["_vis"] = (transform_boxes(b, s_.get("width", 1920),
                                      s_.get("height", 1080))
                      if geometry == "cropped" else b)


def _vessel_expect(m: dict):
    if not m.get("vessel_labelled", True):
        return None
    return int(len(visible_boxes(m)) > 0)


def _side_expect(side: str):
    def f(m: dict):
        if not m.get("vessel_labelled", True):
            return None
        vis = visible_boxes(m)
        if not vis:
            return None
        big = max(vis, key=_area)
        cx = (big[0] + big[2]) / 2
        return int(("left" if cx < 0.5 else "right") == side)
    return f

_YES_FORMS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
_NO_FORMS = ("No", " No", "no", " no", "NO", " NO")


def first_token_ids(tokenizer, forms) -> list[int]:
    ids = []
    for f in forms:
        enc = tokenizer(f, add_special_tokens=False).input_ids
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


class Question:
    """One question asked of every frame, with a per-frame expected answer."""

    def __init__(self, key: str, family: str, text: str, expect):
        self.key = key
        self.family = family
        self.text = text
        self._expect = expect

    def expected(self, meta: dict) -> int:
        return self._expect(meta)


def build_questions(absent: list[str], present: list[str],
                    counts: list[int], vessel_words: list[str] = (),
                    sides: list[str] = ()) -> list[Question]:
    """Interleave the families so both polarities appear in the first results.

    A pass over the full frame set runs for hours per encoder and questions are
    persisted one at a time.  Grouping the families would put the whole control
    at the end, and the discrimination gap that decides whether the absent rate
    means anything would only be readable once the run was over.
    """
    qs: list[Question] = []
    for o in absent:
        qs.append(Question(f"absent:{o}", "absent",
                           QUESTION_OBJ.format(obj=o), lambda m: 0))
    for o in present:
        qs.append(Question(f"present:{o}", "present",
                           QUESTION_OBJ.format(obj=o), lambda m: 1))
    for n in counts:
        qs.append(Question(
            f"count:{n}", "count", QUESTION_COUNT.format(n=n),
            # Bind n per question; a closure over the loop variable would make
            # every count question grade against the last threshold.
            (lambda n_: (lambda m: None if not m.get("vessel_labelled", True)
                         else int(len(visible_boxes(m)) > n_)))(n)))
    for w in vessel_words:
        qs.append(Question(f"vessel:{w}", "vessel",
                           QUESTION_OBJ.format(obj=w), _vessel_expect))
    for sd in sides:
        qs.append(Question(f"side:{sd}", "side",
                           QUESTION_SIDE.format(side=sd), _side_expect(sd)))

    # One from each family first, then the rest, so the gap is readable early.
    order: list[Question] = []
    fams = ("present", "absent", "vessel", "count", "side")
    pools = {f: [q for q in qs if q.family == f] for f in fams}
    while any(pools.values()):
        for f in fams:
            if pools[f]:
                order.append(pools[f].pop(0))
    return order


class FrameOnly(Dataset):
    def __init__(self, samples: list[dict], image_dir: str, processor):
        self.samples = samples
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        path = s["image"]
        if not os.path.isabs(path):
            path = os.path.join(self.image_dir, path)
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            return None
        pv = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return pv, s


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, []
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


@torch.no_grad()
def ask_one(model, tokenizer, loader, q: Question, device, dtype,
            yes_ids: list[int], no_ids: list[int],
            keep_text: int, with_image_token: bool = True) -> list[dict]:
    """One question over the whole shard.

    ``with_image_token`` decides whether the prompt carries ``<image>``.  The
    model splices the encoder's output at that token, and without it the
    features never reach the language model.  Running the pass without the
    token is the control that shows the encoder cannot influence an answer
    through any other path: two encoders must then agree on every frame, and a
    disagreement would mean the evaluation is leaking the encoder somewhere
    else.
    """
    ids_1 = uniform_prompt(tokenizer, q.text, with_image_token).to(device)
    records: list[dict] = []
    for pv, metas in loader:
        if pv is None:
            continue
        pv = pv.to(device=device, dtype=dtype, non_blocking=True)
        ids = ids_1.expand(pv.shape[0], -1).contiguous()
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            pixel_values=pv, max_new_tokens=max(keep_text, 1),
            do_sample=False, temperature=None, top_p=None,
            output_scores=True, return_dict_in_generate=True)
        first = out.scores[0].float()
        logp = torch.log_softmax(first, dim=-1)
        y = torch.logsumexp(logp[:, yes_ids], dim=-1)
        n = torch.logsumexp(logp[:, no_ids], dim=-1)
        margin = (y - n).tolist()
        texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        for meta, m_, t_ in zip(metas, margin, texts):
            said_yes = int(m_ > 0.0)
            exp = q.expected(meta)
            if exp is None:
                # The answer is undefined for this frame (no vessel labels, or
                # no visible vessel for a side question); asking it would add a
                # record that no rate can use.
                continue
            records.append({
                "id": meta["id"],
                "video": video_of(meta),
                "split": meta.get("split", ""),
                "condition": meta.get("condition", ""),
                "n_boxes": len(meta.get("bboxes") or []),
                "n_visible": len(visible_boxes(meta)),
                "dataset": meta.get("dataset", ""),
                "question": q.key,
                "family": q.family,
                "said_yes": said_yes,
                "expected_yes": exp,
                "correct": int(said_yes == exp),
                # The submitted work's quantity: an affirmative answer about a
                # category that is not in the frame.
                "halluc": said_yes if exp == 0 else 0,
                "yes_minus_no_logprob": round(m_, 5),
                "text": t_.strip()[:48],
            })
    return records


def part_paths(out_dir: str, label: str, q_key: str,
               mode: str = "with") -> tuple[str, str]:
    # The mode is part of the name: a with-token and a without-token pass ask
    # the same question of the same frames and would otherwise overwrite each
    # other's results, and the resume check would then reuse the wrong one.
    stem = ("ans_" + label + "__" + mode + "__"
            + q_key.replace(":", "_").replace(" ", "_"))
    d = os.path.join(out_dir, "parts")
    return os.path.join(d, stem + ".jsonl"), os.path.join(d, stem + ".ok")


def write_part(path: str, marker: str, records: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    with open(marker, "w") as f:
        json.dump({"n_records": len(records)}, f)


def read_part(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise(records: list[dict], paper_three: list[str],
              min_present_rate: float = 0.5) -> dict:
    from additional.stats import cluster_proportion, clopper_pearson

    def rate(sub, field):
        if not sub:
            return None
        return cluster_proportion([r[field] for r in sub],
                                  [r["video"] for r in sub])

    fam = {}
    for f in sorted({r["family"] for r in records}):
        sub = [r for r in records if r["family"] == f]
        fam[f] = {
            "n": len(sub),
            "yes_rate": rate(sub, "said_yes"),
            "accuracy": rate(sub, "correct"),
        }

    absent = [r for r in records if r["family"] == "absent"]
    present = [r for r in records if r["family"] == "present"]

    disc = {"n_absent": len(absent), "n_present": len(present)}
    if absent and present:
        ya = rate(absent, "said_yes")
        yp = rate(present, "said_yes")
        disc["yes_rate_absent"] = ya
        disc["yes_rate_present"] = yp
        disc["gap"] = yp["rate"] - ya["rate"]
        # A positive gap is not enough.  An encoder swap can leave the model
        # answering "no" to almost everything, and then the absent rate is low
        # while the model is also wrong about objects that are plainly there:
        # one condition here answers "yes" to water and sky, present in every
        # frame, on 4.51% of them, and its absent rate of 2.04% says nothing
        # about hallucination.  So the gate also requires the model to be right
        # more often than not when the answer is known to be yes.  The
        # threshold is a stated choice, recorded here rather than left implicit.
        disc["min_present_rate"] = min_present_rate
        disc["discriminates"] = bool(yp["rate"] - ya["rate"] > 0.0
                                     and yp["rate"] >= min_present_rate)
        disc["gate_note"] = (
            "부재 긍정률을 환각률로 읽으려면 두 조건이 필요하다. 극성 차이가 "
            "양수여야 하고, 반드시 있는 물체에 대해 절반 이상 맞혀야 한다. "
            "두 번째 조건이 없으면 거의 모든 것에 no 라고 답하는 조합이 낮은 "
            "부재 긍정률로 통과한다")

    per_question = {}
    for k in sorted({r["question"] for r in records}):
        sub = [r for r in records if r["question"] == k]
        cp = clopper_pearson(sum(r["said_yes"] for r in sub), len(sub))
        per_question[k] = {
            "n": len(sub),
            "yes_rate": cp.as_dict(),
            "accuracy": clopper_pearson(sum(r["correct"] for r in sub),
                                        len(sub)).as_dict(),
        }

    p3 = [r for r in records
          if r["family"] == "absent"
          and r["question"].split(":", 1)[-1] in paper_three]

    return {
        "n": len(records),
        "by_family": fam,
        "discrimination": disc,
        "per_question": per_question,
        "hallucination_rate_absent": rate(absent, "halluc"),
        "hallucination_rate_paper_three": rate(p3, "halluc"),
        "n_paper_three_probes": len(p3),
        "answer_text_distribution": dict(
            Counter(r["text"][:12] for r in records).most_common(8)),
    }


def parse_args():
    p = argparse.ArgumentParser("absence protocol on a yes/no capable VLM")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vlm_ckpt", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--projector_type", default="mlp2x_gelu")
    p.add_argument("--encoder", action="append", required=True,
                   help="LABEL=PATH; 'none' keeps the checkpoint's encoder. "
                        "Order matters: earlier encoders finish first, so put "
                        "the decisive comparison first.")
    p.add_argument("--absent", default="airplane,car,building",
                   help="categories absent from every frame; must be held out "
                        "of the fine-tune")
    p.add_argument("--present", default="water,sky",
                   help="categories present in every frame, for the "
                        "discrimination control")
    p.add_argument("--vessel_words", default="",
                   help="comma list; asks 'Is there a {w} in this image?' "
                        "graded by whether a vessel is visible")
    p.add_argument("--sides", default="",
                   help="comma list of left,right; asks which side of the "
                        "image the largest visible vessel is on")
    p.add_argument("--box_geometry", default="raw", choices=["raw", "cropped"],
                   help="coordinate frame the vessel-based answers are graded "
                        "in; 'cropped' uses only what the centre crop shows")
    p.add_argument("--counts", default="3,7",
                   help="vessel-count thresholds; the answer comes from the "
                        "frame's own boxes")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_token_mode", default="with",
                   choices=["with", "without"],
                   help="'without' drops the <image> token so the encoder's "
                        "output never reaches the language model; two "
                        "encoders must then agree on every frame")
    p.add_argument("--min_present_rate", type=float, default=0.5,
                   help="the absent rate is reported as a hallucination rate "
                        "only if the affirmative rate on known-present "
                        "categories reaches this; a swap that makes the model "
                        "answer no to everything would otherwise pass")
    p.add_argument("--keep_text", type=int, default=4,
                   help="tokens to generate and record; the decision is read "
                        "from the first one regardless")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = DTYPES[args.dtype]
    assert_policy_or_die()

    with open(args.data_path) as f:
        samples = json.load(f)
    if args.limit > 0:
        samples = samples[:args.limit]
    attach_visible(samples, args.box_geometry)
    shard = samples[rank::world]

    absent = [s.strip() for s in args.absent.split(",") if s.strip()]
    present = [s.strip() for s in args.present.split(",") if s.strip()]
    counts = [int(s) for s in args.counts.split(",") if s.strip()]
    vessel_words = [s.strip() for s in args.vessel_words.split(",") if s.strip()]
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    questions = build_questions(absent, present, counts, vessel_words, sides)

    if rank == 0:
        os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)
        print(f"[yesno] {len(samples):,} frames x {len(questions)} questions "
              f"= {len(samples) * len(questions):,} per encoder, world={world}",
              flush=True)
        print(f"[yesno] 순서 {[q.key for q in questions]}", flush=True)

    model = build_vlm(args.vlm_ckpt, args.llm, args.vision_model,
                      args.projector_type, dtype, device)
    tokenizer = model.tokenizer
    yes_ids = first_token_ids(tokenizer, _YES_FORMS)
    no_ids = first_token_ids(tokenizer, _NO_FORMS)
    if set(yes_ids) & set(no_ids):
        raise RuntimeError("yes and no share a first token; the read-out "
                           "cannot separate them")
    if not (yes_ids and no_ids):
        raise RuntimeError("could not resolve yes/no token ids")
    stock_state = snapshot_encoder(model)

    summary: dict[str, dict] = {}
    for spec in args.encoder:
        if "=" not in spec:
            raise SystemExit(f"--encoder needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        report = swap_encoder(model, path, stock_state=stock_state)
        processor = model.vision_encoder.image_processor
        loader = DataLoader(FrameOnly(shard, args.image_dir, processor),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)

        for i_q, q in enumerate(questions, 1):
            p_part, p_ok = part_paths(args.out_dir, label, q.key,
                                      args.image_token_mode)
            skip = 0
            if rank == 0:
                skip = 1 if (os.path.exists(p_ok)
                             and os.path.exists(p_part)) else 0
            if distributed:
                flag = torch.tensor([skip], device=device)
                dist.broadcast(flag, src=0)
                skip = int(flag.item())
            if skip:
                if rank == 0:
                    print(f"[yesno] {label} [{i_q}/{len(questions)}] "
                          f"{q.key}: 기존 결과 재사용", flush=True)
                continue

            t0 = time.time()
            recs = ask_one(model, tokenizer, loader, q, device, dtype,
                           yes_ids, no_ids, args.keep_text,
                           args.image_token_mode == "with")
            if distributed:
                gathered: list[list[dict]] = [None] * world  # type: ignore[list-item]
                dist.all_gather_object(gathered, recs)
                merged_q = [r for part in gathered for r in part]
            else:
                merged_q = recs

            if rank == 0:
                merged_q.sort(key=lambda r: r["id"])
                write_part(p_part, p_ok, merged_q)
                dt = time.time() - t0
                n = len(merged_q)
                sy = sum(r["said_yes"] for r in merged_q)
                acc = sum(r["correct"] for r in merged_q)
                print(f"[yesno] {label} [{i_q}/{len(questions)}] {q.key} "
                      f"({q.family}): yes {sy}/{n} = "
                      f"{100.0 * sy / max(n, 1):.4f}%  "
                      f"정답 {100.0 * acc / max(n, 1):.4f}%  "
                      f"{dt:.0f}s {n / max(dt, 1e-9):.1f}/s", flush=True)

        if rank == 0:
            merged: list[dict] = []
            for q in questions:
                p_part, _ = part_paths(args.out_dir, label, q.key,
                                       args.image_token_mode)
                merged.extend(read_part(p_part))
            out_path = os.path.join(
                args.out_dir, f"ans_{label}__{args.image_token_mode}.jsonl")
            with open(out_path, "w") as f:
                for r in merged:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats = summarise(merged, absent[:3], args.min_present_rate)
            stats["encoder"] = report
            stats["image_token_mode"] = args.image_token_mode
            stats["answers_path"] = out_path
            summary[f"{label}__{args.image_token_mode}"] = stats

            d = stats["discrimination"]
            print(f"\n[yesno] ===== {label} "
                  f"({args.image_token_mode}) =====", flush=True)
            if "gap" in d:
                print(f"  판별력: 존재 긍정률 {d['yes_rate_present']['rate']*100:.4f}% "
                      f"− 부재 긍정률 {d['yes_rate_absent']['rate']*100:.4f}% "
                      f"= {d['gap']*100:+.4f}%p", flush=True)
            h = stats["hallucination_rate_absent"]
            if h:
                cb = h["cluster_bootstrap"]
                print(f"  부재 환각률 {h['k']}/{h['n']} = {h['rate']*100:.4f}%  "
                      f"비디오 부트스트랩 95% "
                      f"[{cb['lo']*100:.4f}, {cb['hi']*100:.4f}]", flush=True)
            for f_, v in stats["by_family"].items():
                print(f"  {f_:8s} n={v['n']:,} 정답률 "
                      f"{v['accuracy']['rate']*100:.4f}%", flush=True)

            meta = {
                "data_path": args.data_path, "vlm_ckpt": args.vlm_ckpt,
                "llm": args.llm,
                "absent_categories": absent, "present_categories": present,
                "count_thresholds": counts,
                "image_token_mode": args.image_token_mode,
                "scoring": ("first generated token, affirmative mass against "
                            "negative mass; the generated string is recorded "
                            "alongside"),
                "gate": ("the absent rate is a hallucination rate only if the "
                         "present-minus-absent gap is clearly positive; "
                         "otherwise it is an answer bias"),
                "dtype": args.dtype, "world_size": world,
                "n_frames": len(samples), "results": summary,
            }
            with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    if rank == 0:
        print(f"[yesno] summary -> "
              f"{os.path.join(args.out_dir, 'summary.json')}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

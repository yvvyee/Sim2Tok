"""
Large-scale absence-probe hallucination measurement.

The submitted work reports 5 hallucinations in 864 probes for the baseline and 0
for GAA.  That is 288 SMD frames times three absent-object categories, and at
those counts the comparison does not reach significance: Fisher's exact test
gives p = 0.062 treating frames as independent, and p = 0.239 once frames are
clustered by video.  This module measures the same quantity on far more probes
so the interval is set by data rather than by scarcity.

Two things are deliberately made explicit rather than assumed.

**The image token.**  ``VLM.prepare_inputs_labels_for_multimodal`` splices the
image features at the position of the ``<image>`` token.  When the prompt has no
such token it takes the other branch, which appends ``N-1`` zero embeddings with
attention mask 0 and never inserts ``image_features``, so the language model is
conditioned on text alone and the vision encoder cannot affect the answer.  The
script that produced the submitted 864-probe number is not in the repository, so
which branch it took cannot be read off the source.  ``--image_token_mode``
therefore runs either protocol, and running both settles it by measurement:
under ``without``, swapping the encoder must leave every answer unchanged.

**Clustering.**  Each probe records the video it came from, so the rate is
reported with both the exact binomial interval and a video-level bootstrap.

Batching keeps the protocol intact.  Probes are grouped by object category, so
every prompt in a batch is token-for-token identical and no padding is needed;
only the image varies.  Generation stays greedy with ``max_new_tokens=16`` and
the answer is matched exactly as the submitted work matched it.

Usage::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.halluc_probe \\
        --data_path data/smd_sweep60.json --image_dir frames \\
        --vlm_ckpt paper_ckpt/ctrl_baseline_fix \\
        --llm /home/yvvyee/data/Models/Llama-3.1-8B-Instruct \\
        --encoder stock=none \\
        --encoder gaa=paper_ckpt/geo_encoders_fix/inbox_ce_vision_encoder.bin \\
        --out_dir results/halluc_sweep --image_token_mode with
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

#: Object categories guaranteed absent from open-water maritime imagery.
#: The first three are the submitted set, kept first so the original three-way
#: rate can be recomputed from the same records.
ABSENT_PAPER = ["airplane", "car", "building"]
ABSENT_EXTRA = [
    "train", "bicycle", "motorcycle", "traffic light", "fire hydrant",
    "stop sign", "bench", "horse", "sheep", "cow", "elephant", "zebra",
    "refrigerator", "television", "laptop", "book", "clock",
]

#: Things present in every frame of maritime footage.  These exist to measure
#: whether the read-out discriminates at all.  A model that answers "yes" to
#: everything scores a high rate on the absent categories without seeing
#: anything, and reporting that rate as a hallucination rate would be wrong:
#: the number would be a yes-bias, and an encoder that lowered it could be
#: lowering the bias rather than the hallucination.  The gap between the
#: expected-yes and expected-no rates is what makes the absent-object rate
#: interpretable, so it is measured before the expensive pass and reported
#: alongside it.
PRESENT_ALWAYS = ["water", "sea", "sky", "horizon"]

#: Asked of every frame, but the expected answer follows that frame's boxes:
#: the SMD frame set includes open-water frames with no annotated vessel.  This
#: is the only category whose expected answer varies frame by frame, so it
#: tests presence detection rather than a constant.
VESSEL_OBJECT = "ship"

QUESTION = "Is there a {obj} in this image? Answer yes or no."

#: Surface forms of an affirmative and a negative first token.  Both cases and
#: both with and without a leading space, because which one the tokeniser emits
#: depends on the chat template's trailing whitespace.
_YES_FORMS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
_NO_FORMS = ("No", " No", "no", " no", "NO", " NO")


def _first_token_ids(tokenizer, forms) -> list[int]:
    """Ids of the first token of each surface form, de-duplicated."""
    ids = []
    for f in forms:
        enc = tokenizer(f, add_special_tokens=False).input_ids
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


#: Words that make an answer affirmative even when it does not start with them.
_AFFIRM = ("yes", "affirmative", "there is", "there are", "i can see",
           "correct")
_DENY = ("no", "negative", "there is no", "there are no", "not present",
         "cannot see", "i do not see", "incorrect")


def yes(answer: str) -> bool:
    """The submitted work's match rule, unchanged.

    Kept exactly as it was so the rate stays comparable with the submitted
    number.  ``verdict`` is what new tables should use.
    """
    a = answer.strip().lower()
    return a.startswith("yes") or " yes" in a[:12]


def verdict(answer: str) -> str:
    """``"yes"``, ``"no"`` or ``"unparsed"``.

    The submitted rule is a single boolean, so anything it does not recognise
    counts as "no hallucination".  That folds three different things together:
    a genuine denial, an affirmation phrased differently ("Based on the image,
    yes, ...", whose first twelve characters are "based on the"), and an empty
    or degenerate generation.  The last two inflate the apparent improvement,
    and in the direction the claim predicts: swapping the vision tower without
    re-tuning the projector could plausibly make output more degenerate, which
    would read as fewer hallucinations.

    So the three are separated and counted.  ``unparsed`` probes are reported
    on their own rather than absorbed into either rate.
    """
    a = answer.strip().lower()
    if not a:
        return "unparsed"
    if a.startswith("yes") or " yes" in a[:12]:
        return "yes"
    if a.startswith("no") or " no " in a[:12]:
        return "no"
    for d in _DENY:
        if d in a[:40]:
            return "no"
    for w in _AFFIRM:
        if w in a[:40]:
            return "yes"
    return "unparsed"


class FrameOnly(Dataset):
    """Frames for one probe pass; the prompt is identical across the batch."""

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


def build_prompt(tokenizer, obj: str, with_image_token: bool) -> torch.Tensor:
    """Token ids for one probe question, identical for every frame."""
    return uniform_prompt(tokenizer, QUESTION.format(obj=obj),
                          with_image_token)


def expected_yes_for(obj: str, meta: dict) -> int | None:
    """Whether "yes" is the correct answer for this object on this frame.

    ``None`` for a category whose presence is not established by the data, so
    the record carries no correctness claim rather than a guessed one.
    """
    if obj in ABSENT_PAPER or obj in ABSENT_EXTRA:
        return 0
    if obj in PRESENT_ALWAYS:
        return 1
    if obj == VESSEL_OBJECT:
        return 1 if (meta.get("bboxes") or []) else 0
    return None


@torch.no_grad()
def probe_one(model, tokenizer, loader, obj: str, with_image_token: bool,
              device, dtype, max_new: int, scoring: str,
              yes_ids: list[int], no_ids: list[int]) -> list[dict]:
    """One absence question over the whole shard.

    ``scoring="generate"`` is the submitted protocol: decode greedily and match
    the text.  It only works on a model that answers the question, and the VLM
    here does not -- its LoRA was fine-tuned on navigational narration, so it
    answers "This is a Head-on Situation..." to anything, which the text rule
    scores as a denial for every probe and makes the measurement vacuous.

    ``scoring="logit"`` reads the same decision one step earlier: the model
    still has to put probability somewhere at the first generated position, and
    comparing the mass on affirmative first tokens against negative ones
    recovers a yes/no decision from a model that would rather narrate.  This is
    a different read-out from the submitted one and is labelled as such
    wherever it is reported; the two are never mixed in a table.
    """
    ids_1 = build_prompt(tokenizer, obj, with_image_token).to(device)
    records: list[dict] = []
    for pv, metas in loader:
        if pv is None:
            continue
        pv = pv.to(device=device, dtype=dtype, non_blocking=True)
        b = pv.shape[0]
        ids = ids_1.expand(b, -1).contiguous()

        if scoring == "logit":
            out = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=pv, max_new_tokens=1,
                do_sample=False, temperature=None, top_p=None,
                output_scores=True, return_dict_in_generate=True)
            first = out.scores[0].float()                 # (B, vocab)
            logp = torch.log_softmax(first, dim=-1)
            y = torch.logsumexp(logp[:, yes_ids], dim=-1)
            n = torch.logsumexp(logp[:, no_ids], dim=-1)
            margin = (y - n).tolist()
            # Probability of yes conditional on the answer being yes or no.
            p_yes = torch.sigmoid(y - n).tolist()
            top = first.argmax(dim=-1).tolist()
            for meta, m_, p_, t_ in zip(metas, margin, p_yes, top):
                said_yes = int(m_ > 0.0)
                exp = expected_yes_for(obj, meta)
                records.append({
                    "id": meta["id"],
                    "video": video_of(meta),
                    "split": meta.get("split", ""),
                    "condition": meta.get("condition", ""),
                    "object": obj,
                    "answer": tokenizer.decode([t_]).strip()[:32],
                    "yes_minus_no_logprob": round(m_, 5),
                    "p_yes_given_binary": round(p_, 6),
                    "said_yes": said_yes,
                    "expected_yes": exp,
                    "correct": None if exp is None else int(said_yes == exp),
                    # "halluc" stays "said yes to an absent object", so the
                    # submitted rate is recomputable from these records.
                    "halluc": said_yes if exp == 0 else 0,
                    "verdict": "yes" if said_yes else "no",
                })
            continue

        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            pixel_values=pv, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for meta, text in zip(metas, texts):
            said_yes = int(yes(text))
            exp = expected_yes_for(obj, meta)
            records.append({
                "id": meta["id"],
                "video": video_of(meta),
                "split": meta.get("split", ""),
                "condition": meta.get("condition", ""),
                "object": obj,
                "answer": text.strip()[:160],
                "said_yes": said_yes,
                "expected_yes": exp,
                "correct": None if exp is None else int(said_yes == exp),
                "halluc": said_yes if exp == 0 else 0,
                "verdict": verdict(text),
            })
    return records


#: Which categories a run probes.  ``control`` is the read-out validity check:
#: the submitted three absent categories together with things present in every
#: frame, so the expected-yes and expected-no rates are measured on the same
#: frames by the same pass.
OBJECT_SETS = {
    "paper": ABSENT_PAPER,
    "extended": ABSENT_PAPER + ABSENT_EXTRA,
    "control": ABSENT_PAPER + PRESENT_ALWAYS + [VESSEL_OBJECT],
    # Present and absent categories are interleaved at the front rather than
    # grouped.  A pass over the full frame set runs for hours per encoder, and
    # objects are persisted one at a time, so putting both polarities first
    # makes the expected-yes minus expected-no gap readable after the opening
    # objects instead of after the whole pass.  The gap is what decides whether
    # the rest of the pass is worth its cost, so it has to arrive early.
    "extended_control": (
        ["water", "airplane", "sea", "car", VESSEL_OBJECT, "building",
         "sky", "horizon"]
        + [o for o in ABSENT_EXTRA]),
}


def discrimination(records: list[dict]) -> dict:
    """Yes-rate where "yes" is correct against where it is wrong.

    The absent-object rate on its own does not separate a model that sees
    things that are not there from a model that answers "yes" regardless.  If
    both rates here are equally high, the read-out has measured a yes-bias and
    the absent-object rate carries no claim about hallucination.  The gap is
    reported so that the expensive pass can be read, or discarded, on evidence.
    """
    from additional.stats import cluster_proportion

    pos = [r for r in records if r.get("expected_yes") == 1]
    neg = [r for r in records if r.get("expected_yes") == 0]
    out: dict = {"n_expected_yes": len(pos), "n_expected_no": len(neg)}
    if pos:
        out["yes_rate_where_present"] = cluster_proportion(
            [r["said_yes"] for r in pos], [r["video"] for r in pos])
    if neg:
        out["yes_rate_where_absent"] = cluster_proportion(
            [r["said_yes"] for r in neg], [r["video"] for r in neg])
    if pos and neg:
        out["gap"] = (out["yes_rate_where_present"]["rate"]
                      - out["yes_rate_where_absent"]["rate"])
    return out


def part_paths(out_dir: str, key: str, obj: str) -> tuple[str, str]:
    """Where one object's probes and its completion marker live.

    A pass over the full SMD frame set is 585,060 probes per encoder, which
    runs for hours.  Writing only at the end of an encoder means an
    interruption at hour nine leaves nothing, so each object is persisted as it
    finishes and a separate marker file records that the write completed.  The
    marker exists because a probe file cannot be checked by line count: frames
    whose image fails to open are skipped, so a short file is not necessarily a
    truncated one.
    """
    stem = f"probes_{key}__{obj.replace(' ', '_')}"
    d = os.path.join(out_dir, "parts")
    return os.path.join(d, stem + ".jsonl"), os.path.join(d, stem + ".ok")


def write_part(path: str, marker: str, records: list[dict]) -> None:
    """Write one object's records, then its marker, never the other order."""
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


def parse_args():
    p = argparse.ArgumentParser("large-scale absence-probe hallucination test")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vlm_ckpt", required=True,
                   help="directory with projector.bin and the LoRA adapter")
    p.add_argument("--llm", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--projector_type", default="mlp2x_gelu")
    p.add_argument("--encoder", action="append", required=True,
                   help="LABEL=PATH; PATH 'none' keeps the checkpoint's own "
                        "encoder. Repeatable.")
    p.add_argument("--objects", default="paper",
                   choices=["paper", "extended", "control",
                            "extended_control"],
                   help="'paper' = the submitted three absent categories; "
                        "'extended' = those three plus 17 more; "
                        "'control' = the three plus categories present in "
                        "every frame, to measure whether the read-out "
                        "discriminates at all; 'extended_control' = both")
    p.add_argument("--image_token_mode", default="with",
                   choices=["with", "without", "both"],
                   help="whether the prompt carries the <image> token that "
                        "makes the encoder's output reach the language model")
    p.add_argument("--scoring", default="generate",
                   choices=["generate", "logit"],
                   help="'generate' is the submitted protocol: decode greedily "
                        "and match the text. 'logit' compares the probability "
                        "mass on affirmative versus negative first tokens, "
                        "which recovers a decision from a model whose "
                        "fine-tuning makes it narrate instead of answering. "
                        "Different read-outs; never mix them in one table.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=16)
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
    shard = samples[rank::world]

    objects = OBJECT_SETS[args.objects]
    modes = (["with", "without"] if args.image_token_mode == "both"
             else [args.image_token_mode])

    if rank == 0:
        os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)
        print(f"[probe] {len(samples):,} frames x {len(objects)} objects "
              f"x {len(modes)} mode(s) = "
              f"{len(samples) * len(objects) * len(modes):,} probes, "
              f"world={world}", flush=True)

    model = build_vlm(args.vlm_ckpt, args.llm, args.vision_model,
                      args.projector_type, dtype, device)
    tokenizer = model.tokenizer
    yes_ids = _first_token_ids(tokenizer, _YES_FORMS)
    no_ids = _first_token_ids(tokenizer, _NO_FORMS)
    if args.scoring == "logit" and not (yes_ids and no_ids):
        raise RuntimeError("could not resolve yes/no token ids for this "
                           "tokeniser; logit scoring is not possible")
    if rank == 0 and args.scoring == "logit":
        print(f"[probe] yes ids {yes_ids} / no ids {no_ids}", flush=True)
    # Taken before any swap: every --encoder pass mutates this one model in
    # place, so the stock condition is only stock if the weights are restored.
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

        for mode in modes:
            key = f"{label}__{mode}__{args.scoring}"

            for i_obj, obj in enumerate(objects, 1):
                p_part, p_ok = part_paths(args.out_dir, key, obj)

                # Rank 0 alone decides whether this object is already on disk,
                # and the decision is broadcast.  A rank that decided
                # differently from its peers would enter a collective the
                # others skipped, and the job would hang rather than fail.
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
                        print(f"[probe] {key} [{i_obj}/{len(objects)}] "
                              f"{obj}: 기존 결과 재사용", flush=True)
                    continue

                t0 = time.time()
                recs = probe_one(model, tokenizer, loader, obj,
                                 mode == "with", device, dtype,
                                 args.max_new_tokens, args.scoring,
                                 yes_ids, no_ids)
                if distributed:
                    gathered: list[list[dict]] = [None] * world  # type: ignore[list-item]
                    dist.all_gather_object(gathered, recs)
                    merged_obj = [r for part in gathered for r in part]
                else:
                    merged_obj = recs

                if rank == 0:
                    merged_obj.sort(key=lambda r: r["id"])
                    write_part(p_part, p_ok, merged_obj)
                    dt = time.time() - t0
                    k = sum(r["halluc"] for r in merged_obj)
                    n = len(merged_obj)
                    print(f"[probe] {key} [{i_obj}/{len(objects)}] "
                          f"{obj}: {k}/{n} = {100.0 * k / max(n, 1):.4f}%  "
                          f"{dt:.0f}s  "
                          f"{n / max(dt, 1e-9):.1f} probes/s", flush=True)

            if rank == 0:
                merged: list[dict] = []
                for obj in objects:
                    p_part, _ = part_paths(args.out_dir, key, obj)
                    merged.extend(read_part(p_part))
                merged.sort(key=lambda r: (r["object"], r["id"]))
                path_out = os.path.join(args.out_dir, f"probes_{key}.jsonl")
                with open(path_out, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                from additional.stats import cluster_proportion, clopper_pearson
                overall = cluster_proportion([r["halluc"] for r in merged],
                                             [r["video"] for r in merged])
                by_verdict = Counter(r["verdict"] for r in merged)
                parsed = [r for r in merged if r["verdict"] != "unparsed"]
                overall_parsed = (
                    cluster_proportion([int(r["verdict"] == "yes")
                                        for r in parsed],
                                       [r["video"] for r in parsed])
                    if parsed else None)
                paper3 = [r for r in merged if r["object"] in ABSENT_PAPER]
                summary[key] = {
                    "encoder": report,
                    "image_token_mode": mode,
                    "n_probes": len(merged),
                    "n_objects": len(objects),
                    "overall": overall,
                    "verdict_counts": dict(by_verdict),
                    "n_unparsed": by_verdict.get("unparsed", 0),
                    "overall_excluding_unparsed": overall_parsed,
                    "paper_three_objects": (
                        cluster_proportion([r["halluc"] for r in paper3],
                                           [r["video"] for r in paper3])
                        if paper3 else None),
                    "by_object": {
                        o: clopper_pearson(
                            sum(r["halluc"] for r in merged if r["object"] == o),
                            sum(1 for r in merged if r["object"] == o)).as_dict()
                        for o in objects
                    },
                    "discrimination": discrimination(merged),
                    "probes_path": path_out,
                }
                ex = overall["exact_binomial"]
                cb = overall["cluster_bootstrap"]
                print(f"[probe] {key:24s} {overall['k']}/{overall['n']} = "
                      f"{overall['rate'] * 100:.4f}%  "
                      f"exact 95% [{ex['lo'] * 100:.4f}, {ex['hi'] * 100:.4f}]  "
                      f"video-boot [{cb['lo'] * 100:.4f}, {cb['hi'] * 100:.4f}]  "
                      f"미분류 {by_verdict.get('unparsed', 0)}", flush=True)

    if rank == 0:
        meta = {
            "data_path": args.data_path,
            "vlm_ckpt": args.vlm_ckpt,
            "llm": args.llm,
            "vision_model": args.vision_model,
            "objects": objects,
            "question": QUESTION,
            "scoring": args.scoring,
            "match_rule": ("answer.startswith('yes') or ' yes' in answer[:12]"
                           if args.scoring == "generate"
                           else "logsumexp over yes first-token ids minus the "
                                "same over no ids, positive counts as yes"),
            "yes_token_ids": yes_ids,
            "no_token_ids": no_ids,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "world_size": world,
            "n_frames": len(samples),
            "results": summary,
        }
        out = os.path.join(args.out_dir, "summary.json")
        with open(out, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[probe] summary -> {out}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Grade the model's own narration against the frame's annotated conditions.

The absence-probe route is closed on this VLM.  Its LoRA was fine-tuned on
navigational narration, and the measurement shows it does not answer an absence
question in any read-out:

* Free generation produces seven distinct response openings across 70,800
  probes, 94.1% of them the same template, and every one of them is
  ``unparsed`` under the submitted match rule.  Both conditions therefore score
  0/864 and the submitted baseline of 5 events is not reproduced.
* Constrained first-token scoring does answer, but it does not discriminate.
  On the full 29,253-frame set the model says "yes" to ``airplane`` (absent) in
  86.88% of frames and to ``water`` (present in every maritime frame) in
  82.93%.  The gap is -3.95 points, so the affirmative rate is a yes-bias and
  an encoder that lowered it could be lowering the bias rather than the
  hallucination.

What the model does do is assert scene attributes, and some of those are
annotated in SMD.  The narration claims a visibility state ("in extremely
restricted visibility"), and every SMD frame carries a condition label of
VIS_Clear, VIS_Haze or NIR.  A restricted-visibility claim on a VIS_Clear frame
is a scene attribute the model reports without the image supporting it, which
is gradeable in the model's native output mode rather than in a question format
it cannot use.

This is a different quantity from the submitted work's object hallucination.
It is reported under its own name and never substituted for that number.

Two rates are reported per encoder, because they move independently:

``false_claim_rate``
    Restricted visibility claimed on a VIS_Clear frame.  Lower is a narration
    that asserts less than the image supports.

``sensitivity``
    Restricted visibility claimed on a VIS_Haze frame.  This is not a
    hallucination rate; it is reported alongside so that a drop in the false
    claim rate cannot be read as an improvement when it is only the model
    asserting the attribute less often overall.

Usage::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.narration_eval \\
        --data_path data/smd_full.json --image_dir frames \\
        --vlm_ckpt paper_ckpt/ctrl_baseline_fix \\
        --llm /home/yvvyee/data/Models/Llama-3.1-8B-Instruct \\
        --encoder stock=none \\
        --encoder gaa=paper_ckpt/geo_encoders_fix/inbox_ce_vision_encoder.bin \\
        --out_dir results/narration_full
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

#: The model's own task prompt, carried over from ``gaa/vlm_behavior_eval.py``
#: so the narration measured here is the narration the submitted work elicits.
PROMPT = ("Describe the navigational situation in this image and state the "
          "required action under the COLREG rules.")

#: Phrases by which the narration asserts that visibility is impaired.
#: Matched on word boundaries rather than as substrings, because a substring
#: rule counts "mist" inside "mistake" and would inflate the claim rate with
#: text that says nothing about visibility.
RESTRICTED_PHRASES = (r"restricted visibility", r"reduced visibility",
                      r"poor visibility", r"limited visibility",
                      r"low visibility", r"restricted dawn",
                      r"fog", r"foggy", r"haze", r"hazy", r"mist", r"misty",
                      r"murky", r"obscured")

#: Phrases asserting a time of day.  SMD does not annotate time of day, so
#: these are counted and reported without being graded.
TIME_PHRASES = (r"sunset", r"dawn", r"dusk", r"night", r"nighttime",
                r"sunrise")


def _boundary_re(phrases):
    return re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.IGNORECASE)


_RESTRICTED_RE = _boundary_re(RESTRICTED_PHRASES)
_TIME_RE = _boundary_re(TIME_PHRASES)

#: The condition labels SMD carries.  VIS_Clear is the ungraded-claim negative
#: case and VIS_Haze the positive one.  NIR is near-infrared imagery, whose
#: relation to a visibility claim is not a simple true or false, so it is
#: reported separately rather than folded into either rate.
CLEAR = "VIS_Clear"
HAZE = "VIS_Haze"


def claims_restricted(text: str) -> bool:
    return bool(_RESTRICTED_RE.search(text))


def claims_time(text: str) -> bool:
    return bool(_TIME_RE.search(text))


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
def run_pass(model, tokenizer, loader, with_image_token: bool, device, dtype,
             max_new: int) -> list[dict]:
    ids_1 = uniform_prompt(tokenizer, PROMPT, with_image_token).to(device)
    records: list[dict] = []
    for pv, metas in loader:
        if pv is None:
            continue
        pv = pv.to(device=device, dtype=dtype, non_blocking=True)
        ids = ids_1.expand(pv.shape[0], -1).contiguous()
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            pixel_values=pv, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None)
        # Only the continuation, so a phrase in the prompt cannot be matched.
        gen = out[:, ids.shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for meta, text in zip(metas, texts):
            records.append({
                "id": meta["id"],
                "video": video_of(meta),
                "split": meta.get("split", ""),
                "condition": meta.get("condition", ""),
                "n_boxes": len(meta.get("bboxes") or []),
                "claims_restricted": int(claims_restricted(text)),
                "claims_time_of_day": int(claims_time(text)),
                "text": text.strip()[:400],
            })
    return records


def summarise(records: list[dict]) -> dict:
    from additional.stats import cluster_proportion, clopper_pearson

    def rate(sub, field="claims_restricted"):
        if not sub:
            return None
        return cluster_proportion([r[field] for r in sub],
                                  [r["video"] for r in sub])

    clear = [r for r in records if r["condition"] == CLEAR]
    haze = [r for r in records if r["condition"] == HAZE]

    per_condition = {}
    for cond in sorted({r["condition"] for r in records}):
        sub = [r for r in records if r["condition"] == cond]
        per_condition[cond] = {
            "n": len(sub),
            "n_videos": len({r["video"] for r in sub}),
            "claims_restricted": rate(sub),
            "claims_time_of_day": rate(sub, "claims_time_of_day"),
        }

    out = {
        "n": len(records),
        "per_condition": per_condition,
        "false_claim_rate_on_clear": rate(clear),
        "sensitivity_on_haze": rate(haze),
        "overall_claim_rate": rate(records),
        "distinct_openings": dict(
            Counter(r["text"][:60] for r in records).most_common(10)),
        "n_distinct_openings": len({r["text"][:60] for r in records}),
    }
    if clear and haze:
        out["haze_minus_clear"] = (out["sensitivity_on_haze"]["rate"]
                                   - out["false_claim_rate_on_clear"]["rate"])
    # A degenerate narration asserts nothing anywhere, which would drive the
    # false claim rate to zero without the model reading the image better.
    # Recorded so that reading cannot skip it.
    out["asserts_attribute_anywhere"] = bool(
        out["overall_claim_rate"] and out["overall_claim_rate"]["k"] > 0)
    return out


def parse_args():
    p = argparse.ArgumentParser("grade narration claims against annotations")
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vlm_ckpt", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--vision_model", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--projector_type", default="mlp2x_gelu")
    p.add_argument("--encoder", action="append", required=True,
                   help="LABEL=PATH; 'none' keeps the checkpoint's encoder")
    p.add_argument("--image_token_mode", default="with",
                   choices=["with", "without", "both"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)
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

    modes = (["with", "without"] if args.image_token_mode == "both"
             else [args.image_token_mode])

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[narr] {len(samples):,} frames x {len(modes)} mode(s), "
              f"world={world}, max_new_tokens={args.max_new_tokens}",
              flush=True)
        print(f"[narr] condition "
              f"{dict(Counter(s.get('condition', '') for s in samples))}",
              flush=True)

    model = build_vlm(args.vlm_ckpt, args.llm, args.vision_model,
                      args.projector_type, dtype, device)
    tokenizer = model.tokenizer
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
            t0 = time.time()
            records = run_pass(model, tokenizer, loader, mode == "with",
                               device, dtype, args.max_new_tokens)
            if distributed:
                gathered: list[list[dict]] = [None] * world  # type: ignore[list-item]
                dist.all_gather_object(gathered, records)
                merged = [r for part in gathered for r in part]
            else:
                merged = records

            key = f"{label}__{mode}"
            if rank == 0:
                merged.sort(key=lambda r: r["id"])
                out_path = os.path.join(args.out_dir, f"narr_{key}.jsonl")
                with open(out_path, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats = summarise(merged)
                stats["encoder"] = report
                stats["image_token_mode"] = mode
                stats["narr_path"] = out_path
                stats["wall_seconds"] = round(time.time() - t0, 1)
                summary[key] = stats

                fc = stats["false_claim_rate_on_clear"]
                sn = stats["sensitivity_on_haze"]
                print(f"[narr] {key:22s} {len(merged):,} frames "
                      f"{stats['wall_seconds']:.0f}s "
                      f"서술 종류 {stats['n_distinct_openings']}", flush=True)
                if fc:
                    cb = fc["cluster_bootstrap"]
                    print(f"       VIS_Clear 허위 주장 {fc['k']}/{fc['n']} = "
                          f"{fc['rate'] * 100:.4f}%  "
                          f"비디오 부트스트랩 95% "
                          f"[{cb['lo'] * 100:.4f}, {cb['hi'] * 100:.4f}]",
                          flush=True)
                if sn:
                    cb = sn["cluster_bootstrap"]
                    print(f"       VIS_Haze  민감도    {sn['k']}/{sn['n']} = "
                          f"{sn['rate'] * 100:.4f}%  "
                          f"비디오 부트스트랩 95% "
                          f"[{cb['lo'] * 100:.4f}, {cb['hi'] * 100:.4f}]",
                          flush=True)

    if rank == 0:
        meta = {
            "data_path": args.data_path,
            "vlm_ckpt": args.vlm_ckpt,
            "llm": args.llm,
            "prompt": PROMPT,
            "restricted_phrases": list(RESTRICTED_PHRASES),
            "time_phrases": list(TIME_PHRASES),
            "measures": ("scene-attribute claim graded against the SMD "
                         "condition label; this is not the submitted work's "
                         "object hallucination number and does not replace it"),
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "world_size": world,
            "n_samples": len(samples),
            "results": summary,
        }
        out = os.path.join(args.out_dir, "summary.json")
        with open(out, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[narr] summary -> {out}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

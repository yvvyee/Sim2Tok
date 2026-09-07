"""
COLREGs encounter classification on the 10,000-sample corpus.

The submitted work lists this as a limitation: "COLREG reasoning accuracy did
not measurably improve in our setup".  That statement rests on the 100-episode
20260227 set, where the ground-truth encounter type is recovered by keyword
matching over the English description text, so a null result there is hard to
separate from measurement noise.

The AIVN-SDS corpus makes the same measurement much sharper.  Its file names
carry the scenario the simulator generated (``unity_crossing``,
``unity_headon``, ``unity_overtaking``, each also in a ``_collision`` variant),
so the label is exact rather than inferred from prose, and there are 10,000 of
them instead of 100.  The corpus is also entirely held out for the submitted
VLM, which was fine-tuned on the 264-pair OceanSim pool, so no split is needed.

Protocol and scoring are carried over unchanged from
``gaa/vlm_behavior_eval.py``: the same one-word classification prompt, the same
tolerant ``parse_sit`` keyword matching, and greedy generation with
``max_new_tokens=24``.  Only the label source and the sample count change.

The corpus contains no "safe" scenario, so a "safe" or "unknown" prediction is
counted as wrong and reported separately rather than folded into an accuracy
number that would hide it.

Usage::

    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 \\
        -m additional.colregs_eval \\
        --data_path data/aivn_10k_describe.json \\
        --image_dir /home/yvvyee/data/AIVN-SDS/20260728/png \\
        --vlm_ckpt paper_ckpt/ctrl_baseline_fix \\
        --llm /home/yvvyee/data/Models/Llama-3.1-8B-Instruct \\
        --encoder stock=none \\
        --encoder gaa=paper_ckpt/geo_encoders_fix/inbox_ce_vision_encoder.bin \\
        --out_dir results/colregs_10k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from additional.config import assert_policy_or_die
from additional.vlm_loader import (build_vlm, snapshot_encoder,
                                   swap_encoder, uniform_prompt)

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}

PROMPT = ("Look at the vessels. Classify the COLREG encounter situation as "
          "exactly one word: head-on, crossing, overtaking, or safe. "
          "Answer with only that one word.")

#: AIVN-SDS scenario label -> the class name the prompt asks for.
SCENARIO_TO_CLASS = {"headon": "head-on", "crossing": "crossing",
                     "overtaking": "overtaking"}

CLASSES = ["head-on", "crossing", "overtaking", "safe", "unknown"]


def parse_sit(answer: str) -> str:
    """``gaa/vlm_behavior_eval.py`` matching rule, unchanged."""
    a = answer.strip().lower()
    if "head-on" in a or "head on" in a or "headon" in a:
        return "head-on"
    if "overtak" in a:
        return "overtaking"
    if "crossing" in a or "cross " in a or a.startswith("cross"):
        return "crossing"
    if any(s in a for s in ("no risk", "safe", "none", "no collision",
                            "no encounter", "clear")):
        return "safe"
    return "unknown"


class ImageOnly(Dataset):
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
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for meta, text in zip(metas, texts):
            gt = SCENARIO_TO_CLASS.get(meta.get("scenario", ""), "unknown")
            pred = parse_sit(text)
            records.append({
                "id": meta["id"],
                "scenario": meta.get("scenario", ""),
                "collision": bool(meta.get("collision", False)),
                "gt": gt,
                "pred": pred,
                "correct": int(pred == gt),
                "raw": text.strip()[:96],
            })
    return records


def summarise(records: list[dict]) -> dict:
    """Accuracy overall and per scenario, plus the full confusion matrix."""
    from additional.stats import clopper_pearson

    n = len(records)
    if n == 0:
        return {"n": 0}
    acc = clopper_pearson(sum(r["correct"] for r in records), n)

    per_scenario = {}
    for sc in sorted({r["scenario"] for r in records}):
        sub = [r for r in records if r["scenario"] == sc]
        per_scenario[sc] = {
            **clopper_pearson(sum(r["correct"] for r in sub), len(sub)).as_dict(),
            "pred_distribution": dict(Counter(r["pred"] for r in sub)),
        }

    per_collision = {}
    for flag in (False, True):
        sub = [r for r in records if r["collision"] is flag]
        if sub:
            per_collision[str(flag)] = clopper_pearson(
                sum(r["correct"] for r in sub), len(sub)).as_dict()

    confusion = {g: dict(Counter(r["pred"] for r in records if r["gt"] == g))
                 for g in sorted({r["gt"] for r in records})}

    return {
        "n": n,
        "accuracy": acc.as_dict(),
        "per_scenario": per_scenario,
        "per_collision": per_collision,
        "confusion": confusion,
        "prediction_distribution": dict(Counter(r["pred"] for r in records)),
        "n_safe_or_unknown": sum(1 for r in records
                                 if r["pred"] in ("safe", "unknown")),
    }


def parse_args():
    p = argparse.ArgumentParser("COLREGs encounter classification at scale")
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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=24)
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
    samples = [s for s in samples if s.get("scenario") in SCENARIO_TO_CLASS]
    if args.limit > 0:
        samples = samples[:args.limit]
    shard = samples[rank::world]

    modes = (["with", "without"] if args.image_token_mode == "both"
             else [args.image_token_mode])

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[colregs] {len(samples):,} labelled samples x {len(modes)} "
              f"mode(s), world={world}", flush=True)
        print(f"[colregs] label distribution "
              f"{dict(Counter(s['scenario'] for s in samples))}", flush=True)

    model = build_vlm(args.vlm_ckpt, args.llm, args.vision_model,
                      args.projector_type, dtype, device)
    tokenizer = model.tokenizer
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
        loader = DataLoader(ImageOnly(shard, args.image_dir, processor),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)

        for mode in modes:
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
                out_path = os.path.join(args.out_dir, f"preds_{key}.jsonl")
                with open(out_path, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats = summarise(merged)
                stats["encoder"] = report
                stats["image_token_mode"] = mode
                stats["preds_path"] = out_path
                summary[key] = stats
                a = stats["accuracy"]
                print(f"[colregs] {key:20s} acc {a['k']}/{a['n']} = "
                      f"{a['point'] * 100:.2f}% "
                      f"95% [{a['lo'] * 100:.2f}, {a['hi'] * 100:.2f}]  "
                      f"safe/unknown {stats['n_safe_or_unknown']}", flush=True)

    if rank == 0:
        meta = {
            "data_path": args.data_path,
            "vlm_ckpt": args.vlm_ckpt,
            "llm": args.llm,
            "prompt": PROMPT,
            "scoring": "gaa/vlm_behavior_eval.py parse_sit, unchanged",
            "label_source": "AIVN-SDS file-name scenario field",
            "classes": CLASSES,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "world_size": world,
            "n_samples": len(samples),
            "results": summary,
        }
        out = os.path.join(args.out_dir, "summary.json")
        with open(out, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[colregs] summary -> {out}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

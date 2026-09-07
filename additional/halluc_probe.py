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

QUESTION = "Is there a {obj} in this image? Answer yes or no."


def yes(answer: str) -> bool:
    """The submitted work's match rule, unchanged."""
    a = answer.strip().lower()
    return a.startswith("yes") or " yes" in a[:12]


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


@torch.no_grad()
def probe_one(model, tokenizer, loader, obj: str, with_image_token: bool,
              device, dtype, max_new: int) -> list[dict]:
    ids_1 = build_prompt(tokenizer, obj, with_image_token).to(device)
    records: list[dict] = []
    for pv, metas in loader:
        if pv is None:
            continue
        pv = pv.to(device=device, dtype=dtype, non_blocking=True)
        b = pv.shape[0]
        ids = ids_1.expand(b, -1).contiguous()
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            pixel_values=pv, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for meta, text in zip(metas, texts):
            records.append({
                "id": meta["id"],
                "video": video_of(meta),
                "split": meta.get("split", ""),
                "condition": meta.get("condition", ""),
                "object": obj,
                "answer": text.strip()[:64],
                "halluc": int(yes(text)),
            })
    return records


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
                   choices=["paper", "extended"],
                   help="'paper' = the submitted three categories; "
                        "'extended' = those three plus 17 more")
    p.add_argument("--image_token_mode", default="with",
                   choices=["with", "without", "both"],
                   help="whether the prompt carries the <image> token that "
                        "makes the encoder's output reach the language model")
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

    objects = (ABSENT_PAPER if args.objects == "paper"
               else ABSENT_PAPER + ABSENT_EXTRA)
    modes = (["with", "without"] if args.image_token_mode == "both"
             else [args.image_token_mode])

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[probe] {len(samples):,} frames x {len(objects)} objects "
              f"x {len(modes)} mode(s) = "
              f"{len(samples) * len(objects) * len(modes):,} probes, "
              f"world={world}", flush=True)

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
        loader = DataLoader(FrameOnly(shard, args.image_dir, processor),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True)

        for mode in modes:
            records: list[dict] = []
            for obj in objects:
                records.extend(probe_one(model, tokenizer, loader, obj,
                                         mode == "with", device, dtype,
                                         args.max_new_tokens))
            if distributed:
                gathered: list[list[dict]] = [None] * world  # type: ignore[list-item]
                dist.all_gather_object(gathered, records)
                merged = [r for part in gathered for r in part]
            else:
                merged = records

            key = f"{label}__{mode}"
            if rank == 0:
                merged.sort(key=lambda r: (r["object"], r["id"]))
                path_out = os.path.join(args.out_dir, f"probes_{key}.jsonl")
                with open(path_out, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                from additional.stats import cluster_proportion, clopper_pearson
                overall = cluster_proportion([r["halluc"] for r in merged],
                                             [r["video"] for r in merged])
                paper3 = [r for r in merged if r["object"] in ABSENT_PAPER]
                summary[key] = {
                    "encoder": report,
                    "image_token_mode": mode,
                    "n_probes": len(merged),
                    "n_objects": len(objects),
                    "overall": overall,
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
                    "probes_path": path_out,
                }
                ex = overall["exact_binomial"]
                cb = overall["cluster_bootstrap"]
                print(f"[probe] {key:24s} {overall['k']}/{overall['n']} = "
                      f"{overall['rate'] * 100:.4f}%  "
                      f"exact 95% [{ex['lo'] * 100:.4f}, {ex['hi'] * 100:.4f}]  "
                      f"video-boot [{cb['lo'] * 100:.4f}, {cb['hi'] * 100:.4f}]",
                      flush=True)

    if rank == 0:
        meta = {
            "data_path": args.data_path,
            "vlm_ckpt": args.vlm_ckpt,
            "llm": args.llm,
            "vision_model": args.vision_model,
            "objects": objects,
            "question": QUESTION,
            "match_rule": "answer.startswith('yes') or ' yes' in answer[:12]",
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

"""
Aggregate per-sample attention records into reportable tables.

``eval_attn_mp`` writes one record per frame.  This module turns those records
into the breakdowns the study reports, and it always pairs a mean with a
video-level interval so no table implies more precision than the data supports.

When two labels are given it also reports the paired difference, resampled over
videos.  Pairing matters here: the same frame is scored under both encoders, so
a paired interval on the difference is much tighter than comparing two
independent intervals, and it is the quantity the claim is about.

Usage::

    python -m additional.aggregate \\
        --run stock=results/full_smd_stock/per_sample_stock.jsonl \\
        --run gaa=results/full_smd_gaa/per_sample_gaa.jsonl \\
        --out_json results/full_smd_table.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

from additional.stats import cluster_bootstrap_mean, cluster_bootstrap_paired_diff


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _finite(recs: list[dict], key: str) -> list[dict]:
    return [r for r in recs
            if r.get(key) is not None and not math.isnan(float(r[key]))]


def group_keys(recs: list[dict]) -> list[tuple[str, str]]:
    """The breakdowns to report: everything, then per split, then per condition."""
    keys: list[tuple[str, str]] = [("all", "")]
    for sp in sorted({r.get("split", "") for r in recs} - {""}):
        keys.append(("split", sp))
    for cd in sorted({r.get("condition", "") for r in recs} - {""}):
        keys.append(("condition", cd))
    return keys


def select(recs: list[dict], kind: str, value: str) -> list[dict]:
    if kind == "all":
        return recs
    return [r for r in recs if r.get(kind) == value]


def build(runs: dict[str, list[dict]], metric: str) -> dict:
    labels = list(runs)
    table: dict[str, dict] = {}

    base = runs[labels[0]]
    for kind, value in group_keys(base):
        name = "all" if kind == "all" else f"{kind}:{value}"
        row: dict[str, object] = {}
        for label in labels:
            sub = _finite(select(runs[label], kind, value), metric)
            if not sub:
                continue
            est = cluster_bootstrap_mean([r[metric] for r in sub],
                                         [r["video"] for r in sub])
            row[label] = est.as_dict()
        # Paired difference for exactly two runs, on the frames both scored.
        if len(labels) == 2:
            a, b = labels
            by_id_a = {r["id"]: r for r in _finite(select(runs[a], kind, value), metric)}
            by_id_b = {r["id"]: r for r in _finite(select(runs[b], kind, value), metric)}
            common = sorted(set(by_id_a) & set(by_id_b))
            if common:
                diff = cluster_bootstrap_paired_diff(
                    [by_id_a[i][metric] for i in common],
                    [by_id_b[i][metric] for i in common],
                    [by_id_a[i]["video"] for i in common])
                row["paired_diff"] = {
                    "from": a, "to": b, "n_paired": len(common),
                    **diff.as_dict(),
                }
        table[name] = row
    return table


def render(table: dict, labels: list[str], metric: str) -> str:
    lines = [f"{metric} — 평균과 95% 신뢰구간(비디오 단위 부트스트랩)", ""]
    head = f"{'구분':<18}{'n':>8}{'비디오':>7}"
    for label in labels:
        head += f"{label:>26}"
    if len(labels) == 2:
        head += f"{'차이(' + labels[1] + '-' + labels[0] + ')':>26}"
    lines.append(head)
    lines.append("-" * len(head))

    for name, row in table.items():
        first = next((row[l] for l in labels if l in row), None)
        if first is None:
            continue
        line = f"{name:<18}{first['n_items']:>8,}{first['n_clusters']:>7}"
        for label in labels:
            if label in row:
                e = row[label]
                line += f"{e['point']:>10.4f} [{e['lo']:.4f},{e['hi']:.4f}]"
            else:
                line += f"{'—':>26}"
        if "paired_diff" in row:
            d = row["paired_diff"]
            line += f"{d['point']:>+10.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
        lines.append(line)
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser("aggregate per-sample attention records")
    p.add_argument("--run", action="append", required=True,
                   help="LABEL=path/to/per_sample_LABEL.jsonl; repeatable, "
                        "at most two for a paired difference")
    p.add_argument("--metric", default="attn_iou",
                   choices=["attn_iou", "attn_in_box"])
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_txt", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    runs: dict[str, list[dict]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run needs LABEL=PATH form, got {spec!r}")
        label, path = spec.split("=", 1)
        runs[label] = load(path)
        print(f"[agg] {label:12s} {len(runs[label]):,} records  <- {path}")

    table = build(runs, args.metric)
    text = render(table, list(runs), args.metric)
    print()
    print(text)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"metric": args.metric, "runs": list(runs), "table": table},
                      f, indent=2)
        print(f"\n[agg] -> {args.out_json}")
    if args.out_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)), exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[agg] -> {args.out_txt}")


if __name__ == "__main__":
    main()

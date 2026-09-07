"""
Tabulate the training records left by ``train_geo_ddp``.

Every run writes ``train_meta.json`` next to its checkpoint with the full
configuration, the per-epoch loss and the failure counts.  This collects them
into one table so a sweep can be read at a glance, and so the report can state
what was actually trained rather than what a run script intended to train.

A non-empty ``skipped`` field is printed rather than summarised away: a run that
dropped samples produced a checkpoint from less data than its name suggests.

Usage::

    python -m additional.train_summary --checkpoints_root checkpoints \\
        --out_json results/train_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def collect(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "train_meta.json"))):
        with open(path) as f:
            m = json.load(f)
        m["name"] = os.path.basename(os.path.dirname(path))
        m["final_loss"] = (m["history"][-1]["loss"] if m.get("history")
                           else float("nan"))
        rows.append(m)
    rows.sort(key=lambda m: (m.get("variant", ""), m.get("lam", 0),
                             m.get("n_samples", 0)))
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return "학습 기록이 없습니다."
    head = (f"{'이름':<22}{'변형':<13}{'lam':>5}{'n':>7}{'스텝':>7}"
            f"{'초':>9}{'최종loss':>10}{'유효배치':>9}  스킵")
    lines = [head, "-" * (len(head) + 6)]
    for m in rows:
        lines.append(
            f"{m['name']:<22}{m.get('variant', ''):<13}{m.get('lam', 0):>5}"
            f"{m.get('n_samples', 0):>7}{m.get('total_steps', 0):>7}"
            f"{m.get('wall_seconds', 0):>9.1f}{m['final_loss']:>10.4f}"
            f"{m.get('effective_batch', 0):>9}  {m.get('skipped') or '없음'}")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser("tabulate train_geo_ddp records")
    p.add_argument("--checkpoints_root", required=True)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = collect(args.checkpoints_root)
    print(render(rows))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),
                    exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[train-summary] -> {args.out_json}")


if __name__ == "__main__":
    main()

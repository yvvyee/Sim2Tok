"""
Assemble the additional-experiment results into a claim-by-claim report.

The study exists to strengthen or bound specific claims in the submitted paper,
so the report is organised by claim rather than by experiment.  Each section
states what the paper claims, what the paper measured it on, what this study
measured it on, and what came out — with the video-level interval next to every
mean, because frames from one video are not independent draws.

The generator reads only result files.  A claim whose inputs are missing is
written as pending rather than omitted, so a partial run produces an honest
partial report instead of a document that silently drops what did not finish.

No claim is scored as supported or refuted here.  The report states the
measured quantities and the conditions they were measured under; whether they
support a claim is a judgement for the reader.

Usage::

    python -m additional.report \\
        --results_root /mnt/ssd1/yvvyee/gaa_additional/results \\
        --out additional_exp/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from additional.stats import (clopper_pearson, fisher_exact_2x2,
                              cluster_bootstrap_mean,
                              cluster_bootstrap_paired_diff, video_of)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_per_sample(results_root: str, run_dir: str) -> dict[str, list[dict]]:
    """``{label: records}`` for every ``per_sample_<label>.jsonl`` in a run."""
    d = os.path.join(results_root, run_dir)
    if not os.path.isdir(d):
        return {}
    out = {}
    for name in sorted(os.listdir(d)):
        if name.startswith("per_sample_") and name.endswith(".jsonl"):
            label = name[len("per_sample_"):-len(".jsonl")]
            out[label] = load_jsonl(os.path.join(d, name))
    return out


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_est(est) -> str:
    return f"{est.point:.4f} [{est.lo:.4f}, {est.hi:.4f}]"


def fmt_rate(p) -> str:
    return f"{p.k}/{p.n} = {p.point * 100:.4f}% [{p.lo * 100:.4f}%, {p.hi * 100:.4f}%]"


def attn_table(runs: dict[str, list[dict]], groups: list[tuple[str, str]]) -> list[str]:
    """Markdown rows of Attn-IoU per run for each grouping."""
    if not runs:
        return ["아직 결과가 없습니다."]
    labels = list(runs)
    head = "| 구분 | n | 비디오 | " + " | ".join(labels) + " |"
    rule = "|---|---|---|" + "---|" * len(labels)
    lines = [head, rule]
    for kind, value in groups:
        name = "전체" if kind == "all" else f"{kind}={value}"
        cells, n_items, n_clusters = [], 0, 0
        for label in labels:
            recs = runs[label]
            if kind != "all":
                recs = [r for r in recs if r.get(kind) == value]
            recs = [r for r in recs if r.get("attn_iou") is not None]
            if not recs:
                cells.append("—")
                continue
            est = cluster_bootstrap_mean([r["attn_iou"] for r in recs],
                                         [r["video"] for r in recs])
            n_items, n_clusters = est.n_items, est.n_clusters
            cells.append(fmt_est(est))
        if n_items:
            lines.append(f"| {name} | {n_items:,} | {n_clusters} | "
                         + " | ".join(cells) + " |")
    return lines


def paired_row(a: list[dict], b: list[dict], label_a: str, label_b: str) -> str:
    """Paired difference on the frames both runs scored."""
    ia = {r["id"]: r for r in a if r.get("attn_iou") is not None}
    ib = {r["id"]: r for r in b if r.get("attn_iou") is not None}
    common = sorted(set(ia) & set(ib))
    if not common:
        return "짝지을 수 있는 프레임이 없습니다."
    est = cluster_bootstrap_paired_diff(
        [ia[i]["attn_iou"] for i in common],
        [ib[i]["attn_iou"] for i in common],
        [ia[i]["video"] for i in common])
    sign = "+" if est.point >= 0 else ""
    return (f"{label_b} − {label_a} = {sign}{est.point:.4f} "
            f"[{est.lo:+.4f}, {est.hi:+.4f}]  "
            f"(짝지은 프레임 {est.n_items:,}, 비디오 {est.n_clusters})")


# ── Sections ─────────────────────────────────────────────────────────────────

def section_alignment(root: str) -> list[str]:
    """C2: the alignment gain, measured on the full SMD."""
    L = ["## C2. inbox_ce가 시선 정렬을 올린다", "",
         "논문 근거: SMD 288 프레임에서 stock 0.145 → GAA 0.437.", ""]

    stock = find_per_sample(root, "full_smd_stock").get("stock", [])
    paper = find_per_sample(root, "full_smd_paper")
    mine = find_per_sample(root, "full_smd_gaa").get("gaa", [])

    runs: dict[str, list[dict]] = {}
    if stock:
        runs["stock"] = stock
    for k, v in paper.items():
        runs[k] = v
    if mine:
        runs["gaa_10k"] = mine

    if not runs:
        return L + ["결과 대기 중입니다.", ""]

    conds = sorted({r.get("condition", "") for rs in runs.values() for r in rs} - {""})
    splits = sorted({r.get("split", "") for rs in runs.values() for r in rs} - {""})
    groups = ([("all", "")] + [("split", s) for s in splits]
              + [("condition", c) for c in conds])

    L += ["측정: SMD 라벨 프레임 전량. 평균과 95% 신뢰구간(비디오 단위 부트스트랩).", ""]
    L += attn_table(runs, groups)
    L += ["", "짝지은 차이(같은 프레임을 두 인코더로 채점):", ""]
    for other in runs:
        if other == "stock" or not stock:
            continue
        L.append(f"- {paired_row(stock, runs[other], 'stock', other)}")
    L.append("")
    return L


def section_controls(root: str) -> list[str]:
    """C4: the gain comes from the objective, not from having the boxes."""
    L = ["## C4. 이득은 상자 가용성이 아니라 목적함수에서 온다", "",
         "논문 근거: 같은 상자를 다른 방식으로 쓴 통제군이 기본선(0.145)보다 낮음. "
         "SFT-only 0.079, 상자 크롭 0.091. inbox_ce만 0.437.", ""]

    runs = find_per_sample(root, "full_smd_controls")
    stock = find_per_sample(root, "full_smd_stock").get("stock", [])
    paper = find_per_sample(root, "full_smd_paper")
    if stock:
        runs = {"stock": stock, **runs}
    if "paper_gaa" in paper:
        runs["paper_gaa"] = paper["paper_gaa"]
    if not runs:
        return L + ["결과 대기 중입니다.", ""]

    L += ["측정: 논문 체크포인트를 SMD 전량에 평가.", ""]
    L += attn_table(runs, [("all", "")])
    L.append("")

    sweep = find_per_sample(root, "sweep_variants")
    if sweep:
        L += ["1만개 규모에서 손실 변형을 다시 학습한 결과(비디오당 60프레임 부표본):", ""]
        L += attn_table(sweep, [("all", "")])
        L.append("")
    return L


def section_scaling(root: str) -> list[str]:
    """New evidence: does the gain hold as the training corpus grows?"""
    L = ["## 추가 근거. 학습 규모에 따른 정렬", "",
         "논문은 264 샘플에서만 측정했다. 같은 목적함수로 코퍼스를 늘렸을 때의 변화다.", ""]
    runs = find_per_sample(root, "sweep_scaling")
    if not runs:
        return L + ["결과 대기 중입니다.", ""]
    order = ["stock", "n264", "n1000", "n3000", "n10000"]
    runs = {k: runs[k] for k in order if k in runs} | \
           {k: v for k, v in runs.items() if k not in order}
    L += ["측정: 비디오당 60프레임 부표본(3,540 프레임, 59 비디오).", ""]
    L += attn_table(runs, [("all", "")])
    L.append("")
    return L


def section_halluc(root: str) -> list[str]:
    """C1/C2: the hallucination rate, and the image-token question."""
    L = ["## C1. 낮은 정렬이 환각의 근접 원인이다", "",
         "논문 근거: baseline 5/864, GAA 0/864. 그 표본에서 Fisher 정확검정 "
         "p = 0.062(프레임 독립 가정), 비디오 단위 3/36 대 0/36에서 p = 0.239.", ""]

    for run_dir, title in (("halluc_paper288", "논문 288 프레임 재현"),
                           ("halluc_sweep20", "확장 프로브 (3,540 프레임 × 20 범주)")):
        s = load_json(os.path.join(root, run_dir, "summary.json"))
        if not s:
            L += [f"### {title}", "", "결과 대기 중입니다.", ""]
            continue
        L += [f"### {title}", ""]
        L += ["| 조건 | 프로브 | 환각 | 비율 | 정확 이항 95% | 비디오 부트스트랩 95% |",
              "|---|---|---|---|---|---|"]
        for key, r in s.get("results", {}).items():
            o = r.get("overall") or {}
            ex, cb = o.get("exact_binomial", {}), o.get("cluster_bootstrap", {})
            L.append(f"| {key} | {o.get('n', 0):,} | {o.get('k', 0)} | "
                     f"{o.get('rate', 0) * 100:.4f}% | "
                     f"[{ex.get('lo', 0) * 100:.4f}%, {ex.get('hi', 0) * 100:.4f}%] | "
                     f"[{cb.get('lo', 0) * 100:.4f}%, {cb.get('hi', 0) * 100:.4f}%] |")
        L.append("")

        # Fisher between the stock and gaa conditions within each mode.
        res = s.get("results", {})
        for mode in ("with", "without"):
            a = res.get(f"stock__{mode}")
            b = res.get(f"gaa__{mode}")
            if not (a and b):
                continue
            ka, na = a["overall"]["k"], a["overall"]["n"]
            kb, nb = b["overall"]["k"], b["overall"]["n"]
            p = fisher_exact_2x2(ka, na - ka, kb, nb - kb)
            L.append(f"- `{mode}` 프로토콜: stock {ka}/{na} 대 GAA {kb}/{nb}, "
                     f"Fisher 정확검정 p = {p:.4g}")
        L.append("")

        # The image-token check: identical answers means no visual input.
        probes = {}
        for key in s.get("results", {}):
            probes[key] = load_jsonl(os.path.join(root, run_dir,
                                                  f"probes_{key}.jsonl"))
        for mode in ("with", "without"):
            a, b = probes.get(f"stock__{mode}"), probes.get(f"gaa__{mode}")
            if not (a and b):
                continue
            ia = {(r["id"], r["object"]): r["answer"] for r in a}
            ib = {(r["id"], r["object"]): r["answer"] for r in b}
            common = set(ia) & set(ib)
            same = sum(1 for k in common if ia[k] == ib[k])
            L.append(f"- `{mode}` 프로토콜에서 인코더 교체 후 답이 동일한 프로브: "
                     f"{same:,}/{len(common):,} ({same / max(len(common), 1) * 100:.2f}%)")
        L.append("")
    return L


def section_colregs(root: str) -> list[str]:
    """The paper's own stated limitation, measured on exact labels."""
    L = ["## 한계 재검증. COLREGs 조우 상황 분류", "",
         "논문은 이 항목에서 측정 가능한 개선이 없었다고 적었다. 그 측정은 100 에피소드의 "
         "영문 서술 키워드 매칭으로 정답을 복원한 것이다. 여기서는 시뮬레이터가 생성한 "
         "시나리오 라벨을 그대로 쓰고 표본이 1만개다.", ""]
    s = load_json(os.path.join(root, "colregs_10k", "summary.json"))
    if not s:
        return L + ["결과 대기 중입니다.", ""]
    L += ["| 조건 | n | 정확도 | 95% 구간 | safe/unknown 예측 |",
          "|---|---|---|---|---|"]
    for key, r in s.get("results", {}).items():
        a = r.get("accuracy") or {}
        L.append(f"| {key} | {a.get('n', 0):,} | {a.get('point', 0) * 100:.2f}% | "
                 f"[{a.get('lo', 0) * 100:.2f}%, {a.get('hi', 0) * 100:.2f}%] | "
                 f"{r.get('n_safe_or_unknown', 0):,} |")
    L.append("")
    for key, r in s.get("results", {}).items():
        per = r.get("per_scenario") or {}
        if not per:
            continue
        L.append(f"`{key}` 시나리오별 정확도: " + ", ".join(
            f"{sc} {v['point'] * 100:.1f}%" for sc, v in sorted(per.items())))
    L.append("")
    return L


def section_cross_arch(root: str) -> list[str]:
    L = ["## C3. 인코더 계열 전반 일반화", "",
         "논문 근거: 10모델 평균 0.079 → 0.438, 264 샘플 학습. "
         "요약 토큰이 없는 계열은 Saliency-IoU를 쓰므로 계열 간 절대값 비교는 "
         "성립하지 않고, 계열 내 전후 비교만 유효하다.", ""]
    rows = load_json(os.path.join(root, "cross_arch_10k.json"))
    if not rows:
        return L + ["결과 대기 중입니다.", ""]
    L += ["| 모델 | 인코더 | 계열 | 학습 전 | 학습 후 | 실패율 | 스텝 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        b, a = r.get("before"), r.get("after")
        before = f"{b:.4f}" if b is not None else "—"
        after = f"{a:.4f}" if a is not None else "—"
        L.append(f"| {r['model']} | {r['encoder']} | {r['family']} | "
                 f"{before} | {after} | {r['fail_fraction'] * 100:.2f}% | "
                 f"{r['steps']:,} |")
    L.append("")

    finite = [(r["before"], r["after"]) for r in rows
              if r.get("before") is not None and r.get("after") is not None]
    if finite:
        mb = sum(x for x, _ in finite) / len(finite)
        ma = sum(y for _, y in finite) / len(finite)
        L += [f"{len(finite)}개 모델 평균: {mb:.4f} → {ma:.4f}. "
              f"계열이 섞인 평균이므로 방향만 읽고 절대값 비교로 쓰지 않는다.", ""]
    return L


def section_detector(root: str) -> list[str]:
    L = ["## C5. 탐지기를 대체하지 않는다", "",
         "논문 근거: YOLOv8s mAP@50 0.433 대 GAA 주의 프록시 0.006. "
         "이 격차를 숨기지 않는 것이 주장의 범위를 정한다.", ""]
    y = load_json(os.path.join(root, "yolo", "yolov8s_10k", "results.json"))
    proxy = os.path.join(root, "detect_proxy.txt")
    if not y and not os.path.exists(proxy):
        return L + ["결과 대기 중입니다.", ""]
    if y:
        L += [f"YOLOv8s(1만개 학습) 결과: `{json.dumps(y)[:200]}`", ""]
    if os.path.exists(proxy):
        L += ["주의 프록시 탐지 평가:", "", "```"]
        with open(proxy) as f:
            L += [ln.rstrip() for ln in f if ln.strip()][-12:]
        L += ["```", ""]
    return L


def parse_args():
    p = argparse.ArgumentParser("assemble the additional-experiment report")
    p.add_argument("--results_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="추가 실험 결과: 논문 주장별 대조")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.results_root

    L = [f"# {args.title}", "",
         "논문 작성본은 수정하지 않았다. 여기의 수치는 논문 표를 대체하지 않고 병기한다.",
         "",
         "모든 평균에는 비디오 단위 부트스트랩 95% 구간을 붙였다. 한 비디오의 인접 "
         "프레임은 독립 표본이 아니므로, 프레임 수를 늘리는 것만으로는 구간이 그 비율만큼 "
         "좁아지지 않는다.", "",
         f"결과 원본: `{root}`", "", "---", ""]

    for fn in (section_alignment, section_controls, section_scaling,
               section_halluc, section_colregs, section_cross_arch,
               section_detector):
        L += fn(root)
        L += ["---", ""]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] -> {args.out}  ({len(L)} 줄)")


if __name__ == "__main__":
    main()

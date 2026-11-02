"""Attention overlap of the revision encoders: per-seed paired differences against CLIP and between GAA variants, and absolute means over seeds (main-paper Table 4).

Usage: python -m additional.tables.attagg $GAA_ROOT/results/rev_attn
"""
import json, os, sys, math
from additional.stats import cluster_bootstrap_paired_diff
R = sys.argv[1]
DS = ["smd", "mcad_rgb", "mcad_ir", "lars", "seadronessee"]
TAG = {"gaa": "rev_gaa", "gaam": "rev_gaam", "gaashuf": "rev_gaashuf",
       "gaa23_l23": "rev_gaa23", "gaa23_l24": "rev_gaa23"}
def load(t, s, ds, lab):
    p = f"{R}/{t}_s{s}_{ds}/per_sample_{lab}.jsonl"
    if not os.path.exists(f"{R}/{t}_s{s}_{ds}/summary.json"): return None
    return {r["id"]: r for r in map(json.loads, open(p))}
def get(c, s, ds):
    if c == "stock": return load("gaa", s, ds, "stock")
    return load(c, s, ds, TAG[c])
PAIRS = [("stock", c) for c in TAG] + [("gaa", "gaam"), ("gaa", "gaashuf"), ("gaa23_l23", "gaa23_l24")]
out = {}
for a, b in PAIRS:
    for ds in DS:
        row = []
        for s in (0, 1, 2):
            A, B = get(a, s, ds), get(b, s, ds)
            if A is None or B is None: row.append(None); continue
            ids = sorted(set(A) & set(B))
            e = cluster_bootstrap_paired_diff([A[i]["attn_iou"] for i in ids], [B[i]["attn_iou"] for i in ids],
                                              [A[i]["video"] for i in ids])
            ma = sum(A[i]["attn_iou"] for i in ids if A[i]["attn_iou"] is not None and not math.isnan(A[i]["attn_iou"])) / max(1, len(ids))
            row.append(e.as_dict())
        out[f"{b}-{a}|{ds}"] = row
        pts = [r["point"] for r in row if r]
        sg = "".join("-" if r is None else ("+" if r["lo"] > 0 else ("-" if r["hi"] < 0 else "0")) for r in row)
        print(f"{b+' - '+a:24s} {ds:13s} mean {sum(pts)/len(pts):+.3f}  ({sg})  n_seeds {len(pts)}" if pts else f"{b} - {a} {ds} none")
# absolute means
print("\nabsolute Attn-IoU (mean over seeds, item mean)")
for c in ["stock"] + list(TAG):
    vals = []
    for ds in DS:
        v = []
        for s in (0, 1, 2):
            p = f"{R}/{'gaa' if c=='stock' else c}_s{s}_{ds}/summary.json"
            if os.path.exists(p):
                v.append(json.load(open(p))["results"]["stock" if c == "stock" else TAG[c]]["attn_iou"]["point"])
        vals.append(f"{sum(v)/len(v):.3f}(n{len(v)})" if v else "  -  ")
    print(f"{c:10s} " + "  ".join(vals))
json.dump(out, open(f"{R}/../rev_attn_cmp.json", "w"), indent=1)

"""Sim-to-real gap of the frozen-encoder probe in three measures, mean +- std over seeds."""
import json, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
ROOT = os.environ.get("GAA_ROOT", "/mnt/ssd1/yvvyee/gaa_additional")
# Revision protocol: every encoder trained on the simulator training split only.
R = f"{ROOT}/results/rev_cmp"
per = []
for s in (0, 1, 2):
    p = json.load(open(f"{R}/A_ref_stock_s{s}.json"))["per_encoder"]
    p["rev_gaam"] = json.load(open(f"{R}/A_ref_gaam_s{s}.json"))["per_encoder"]["rev_gaam"]
    per.append(p)
enc = [("rev_stock", "CLIP", "#1baf7a"), ("rev_gaa", "GAA", "#f2a481"), ("rev_gaam", "GAA (matched)", "#eb6834"),
       ("rev_tok", "Token occupancy (ours)", "#2a78d6")]
panels = [("token_ap", "Token-AP gap (sim $-$ real)", ["smd", "mcad_rgb", "mcad_ir", "lars", "seadronessee"], 1),
          ("presence_auroc", "AUROC gap (sim $-$ real)", ["smd", "mcad_rgb", "mcad_ir", "lars", "seadronessee"], 1),
          ("count_mae", "Count-MAE gap (real $-$ sim)", ["smd", "mcad_rgb", "mcad_ir", "lars", "seadronessee"], -1)]
ZERO_FRAC = 0.1
short = {"smd": "SMD", "mcad_rgb": "MCAD\nRGB", "mcad_ir": "MCAD\nIR", "lars": "LaRS", "seadronessee": "SDS"}
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [5, 5, 5]})
for ax, (m, title, sets, sgn) in zip(axes, panels):
    x = np.arange(len(sets)); w = 0.2
    for k, (e, lab, c) in enumerate(enc):
        g = [[sgn * (p[e]["aivn_val"][m] - p[e][d][m]) for p in per] for d in sets]
        ax.bar(x + (k - 1.5) * w, [st.mean(v) for v in g], w - 0.03, yerr=[st.stdev(v) for v in g], color=c,
               label=lab, capsize=1.0, error_kw={"elinewidth": 0.5, "capthick": 0.5}, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([short[d] for d in sets], fontsize=6)
    ax.set_title(title + r" $\downarrow$", fontsize=7)
    ax.axhline(0, color="#888", lw=0.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.4); ax.set_axisbelow(True)
# The zero line sits at the same fraction of the axis height in every panel: at
# least ZERO_FRAC, or more if a panel's negative bars need more room.
frac = max([ZERO_FRAC] + [-lo / (hi - lo) for lo, hi in (ax.get_ylim() for ax in axes)])
for ax in axes:
    hi = ax.get_ylim()[1]
    ax.set_ylim(-frac * hi / (1 - frac), hi)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", ncol=4)
plt.tight_layout(pad=0.2, w_pad=0.8, rect=(0, 0, 1, 0.9))
plt.savefig(f"{ROOT}/results/fig_gap_rev.pdf")
print("ok")

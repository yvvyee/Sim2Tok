"""Texture and shape cue test: token AP on clean, texture-suppressed and block-permuted simulator validation images, restricted to images with permuted boxes (supplement).

Usage: python -m additional.tables.cue_sub rev_stock,rev_gaam,rev_tok
"""
import json,sys,numpy as np
from additional.probe_compare import ap_from_hist
import os
ROOT = os.environ.get("GAA_ROOT", "/mnt/ssd1/yvvyee/gaa_additional")
R=f"{ROOT}"
cue={r["id"]:r for r in json.load(open(f"{R}/cue_sets/aivn_val_shape.json"))}
sub={i for i,r in cue.items() if r["n_shuffled"]>0}
print("images with shuffled boxes:",len(sub))
for enc in sys.argv[1].split(","):
    out={}
    for st in ["aivn_val","aivn_val_tex","aivn_val_shape"]:
        v=[]
        for s in (0,1,2):
            rows=[json.loads(l) for l in open(f"{R}/results/rev_probe/{enc}_s{s}/{st}.jsonl")]
            rows=[r for r in rows if r["id"] in sub]
            hp=np.sum([r["hist_pos"] for r in rows],0); hn=np.sum([r["hist_neg"] for r in rows],0)
            v.append(ap_from_hist(hp,hn))
        out[st]=v
    base=np.array(out["aivn_val"])
    print(f"{enc:10s} clean {base.mean():.3f}±{base.std(ddof=1):.3f}  tex drop {(base-np.array(out['aivn_val_tex'])).mean():+.3f}±{(base-np.array(out['aivn_val_tex'])).std(ddof=1):.3f}  shape drop {(base-np.array(out['aivn_val_shape'])).mean():+.3f}±{(base-np.array(out['aivn_val_shape'])).std(ddof=1):.3f}")


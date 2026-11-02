"""Detection probe: per-seed paired differences from det_compare.py outputs, printed like pagg.py (supplement detection table).

Usage: python -m additional.tables.dagg <dir> <prefix> ref:cmp [ref:cmp ...]
"""
import json, sys, os
os.chdir(sys.argv[1]); pre=sys.argv[2]
DS=["aivn_val","smd","mcad_rgb","mcad_ir","lars","seadronessee"]
for pair in sys.argv[3:]:
    ref,cmp=pair.split(":")
    J=[json.load(open(f"{pre}_ref_{ref}_s{s}.json"))["paired"][cmp] for s in range(3)]
    print(f"### {cmp} - {ref}")
    for m in [k for k,x in J[0]["aivn_val"].items() if isinstance(x,dict)]:
        row=[]
        for ds in DS:
            v=[j.get(ds,{}).get(m) for j in J]
            if any(x is None for x in v): continue
            a=sum(x["a"] for x in v)/3; b=sum(x["b"] for x in v)/3
            sg="".join("+" if x["ci95"][0]>0 else "-" if x["ci95"][1]<0 else "0" for x in v)
            row.append(f"{ds} {a:.3f}/{b:.3f} {b-a:+.3f}({sg})")
        print(f"  {m}: "+" | ".join(row))

"""VLM yes/no study: per-seed paired differences M1-M0, M4-M0, M4-M1 from yesno_eval.py comparisons (supplement VLM table, paired rows).

Usage: python -m additional.tables.vlmagg <dir with M*_vs_M*_s{0,1,2}.json>
"""
import json, sys, os
os.chdir(sys.argv[1])
DS=["aivn_val","smd","mcad_rgb","mcad_ir","lars","seadronessee","buoy_distance"]
MET=[("vessel","fp_rate"),("vessel","recall"),("vessel","balanced_accuracy"),("count","balanced_accuracy"),("side","accuracy"),("absent","yes_rate"),("present","yes_rate")]
for pair in ["M1_vs_M0","M4_vs_M0","M4_vs_M1"]:
    J=[json.load(open(f"{pair}_s{s}.json"))["datasets"] for s in range(3)]
    print("####",pair)
    for fam,m in MET:
        row=[]
        for ds in DS:
            v=[j.get(ds,{}).get(fam,{}).get(m) for j in J]
            if any(x is None for x in v): continue
            a=sum(x["a"] for x in v)/3; b=sum(x["b"] for x in v)/3
            sg="".join("+" if x["ci95"][0]>0 else "-" if x["ci95"][1]<0 else "0" for x in v)
            row.append(f"{ds}:{a:.3f}/{b:.3f} {b-a:+.3f}({sg})")
        print(f"  {fam}.{m}: "+" | ".join(row))

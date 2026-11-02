"""VLM yes/no study: mean +- std over seeds for M0, M1 and M4 (supplement VLM table, absolute rows).

Usage: python -m additional.tables.vlmtab <dir with M*_vs_M*_s{0,1,2}.json>
"""
import json, sys, os, statistics as st
os.chdir(sys.argv[1])
DS=["aivn_val","smd","mcad_rgb","mcad_ir","lars","seadronessee","buoy_distance"]
MET=[("count","balanced_accuracy"),("vessel","fp_rate"),("present","yes_rate"),("absent","yes_rate"),("side","accuracy")]
src={"M0":("M1_vs_M0","a"),"M1":("M1_vs_M0","b"),"M4":("M4_vs_M0","b")}
for fam,m in MET:
    print("##",fam,m, "  ".join(DS))
    for mod,(p,k) in src.items():
        J=[json.load(open(f"{p}_s{s}.json"))["datasets"] for s in range(3)]
        cells=[]
        for ds in DS:
            v=[j.get(ds,{}).get(fam,{}).get(m) for j in J]
            if any(x is None for x in v): cells.append("   --      "); continue
            xs=[x[k] for x in v]
            cells.append(f"{st.mean(xs):.3f}±{st.stdev(xs):.3f}")
        print(f"  {mod}", " ".join(cells))

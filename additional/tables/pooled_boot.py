"""Paired difference B - A pooled over seeds 0-2: point = mean of per-seed
differences, 95% interval from 1000 bootstrap draws over resampling units,
the same draw applied to every seed. CPU only, reads saved probe outputs."""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np
from additional.probe_compare import load, Units, by_unit, best_thr
ROOT = os.environ.get("GAA_ROOT", "/mnt/ssd1/yvvyee/gaa_additional")
R = f'{ROOT}/results/rev_probe'
SETS = ['aivn_val', 'smd', 'mcad_rgb', 'mcad_ir', 'lars', 'seadronessee']
# "main": token occupancy against GAA and the controls (main-paper Tables 3 and 5).
# "stock": every encoder against the untouched CLIP encoder.
CMPS = {'main': [('rev_gaam', 'rev_tok'), ('rev_gaa', 'rev_tok'), ('rev_gaa23', 'rev_tok'),
                 ('rev_gaadr', 'rev_tokdr'), ('rev_img', 'rev_tok'), ('sig_gaam', 'sig_tok')],
         'stock': [('rev_stock', 'rev_gaa'), ('rev_stock', 'rev_gaam'), ('rev_stock', 'rev_tok')]}
THR = {}
def thr(run):
    if run not in THR:
        THR[run] = best_thr(list(load(f'{R}/{run}/aivn_val.jsonl').values()))
    return THR[run]
def job(args):
    a, b, s = args
    if not os.path.exists(f'{R}/{a}_s0/{s}.jsonl'):
        return (a, b, s, None)
    runs = [(f'{a}_s{k}', f'{b}_s{k}') for k in range(3)]
    data = {r: load(f'{R}/{r}/{s}.jsonl') for pr in runs for r in pr}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    U = [(Units(by_unit([data[ra][i] for i in ids]), thr(ra)),
          Units(by_unit([data[rb][i] for i in ids]), thr(rb))) for ra, rb in runs]
    keys = U[0][0].keys
    def diffs(draw):
        out = {}
        for ua, ub in U:
            xa, xb = ua.metrics(draw), ub.metrics(draw)
            for k in xa:
                if k in xb:
                    out.setdefault(k, []).append(xb[k] - xa[k])
        return {k: (float(np.mean(v)), v) for k, v in out.items() if len(v) == 3}
    pt = diffs(keys)
    rng = random.Random(0)
    boots = {k: [] for k in pt}
    for _ in range(1000):
        d = diffs(rng.choices(keys, k=len(keys)))
        for k in boots:
            if k in d:
                boots[k].append(d[k][0])
    res = {'n_images': len(ids), 'n_units': len(keys)}
    for k in pt:
        bs = sorted(boots[k])
        res[k] = {'diff': pt[k][0], 'per_seed': pt[k][1], 'std': float(np.std(pt[k][1])),
                  'ci95': [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]]}
    return (a, b, s, res)
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'main'
    jobs = [(a, b, s) for a, b in CMPS[which] for s in SETS]
    with Pool(len(jobs)) as p:
        out = p.map(job, jobs)
    res = {}
    for a, b, s, r in out:
        res.setdefault(f'{b}-{a}', {})[s] = r
    name = 'pooled_boot.json' if which == 'main' else 'pooled_boot_stock.json'
    json.dump(res, open(f'{ROOT}/results/{name}', 'w'), indent=1)
    print('done')

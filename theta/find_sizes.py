"""Machine-size rule for Theta (pre-registered). Usage: python3 -I find_sizes.py CENTILE_REPO [YEAR]"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json, importlib
sys.path.insert(0, _ROOT + "/online")
import numpy as np
os.environ["NODES"] = "4392"; sys.path.insert(0, os.path.join(sys.argv[1], "pretrain"))
rb = importlib.import_module("run_backfill_replay")
from theta_common import load_theta
YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else None
d = load_theta(YEAR); rows = d["win_rows"]
sub = d["ts"][rows].astype(float); nn = d["nn"][rows]; dur = d["dur"][rows]; lim = np.maximum(d["lim"][rows], dur)
def util(N):
    os.environ["NODES"] = str(N)
    fit = nn <= N
    st, _, nbf = rb.easy_sim(sub[fit], nn[fit], dur[fit], lim[fit], np.maximum(lim[fit], 60.0), N)
    ok = st >= 0; run = np.minimum(dur[fit][ok], lim[fit][ok]); mk = (st[ok] + run).max() - sub[fit].min()
    return float((nn[fit][ok] * run).sum() / (N * mk) * 100), int((~fit).sum()), float(np.mean(st[ok] - sub[fit][ok]))
out = {}
N = 4392
while True:
    u, dropped, w = util(N); out[N] = dict(util_pct=u, dropped_too_wide=dropped, mean_wait_s=w); print(N, out[N], flush=True)
    if u <= 70 or N >= 9000: break
    N += 100
cands = sorted(out); n0 = min(cands, key=lambda k: abs(out[k]["util_pct"] - 70))
sizes = [int(round(n0 * 0.82 / 100) * 100), int(n0), int(round(n0 * 1.25 / 100) * 100)]
res = {"grid": out, "N0": int(n0), "sizes": sizes}
json.dump(res, open(f"{_OUT}/theta{YEAR or ''}/sizes.json", "w"), indent=1); print(json.dumps({"N0": n0, "sizes": sizes}))

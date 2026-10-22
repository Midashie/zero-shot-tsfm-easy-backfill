"""Pre-registered Theta year analysis (usage: analyze_theta.py YEAR, 2018 to 2020; no argument = the 2017 run): Bolt-96 and Bolt-192 vs user limits at 3 sizes, 6 paired by-user bootstrap tests, Holm over 6.
Sensitivity (outside the family, Holm within each 6): submit date from 2017-09-01, and without the account that submitted 17% of all jobs."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, json, os
sys.path.insert(0, _ROOT + "/online")
import numpy as np, pandas as pd
from theta_common import load_theta
O = _OUT + "/online"; B = 10000
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else None
yy = f"t{str(YEAR)[2:]}" if YEAR else ""
SIZES = json.load(open(f"{_OUT}/theta{YEAR or ''}/sizes.json"))["sizes"]
if YEAR:
    from theta_common import FILES
    TOP = pd.read_csv(FILES[YEAR], usecols=["USERNAME_GENID"]).USERNAME_GENID.value_counts().index[0]   # largest account of the calendar-year file
else:
    TOP = 53525247310067
d = load_theta(YEAR); rows = d["win_rows"]; sub_all = d["ts"][rows]
SEP1 = int(pd.Timestamp("2017-09-01T00:00:00Z").timestamp())


def load(name, n):
    z = np.load(f"{O}/online_{name}_{n}.npz", allow_pickle=True); return z["usr"], z["bs"], z["fit_index"]


def paired(u, b0, b1, rng, mask):
    ok = ~np.isnan(b0) & ~np.isnan(b1) & mask
    us, inv = np.unique(u[ok], return_inverse=True)
    s0 = np.bincount(inv, b0[ok]); s1 = np.bincount(inv, b1[ok]); c = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(us), (B, len(us)))
    dd = (s1[idx].sum(1) - s0[idx].sum(1)) / c[idx].sum(1)
    p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / B)
    return dict(m0=float(s0.sum() / c.sum()), m1=float(s1.sum() / c.sum()), ci=[float(np.quantile(dd, .025)), float(np.quantile(dd, .975))],
                p=float(p), n_users=int(len(us)), n_jobs=int(ok.sum()))


def holm(L):
    ps = np.array([r["p"] for r in L]); o = np.argsort(ps); m = len(ps); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); L[i]["p_holm"] = float(run)


fams = {"primary": [], "without_top_user": []} if YEAR else {"primary": [], "from_sep1": [], "without_top_user": []}
for n in SIZES:
    V = f"validate_{yy}" if YEAR else "validate"; BN = (lambda c: f"bolt_{yy}c{c}") if YEAR else (lambda c: f"bolt_th{c}")
    if not all(os.path.exists(f"{O}/online_{x}_{n}.npz") for x in (V, BN(96), BN(192))): continue
    u, ul, fi = load(V, n); sub = sub_all[fi]
    masks = {"primary": np.ones(len(u), bool), "without_top_user": u != TOP}
    if not YEAR: masks["from_sep1"] = sub >= SEP1
    for c in (96, 192):
        _, bb, _ = load(BN(c), n)
        for fam, m in masks.items():
            rng = np.random.default_rng(3000 + n + c)
            r = paired(u, ul, bb, rng, m); r.update(size=n, ctx=c); fams[fam].append(r)
for fam in fams: holm(fams[fam])
for fam, L in fams.items():
    print("==", fam)
    for r in L:
        print(f'  size {r["size"]} ctx {r["ctx"]:3d}: user limit {r["m0"]:8.2f} -> Bolt {r["m1"]:8.2f} ({(r["m1"] / r["m0"] - 1) * 100:+6.1f}%) CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f} holm {r["p_holm"]:.4f} users {r["n_users"]} jobs {r["n_jobs"]}')
P = fams["primary"]
if len(P) == 6:
    harm = any(r["p_holm"] < .05 and r["m1"] > r["m0"] for r in P)
    win = False
    for c in (96, 192):
        Q = [r for r in P if r["ctx"] == c]
        if sum(1 for r in Q if r["p_holm"] < .05 and r["m1"] < r["m0"]) >= 2 and not any(r["p_holm"] < .05 and r["m1"] > r["m0"] for r in Q): win = True
    verdict = "MIXED" if (win and harm) else ("WIN" if win else ("HARM" if harm else "NO EFFECT"))
    print("verdict:", verdict); fams["verdict"] = verdict
json.dump(fams, open(f"{O}/theta{YEAR or ''}_results.json", "w"), indent=1)

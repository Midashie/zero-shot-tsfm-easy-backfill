"""Pre-registered analysis of the SafeWall-style baseline on the four Theta windows. Per window: 3 paired by-user bootstrap tests (swb vs user limit, one per size), Holm over 3.
Outside the family: unbuffered XGBoost vs user limit, swb vs Bolt-96, swb vs user limit without the largest account of the year."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, json, os
sys.path.insert(0, _ROOT + "/online")
import numpy as np, pandas as pd
from theta_common import load_theta, FILES
O = _OUT + "/online"; B = 10000
def load(name, n):
    z = np.load(f"{O}/online_{name}_{n}.npz", allow_pickle=True); return z["usr"], z["bs"]
def paired(u, b0, b1, rng, mask):
    ok = ~np.isnan(b0) & ~np.isnan(b1) & mask
    us, inv = np.unique(u[ok], return_inverse=True)
    s0 = np.bincount(inv, b0[ok]); s1 = np.bincount(inv, b1[ok]); c = np.bincount(inv).astype(float)
    idx = rng.integers(0, len(us), (B, len(us)))
    dd = (s1[idx].sum(1) - s0[idx].sum(1)) / c[idx].sum(1)
    p = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / B)
    return dict(m0=float(s0.sum() / c.sum()), m1=float(s1.sum() / c.sum()), ci=[float(np.quantile(dd, .025)), float(np.quantile(dd, .975))], p=float(p), n_users=int(len(us)), n_jobs=int(ok.sum()))
def holm(L):
    ps = np.array([r["p"] for r in L]); o = np.argsort(ps); m = len(ps); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * ps[i])); L[i]["p_holm"] = float(run)
res = {}
for y in (2017, 2018, 2019, 2020):
    yy = f"t{str(y)[2:]}"; d = "theta" if y == 2017 else f"theta{y}"
    sizes = json.load(open(f"{_OUT}/{d}/sizes.json"))["sizes"]
    V = "validate" if y == 2017 else f"validate_{yy}"; BO = (lambda n: "bolt_th96") if y == 2017 else (lambda n: f"bolt_{yy}c96")
    top = 53525247310067 if y == 2017 else pd.read_csv(FILES[y], usecols=["USERNAME_GENID"]).USERNAME_GENID.value_counts().index[0]
    fam = {"primary": [], "xgb_unbuffered": [], "vs_bolt96": [], "without_top_user": []}
    for n in sizes:
        if not all(os.path.exists(f"{O}/online_{x}_{n}.npz") for x in (V, f"swb_{yy}", f"xgbraw_{yy}", BO(n))): continue
        u, ul = load(V, n); _, sw = load(f"swb_{yy}", n); _, xr = load(f"xgbraw_{yy}", n); _, bo = load(BO(n), n)
        allm = np.ones(len(u), bool); rng = lambda k: np.random.default_rng(5000 + n + k)
        for name, a, b, m, k in (("primary", ul, sw, allm, 1), ("xgb_unbuffered", ul, xr, allm, 2), ("vs_bolt96", bo, sw, allm, 3), ("without_top_user", ul, sw, u != top, 4)):
            r = paired(u, a, b, rng(k), m); r["size"] = n; fam[name].append(r)
    for k in fam: holm(fam[k])
    P = fam["primary"]
    if len(P) == 3:
        loss = any(r["p_holm"] < .05 and r["m1"] > r["m0"] for r in P); gain = sum(1 for r in P if r["p_holm"] < .05 and r["m1"] < r["m0"])
        fam["verdict"] = "HARM" if loss else ("WIN" if gain >= 2 else "NO EFFECT")
    res[y] = fam
    print("==", y, "verdict:", fam.get("verdict"))
    for k in ("primary", "xgb_unbuffered", "vs_bolt96", "without_top_user"):
        lab = {"primary": "swb vs user limit", "xgb_unbuffered": "xgb raw vs user limit", "vs_bolt96": "swb vs Bolt-96", "without_top_user": "swb vs user limit, no top acct"}[k]
        for r in fam[k]: print(f'  {lab:32s} size {r["size"]}: {r["m0"]:8.2f} -> {r["m1"]:8.2f} ({(r["m1"]/r["m0"]-1)*100:+6.1f}%) CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f} holm {r["p_holm"]:.4f}')
json.dump(res, open(f"{O}/swb_results.json", "w"), indent=1)

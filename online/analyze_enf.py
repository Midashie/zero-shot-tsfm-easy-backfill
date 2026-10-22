"""Pre-registered analysis of the kill-and-requeue replay. Per window: 3 arms (Bolt-96 median, Bolt-96 q0.8, SafeWall-style) x 3 sizes vs user limits,
paired by-user bootstrap (10,000), Holm over the 9 tests of the window. Verdict per arm. Outside the family: unbuffered XGBoost (Holm over 3), the 9 tests without the year's largest account, kill and waste shares."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, json, os
sys.path.insert(0, _ROOT + "/online")
import numpy as np, pandas as pd
from theta_common import FILES
O = _OUT + "/online"; B = 10000
def load(name, n):
    z = np.load(f"{O}/online_{name}_{n}.npz", allow_pickle=True); return z["usr"], z["bs"]
def info(name, n): return json.load(open(f"{O}/online_{name}_{n}.json"))
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
ARMS = [("bolt_median", "bolt_e{yy}c96"), ("bolt_q80", "bolt_e{yy}q80"), ("swb", "swb_e{yy}")]
res = {}
for y in (2017, 2018, 2019, 2020):
    yy = str(y)[2:]; d = "theta" if y == 2017 else f"theta{y}"
    sizes = json.load(open(f"{_OUT}/{d}/sizes.json"))["sizes"]
    top = 53525247310067 if y == 2017 else pd.read_csv(FILES[y], usecols=["USERNAME_GENID"]).USERNAME_GENID.value_counts().index[0]
    fam = {"primary": [], "without_top_user": [], "xgb_unbuffered": []}
    for n in sizes:
        names = [f"validate_e{yy}", f"xgbraw_e{yy}"] + [a[1].format(yy=yy) for a in ARMS]
        if not all(os.path.exists(f"{O}/online_{x}_{n}.npz") for x in names): continue
        u, ul = load(f"validate_e{yy}", n); allm = np.ones(len(u), bool)
        for k, (an, nm) in enumerate(ARMS):
            _, b = load(nm.format(yy=yy), n); ij = info(nm.format(yy=yy), n)
            for fk, m, off in (("primary", allm, 0), ("without_top_user", u != top, 10)):
                r = paired(u, ul, b, np.random.default_rng(7000 + n + k + off), m); r.update(size=n, arm=an, killed_pct=ij["killed_jobs_pct"], wasted_pct=ij["wasted_node_hours_pct_of_used"]); fam[fk].append(r)
        _, xr = load(f"xgbraw_e{yy}", n); ij = info(f"xgbraw_e{yy}", n)
        r = paired(u, ul, xr, np.random.default_rng(7900 + n), allm); r.update(size=n, arm="xgb_unbuffered", killed_pct=ij["killed_jobs_pct"], wasted_pct=ij["wasted_node_hours_pct_of_used"]); fam["xgb_unbuffered"].append(r)
    holm(fam["primary"]); holm(fam["without_top_user"]); holm(fam["xgb_unbuffered"])
    P = fam["primary"]; fam["verdict"] = {}
    if len(P) == 9:
        for an, _ in ARMS:
            Q = [r for r in P if r["arm"] == an]
            loss = any(r["p_holm"] < .05 and r["m1"] > r["m0"] for r in Q); gain = sum(1 for r in Q if r["p_holm"] < .05 and r["m1"] < r["m0"])
            fam["verdict"][an] = "HARM" if loss else ("WIN" if gain >= 2 else "NO EFFECT")
    res[y] = fam
    print("==", y, "verdicts:", fam["verdict"])
    for k in ("primary", "xgb_unbuffered", "without_top_user"):
        for r in sorted(fam[k], key=lambda r: (r["arm"], r["size"])):
            print(f'  {k:16s} {r["arm"]:14s} size {r["size"]}: {r["m0"]:8.2f} -> {r["m1"]:8.2f} ({(r["m1"]/r["m0"]-1)*100:+6.1f}%) CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] holm {r["p_holm"]:.4f} killed {r["killed_pct"]:.1f}% wasted {r["wasted_pct"]:.1f}%')
json.dump(res, open(f"{O}/enf_results.json", "w"), indent=1)

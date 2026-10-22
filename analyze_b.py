"""Task B analysis: decision metrics from the replays, by-user paired bootstrap, forecast diagnostics, rank agreement.
Usage: python3 -I analyze_b.py TASKB_DIR G3_DIR OUT_JSON [B]
Pre-set before any replay number existed:
  - primary quantile level per model: q=0.5 (the plan); q=0.6 to 0.8 are a sweep, reported but not tested
  - bootstrap: B=10,000 resamples of USERS (jobs inside a user are correlated); Holm inside each family
  - family 1: five models at q=0.5 vs the best classical arm at each node count (10 tests)
  - family 2: all pairs of the five models at q=0.5 at each node count (20 tests); this includes Chronos-2 with vs without covariates
  - best classical arm = lowest mean bounded slowdown among user-elpl, completion-causal Tsafrir last-2 and completion-causal history q90
  - CENTILE is not in the comparison (reference only, see the retrain result)
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json, itertools
import numpy as np
from scipy.stats import kendalltau

tdir, g3dir, out = sys.argv[1:4]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
rng = np.random.default_rng(777)
NODES = (20000, 23000)
CLASSICAL = ["user-elpl", "classical:Tsafrir last-2 (completion-causal)", "classical:history q90 (completion-causal)"]
MODELS = ["Chronos-Bolt-small", "Chronos-2", "Chronos-2 + covariates", "Moirai-2.0", "TTM-90 + conformal"]
QS = (0.5, 0.6, 0.7, 0.8)
key = lambda m, q: f"{m} q{q}"

R = {"design": {"B": B, "primary_q": 0.5, "resampling_unit": "user"}, "nodes": {}, "tests": {}, "forecast": {}, "rank": {}, "latency_cpu_2_threads": {}}


def counts(n, B_):
    for i in range(0, B_, 2000):
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=min(2000, B_ - i)).astype(np.float64)


def holm(ps):
    order = np.argsort(ps)
    m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


# ---------- per-node decision results ----------
USERS, USUM, UCNT, BOOT, POINT = {}, {}, {}, {}, {}
CB = None
for N in NODES:
    Zn = np.load(os.path.join(tdir, f"replay_bs_{N}.npz"), allow_pickle=True)
    summ = json.load(open(os.path.join(tdir, f"replay_summary_{N}.json")))
    usr = Zn["usr"]
    uniq, inv = np.unique(usr, return_inverse=True)
    names = [k for k in Zn.files if k not in ("usr", "fit_index")]
    S = np.zeros((len(names), len(uniq))); Cn = np.zeros((len(names), len(uniq)))
    for a, nm in enumerate(names):
        x = Zn[nm]; m = ~np.isnan(x)
        np.add.at(S[a], inv[m], x[m].astype(np.float64)); np.add.at(Cn[a], inv[m], 1.0)
    point = S.sum(1) / Cn.sum(1)
    bm = np.zeros((B, len(names)))
    i0 = 0
    for Cb in counts(len(uniq), B):
        bm[i0:i0 + len(Cb)] = (Cb @ S.T) / (Cb @ Cn.T); i0 += len(Cb)
    BOOT[N] = {nm: bm[:, a] for a, nm in enumerate(names)}
    POINT[N] = {nm: float(point[a]) for a, nm in enumerate(names)}
    USERS[N], USUM[N], UCNT[N] = uniq, S, Cn
    R["nodes"][str(N)] = {"users": int(len(uniq)), "dropped_too_wide": summ["dropped_too_wide"], "arms": {}}
    for a, nm in enumerate(names):
        s = summ["arms"][nm]
        R["nodes"][str(N)]["arms"][nm] = {"mean_bs": s["mean_bs"], "mean_bs_ci95": np.percentile(bm[:, a], [2.5, 97.5]).tolist(),
                                          "p95_bs": s["p95_bs"], "util_pct": s["util_pct"], "median_wait_s": s["median_wait_s"],
                                          "backfilled": s["backfilled"], "started_pct": s["started_pct"]}

# ---------- paired tests ----------
fam1, fam2 = [], []
for N in NODES:
    best = min(CLASSICAL, key=lambda c: POINT[N][c])
    R["tests"][str(N)] = {"best_classical": best, "vs_best_classical": {}, "pairwise": {}}
    for m in MODELS:
        d = BOOT[N][key(m, 0.5)] - BOOT[N][best]
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1))
        e = {"diff_mean_bs": float(POINT[N][key(m, 0.5)] - POINT[N][best]), "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_boot": float(p)}
        R["tests"][str(N)]["vs_best_classical"][m] = e; fam1.append(e)
    for a, b in itertools.combinations(MODELS, 2):
        d = BOOT[N][key(a, 0.5)] - BOOT[N][key(b, 0.5)]
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1))
        e = {"diff_mean_bs": float(POINT[N][key(a, 0.5)] - POINT[N][key(b, 0.5)]), "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_boot": float(p)}
        R["tests"][str(N)]["pairwise"][f"{a} minus {b}"] = e; fam2.append(e)
for fam, nm in ((fam1, "p_holm_family1"), (fam2, "p_holm_family2")):
    for e, a in zip(fam, holm([x["p_boot"] for x in fam])):
        e[nm] = float(a)

# ---------- forecast diagnostics on replayed jobs (log1p runtime) ----------
Z = np.load(os.path.join(tdir, "contexts_c96.npz"))
rows = Z["win_rows"]
y = np.log1p(Z["dur"][rows])
usr_all = np.load(os.path.join(tdir, "replay_bs_20000.npz"), allow_pickle=True)["usr"]       # user ids of fit jobs, used only for the unit of resampling below
import pandas as pd
man = pd.read_csv(_OUT + "/manifest/fdata_replay_jobs.csv")
user_of_job = man["usr"].values
uq, uinv = np.unique(user_of_job, return_inverse=True)
last = np.where(Z["win_len"] >= 1, np.log1p(Z["dur"][np.maximum(Z["win_ctx"][:, -1], 0)]), np.nan)
LEV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fc = {}                                       # arm -> dict(point (n,), crps (n,))   NaN where not forecast
mask_all = np.ones(len(rows), bool)
for name, fn in (("Chronos-Bolt-small", "bolt"), ("Chronos-2", "chronos2"), ("Chronos-2 + covariates", "chronos2cov"), ("Moirai-2.0", "moirai2")):
    F = np.load(os.path.join(tdir, f"F_{fn}.npz")); Q, m = F["win_Q"], F["win_forecast_mask"]
    pt = np.where(m, Q[:, 4], np.nan)
    pin = np.stack([np.maximum(t * (y - Q[:, j]), (t - 1) * (y - Q[:, j])) for j, t in enumerate(LEV)], 1)
    fc[name] = {"point": pt, "crps": np.where(m, 2 * pin.mean(1), np.nan)}
    mask_all &= m
F = np.load(os.path.join(tdir, "F_ttm90.npz"))
fc["TTM-90 + conformal"] = {"point": np.where(F["win_forecast_mask"], F["win_P"], np.nan), "crps": np.where(F["win_forecast_mask"], np.abs(F["win_P"] - y), np.nan)}
mask_all &= F["win_forecast_mask"]
for nm, f in (("classical:Tsafrir last-2 (completion-causal)", "causal_tsafrir.npz"), ("classical:history q90 (completion-causal)", "causal_q90.npz")):
    e = np.log1p(np.load(os.path.join(g3dir, f))["est"]); fc[nm] = {"point": e, "crps": np.abs(e - y)}
fc["user-elpl"] = {"point": np.log1p(Z["lim"][rows]), "crps": np.abs(np.log1p(Z["lim"][rows]) - y)}
ev = mask_all & ~np.isnan(last)               # jobs where every arm produced a forecast
naive = np.abs(last - y)[ev].mean()
R["forecast"] = {"jobs_evaluated": int(ev.sum()), "naive_last_runtime_mae_log": float(naive), "arms": {}}
mase_u, crps_u = {}, {}
for nm, f in fc.items():
    ae = np.abs(f["point"] - y)
    R["forecast"]["arms"][nm] = {"MASE_vs_last_runtime": float(ae[ev].mean() / naive), "CRPS9_or_MAE_log": float(f["crps"][ev].mean())}
    s1, s2 = np.zeros(len(uq)), np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s1, uinv[ev], ae[ev]); np.add.at(s2, uinv[ev], f["crps"][ev]); np.add.at(c, uinv[ev], 1.0)
    mase_u[nm], crps_u[nm] = s1, s2
cnt_u = c
Rk = list(fc.keys())
Ub = np.concatenate(list(counts(len(uq), B)), 0)
mb = {k: (Ub @ mase_u[k]) / (Ub @ cnt_u) / naive for k in Rk}
cb = {k: (Ub @ crps_u[k]) / (Ub @ cnt_u) for k in Rk}
# decision ranking on arms at q=0.5 (classical arms as they are)
names_dec = {nm: (key(nm, 0.5) if nm in MODELS else nm) for nm in Rk}
for N in NODES:
    pt = {nm: POINT[N][names_dec[nm]] for nm in Rk}
    # map resampled users: replay users and manifest users are the same set; align by name
    ur = USERS[N]
    pos = np.searchsorted(uq, ur)
    # bootstrap decision metric must use the SAME user resamples as the forecast metric: recompute from per-user sums on the common user list
    S, Cn = USUM[N], UCNT[N]
    names = [k for k in np.load(os.path.join(tdir, f"replay_bs_{N}.npz"), allow_pickle=True).files if k not in ("usr", "fit_index")]
    full_S = np.zeros((len(names), len(uq))); full_C = np.zeros((len(names), len(uq)))
    full_S[:, pos] = S; full_C[:, pos] = Cn
    db = {}
    for nm in Rk:
        a = names.index(names_dec[nm])
        db[nm] = (Ub @ full_S[a]) / np.maximum(Ub @ full_C[a], 1)
    tau_m = kendalltau([pt[k] for k in Rk], [R["forecast"]["arms"][k]["MASE_vs_last_runtime"] for k in Rk])[0]
    tau_c = kendalltau([pt[k] for k in Rk], [R["forecast"]["arms"][k]["CRPS9_or_MAE_log"] for k in Rk])[0]
    tm = [kendalltau([db[k][b] for k in Rk], [mb[k][b] for k in Rk])[0] for b in range(B)]
    tc = [kendalltau([db[k][b] for k in Rk], [cb[k][b] for k in Rk])[0] for b in range(B)]
    inv = []
    for a, b2 in itertools.combinations(Rk, 2):
        do = db[a] - db[b2]
        for fname, fb in (("MASE", mb), ("CRPS", cb)):
            df = fb[a] - fb[b2]
            co, cf = np.percentile(do, [2.5, 97.5]), np.percentile(df, [2.5, 97.5])
            sig = lambda c_: c_[0] > 0 or c_[1] < 0
            if sig(co) and sig(cf) and np.sign(do.mean()) != np.sign(df.mean()):
                inv.append({"pair": f"{a} vs {b2}", "forecast_metric": fname, "decision_diff_ci95": co.tolist(), "forecast_diff_ci95": cf.tolist()})
    R["rank"][str(N)] = {"arms": Rk, "decision_mean_bs": pt, "kendall_decision_vs_MASE": float(tau_m), "ci95_MASE": np.nanpercentile(tm, [2.5, 97.5]).tolist(),
                         "kendall_decision_vs_CRPS": float(tau_c), "ci95_CRPS": np.nanpercentile(tc, [2.5, 97.5]).tolist(), "inversions": inv}

for fn in ("bolt", "chronos2", "chronos2cov", "moirai2", "ttm90"):
    R["latency_cpu_2_threads"][fn] = json.load(open(os.path.join(tdir, f"lat_{fn}.json")))
json.dump(R, open(out, "w"), indent=1, default=float)
print("ANALYSIS_B_DONE")

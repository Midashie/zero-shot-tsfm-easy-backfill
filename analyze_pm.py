"""PM100 analysis: mean bounded slowdown at 512 nodes, paired by-user bootstrap, forecast diagnostics, rank agreement.
Usage: python3 -I analyze_pm.py PM_DIR OUT_JSON [B]
Design fixed before results (see the plan doc): q=0.5 primary; 10,000 resamples of users; family 1 = six models vs best classical arm (6 tests);
family 2 = all 15 pairs; Holm inside each family; best classical = lowest mean slowdown among user limits, Tsafrir last-2 and history q90 (both completion-causal)."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import kendalltau
from pm_common import load_pm

pdir, out = sys.argv[1:3]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
rng = np.random.default_rng(777); N = 512
CLASSICAL = ["user-elpl", "classical:Tsafrir last-2 (completion-causal)", "classical:history q90 (completion-causal)"]
MODELS = ["Chronos-Bolt-small", "Chronos-2", "Chronos-2 + covariates", "Moirai-2.0", "TimesFM-2.5", "TTM-90 + conformal"]
FN = {"Chronos-Bolt-small": "bolt", "Chronos-2": "chronos2", "Chronos-2 + covariates": "chronos2cov", "Moirai-2.0": "moirai2", "TimesFM-2.5": "timesfm"}
key = lambda m, q: f"{m} q{q}"
R = {"design": {"B": B, "N": N, "primary_q": 0.5, "unit": "user"}, "arms": {}, "tests": {}, "forecast": {}, "rank": {}, "latency_cpu_2_threads": {}}


def counts(n, B_):
    for i in range(0, B_, 2000):
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=min(2000, B_ - i)).astype(np.float64)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


Zn = np.load(os.path.join(pdir, f"replay_bs_{N}.npz"), allow_pickle=True)
summ = json.load(open(os.path.join(pdir, f"replay_summary_{N}.json")))
usr = Zn["usr"]; uniq, inv = np.unique(usr, return_inverse=True)
names = [k for k in Zn.files if k not in ("usr", "fit_index")]
S = np.zeros((len(names), len(uniq))); Cn = np.zeros_like(S)
for a, nm in enumerate(names):
    x = Zn[nm]; m = ~np.isnan(x)
    np.add.at(S[a], inv[m], x[m].astype(np.float64)); np.add.at(Cn[a], inv[m], 1.0)
point = S.sum(1) / Cn.sum(1)
bm = np.zeros((B, len(names))); i0 = 0
Cb_list = []
for Cb in counts(len(uniq), B):
    bm[i0:i0 + len(Cb)] = (Cb @ S.T) / (Cb @ Cn.T); i0 += len(Cb)
BOOT = {nm: bm[:, a] for a, nm in enumerate(names)}; POINT = {nm: float(point[a]) for a, nm in enumerate(names)}
for a, nm in enumerate(names):
    s = summ["arms"][nm]
    R["arms"][nm] = {"mean_bs": s["mean_bs"], "mean_bs_ci95": np.percentile(bm[:, a], [2.5, 97.5]).tolist(), "p95_bs": s["p95_bs"], "util_pct": s["util_pct"],
                     "median_wait_s": s["median_wait_s"], "backfilled": s["backfilled"], "started_pct": s["started_pct"]}
best = min(CLASSICAL, key=lambda c: POINT[c])
R["tests"] = {"best_classical": best, "vs_best_classical": {}, "pairwise": {}}
fam1, fam2 = [], []
for m in MODELS:
    d = BOOT[key(m, 0.5)] - BOOT[best]
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1))
    e = {"diff_mean_bs": POINT[key(m, 0.5)] - POINT[best], "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_boot": float(p)}
    R["tests"]["vs_best_classical"][m] = e; fam1.append(e)
for a, b in itertools.combinations(MODELS, 2):
    d = BOOT[key(a, 0.5)] - BOOT[key(b, 0.5)]
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1))
    e = {"diff_mean_bs": POINT[key(a, 0.5)] - POINT[key(b, 0.5)], "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_boot": float(p)}
    R["tests"]["pairwise"][f"{a} minus {b}"] = e; fam2.append(e)
for fam, nm in ((fam1, "p_holm_family1"), (fam2, "p_holm_family2")):
    for e, a in zip(fam, holm([x["p_boot"] for x in fam])): e[nm] = float(a)

# ---------- forecast diagnostics ----------
d = load_pm(); Z = np.load(os.path.join(pdir, "contexts_c96.npz")); rows = Z["win_rows"]
y = np.log1p(Z["dur"][rows]); lim = np.maximum(d["lim"][rows], Z["dur"][rows])
assert np.array_equal(usr, d["usr"][rows])
ctx = Z["win_ctx"]
last = np.where(Z["win_len"] >= 1, np.log1p(Z["dur"][np.maximum(ctx[:, -1], 0)]), np.nan)
LEV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fc = {}; mask_all = np.ones(len(rows), bool)
for nm, fn in FN.items():
    F = np.load(os.path.join(pdir, f"F_{fn}.npz")); Q, m = F["win_Q"], F["win_forecast_mask"]
    pin = np.stack([np.maximum(t * (y - Q[:, j]), (t - 1) * (y - Q[:, j])) for j, t in enumerate(LEV)], 1)
    fc[nm] = {"point": np.where(m, Q[:, 4], np.nan), "crps": np.where(m, 2 * pin.mean(1), np.nan)}; mask_all &= m
F = np.load(os.path.join(pdir, "F_ttm90.npz")); m = F["win_forecast_mask"]
fc["TTM-90 + conformal"] = {"point": np.where(m, F["win_P"], np.nan), "crps": np.where(m, np.abs(F["win_P"] - y), np.nan)}; mask_all &= m
q90 = np.empty(len(rows)); tsa = np.empty(len(rows))
for i in range(len(rows)):
    c = ctx[i][ctx[i] >= 0]; h = Z["dur"][c]; h50, h2 = h[-50:], h[-2:]
    q90[i] = min(np.quantile(h50, 0.9) * 1.1, lim[i]) if len(h50) >= 3 else lim[i]
    tsa[i] = np.mean(h2) if len(h2) >= 2 else lim[i]
q90 = np.maximum(q90, 60.0); tsa = np.clip(tsa, 60.0, np.maximum(lim, 60.0))
for nm, e in (("classical:Tsafrir last-2 (completion-causal)", tsa), ("classical:history q90 (completion-causal)", q90), ("user-elpl", lim)):
    e = np.log1p(e); fc[nm] = {"point": e, "crps": np.abs(e - y)}
ev = mask_all & ~np.isnan(last)
naive = np.abs(last - y)[ev].mean()
R["forecast"] = {"jobs_evaluated": int(ev.sum()), "naive_last_runtime_mae_log": float(naive), "arms": {}}
uq, uinv = np.unique(usr, return_inverse=True)
mase_u, crps_u = {}, {}
for nm, f in fc.items():
    ae = np.abs(f["point"] - y)
    R["forecast"]["arms"][nm] = {"MASE_vs_last_runtime": float(ae[ev].mean() / naive), "CRPS9_or_MAE_log": float(f["crps"][ev].mean())}
    s1, s2, c = np.zeros(len(uq)), np.zeros(len(uq)), np.zeros(len(uq))
    np.add.at(s1, uinv[ev], ae[ev]); np.add.at(s2, uinv[ev], f["crps"][ev]); np.add.at(c, uinv[ev], 1.0)
    mase_u[nm], crps_u[nm] = s1, s2
cnt_u = c
Rk = list(fc.keys())
Ub = np.concatenate(list(counts(len(uq), B)), 0)
mb = {k: (Ub @ mase_u[k]) / (Ub @ cnt_u) / naive for k in Rk}
cb = {k: (Ub @ crps_u[k]) / (Ub @ cnt_u) for k in Rk}
names_dec = {nm: (key(nm, 0.5) if nm in MODELS else nm) for nm in Rk}
pt = {nm: POINT[names_dec[nm]] for nm in Rk}
pos = np.searchsorted(uq, uniq)
full_S = np.zeros((len(names), len(uq))); full_C = np.zeros_like(full_S); full_S[:, pos] = S; full_C[:, pos] = Cn
db = {nm: (Ub @ full_S[names.index(names_dec[nm])]) / np.maximum(Ub @ full_C[names.index(names_dec[nm])], 1) for nm in Rk}
tau_m = kendalltau([pt[k] for k in Rk], [R["forecast"]["arms"][k]["MASE_vs_last_runtime"] for k in Rk])[0]
tau_c = kendalltau([pt[k] for k in Rk], [R["forecast"]["arms"][k]["CRPS9_or_MAE_log"] for k in Rk])[0]
tm = [kendalltau([db[k][b] for k in Rk], [mb[k][b] for k in Rk])[0] for b in range(B)]
tc = [kendalltau([db[k][b] for k in Rk], [cb[k][b] for k in Rk])[0] for b in range(B)]
inv_ = []
sig = lambda c_: c_[0] > 0 or c_[1] < 0
for a, b2 in itertools.combinations(Rk, 2):
    do = db[a] - db[b2]
    for fname, fb in (("MASE", mb), ("CRPS", cb)):
        df = fb[a] - fb[b2]; co, cf = np.percentile(do, [2.5, 97.5]), np.percentile(df, [2.5, 97.5])
        if sig(co) and sig(cf) and np.sign(do.mean()) != np.sign(df.mean()):
            inv_.append({"pair": f"{a} vs {b2}", "forecast_metric": fname, "decision_diff_ci95": co.tolist(), "forecast_diff_ci95": cf.tolist()})
R["rank"] = {"arms": Rk, "decision_mean_bs": pt, "kendall_decision_vs_MASE": float(tau_m), "ci95_MASE": np.nanpercentile(tm, [2.5, 97.5]).tolist(),
             "kendall_decision_vs_CRPS": float(tau_c), "ci95_CRPS": np.nanpercentile(tc, [2.5, 97.5]).tolist(), "inversions": inv_}
for fn in ("bolt", "chronos2", "chronos2cov", "moirai2", "timesfm", "ttm90"):
    R["latency_cpu_2_threads"][fn] = json.load(open(os.path.join(pdir, f"lat_{fn}.json")))
json.dump(R, open(out, "w"), indent=1, default=float); print("ANALYSIS_PM_DONE")

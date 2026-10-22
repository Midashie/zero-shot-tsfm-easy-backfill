"""G3 on F-DATA: leakage quantification, completion-causal estimators, shifted-future test.

Usage: python3 -I g3_fdata.py CENTILE_REPO PARQUET MANIFEST_CSV TSAFRIR_NPZ OUT_DIR
(CENTILE's loader reads ~/nfm/data/fdata/*.parquet, so that path must exist.)
"""
import sys, os, json, importlib
import numpy as np, pandas as pd

repo, pq, man_csv, ts_npz, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.join(repo, "pretrain"))
rb = importlib.import_module("run_backfill_replay")
d_rb, w_rb, t0 = rb.load()                       # CENTILE's own window
est_sub_q90 = rb.hist_quantile_estimates(d_rb, w_rb, 0.9)   # CENTILE's submission-order history q90
est_sub_ts = np.load(ts_npz)["est"]              # CENTILE's submission-order Tsafrir last-2

raw = pd.read_parquet(pq, columns=["usr", "adt", "sdt", "edt", "duration", "elpl"])
raw["src_row"] = np.arange(len(raw))
u = lambda s: pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
raw["t"], raw["sd"], raw["ed"] = u(raw["adt"]), u(raw["sdt"]), u(raw["edt"])
raw["dur"] = pd.to_numeric(raw["duration"], errors="coerce").clip(lower=1)
raw["lim_decl"] = pd.to_numeric(raw["elpl"], errors="coerce")
raw["lim"] = np.maximum(raw["lim_decl"], raw["dur"])

man = pd.read_csv(man_csv)
w = raw.iloc[man["parquet_row"].values].reset_index(drop=True)
assert np.array_equal(w["t"].values, w_rb["t"].values) and np.array_equal(w["dur"].values, w_rb["dur"].values)
assert np.array_equal(w["lim"].values, w_rb["lim"].values)
n = len(w)

res = {"window_jobs": int(n)}
# ---------- own-runtime leak in the declared limit ----------
over = (raw["lim_decl"] < raw["dur"])
wover = (w["lim_decl"] < w["dur"])
res["limit_patch"] = {
    "rows_file_with_runtime_above_declared_limit": int(over.sum()),
    "window_jobs_with_runtime_above_declared_limit": int(wover.sum()),
    "share_of_window_jobs": round(float(wover.mean()), 4),
    "note": "CENTILE sets limit = max(declared, true runtime), so for these jobs the 'declared limit' equals the true runtime"}

# ---------- submission-order history (CENTILE) vs completion-causal history ----------
d_sorted = raw.sort_values("adt")                # same call as CENTILE's loader
d_sorted = d_sorted.dropna(subset=["dur", "lim_decl"])
rows_by_user = {}
for usr, g in d_sorted.groupby("usr", sort=False):
    rows_by_user[usr] = g
# position of each window job in its user's submission-order list
pos = {}
for usr, g in rows_by_user.items():
    for p, sr in enumerate(g["src_row"].values):
        pos[sr] = p


def leak_flags(k):
    """share of window jobs whose last-k submission-order history contains a job that had not finished at submit."""
    flag = np.zeros(n, bool)
    for i in range(n):
        g = rows_by_user[w["usr"].iat[i]]
        p = pos[w["src_row"].iat[i]]
        h = g["ed"].values[max(0, p - k):p]
        flag[i] = bool((h > w["t"].iat[i]).any())
    return flag


f2, f50 = leak_flags(2), leak_flags(50)
res["unfinished_history"] = {
    "tsafrir_last2_has_unfinished_job": round(float(f2.mean()), 4),
    "hist_last50_has_unfinished_job": round(float(f50.mean()), 4)}

# completion-causal arrays per user, ordered by completion time
comp = {}
for usr, g in rows_by_user.items():
    o = np.argsort(g["ed"].values, kind="stable")
    comp[usr] = (g["ed"].values[o], g["t"].values[o], g["dur"].values[o])


def causal_hist(usr, t_i, k, dur_override=None):
    ed, ts, du = comp[usr]
    idx = int(np.searchsorted(ed, t_i, side="right"))
    m = ts[:idx] < t_i
    return du[:idx][m][-k:]


est_c_q90 = np.empty(n)
est_c_ts = np.empty(n)
for i in range(n):
    usr, ti, lim = w["usr"].iat[i], w["t"].iat[i], w["lim"].iat[i]
    h50 = causal_hist(usr, ti, 50)
    est_c_q90[i] = min(np.quantile(h50, 0.9) * 1.1, lim) if len(h50) >= 3 else lim
    h2 = causal_hist(usr, ti, 2)
    est_c_ts[i] = np.mean(h2) if len(h2) >= 2 else lim
est_c_q90 = np.maximum(est_c_q90, 60.0)
est_c_ts = np.clip(est_c_ts, 60.0, np.maximum(w["lim"].values, 60.0))
np.savez(os.path.join(out, "causal_q90.npz"), est=est_c_q90)
np.savez(os.path.join(out, "causal_tsafrir.npz"), est=est_c_ts)

dur = w["dur"].values
res["estimate_change"] = {
    "q90_jobs_with_different_estimate": round(float((np.abs(est_c_q90 - est_sub_q90) > 1e-9).mean()), 4),
    "tsafrir_jobs_with_different_estimate": round(float((np.abs(est_c_ts - est_sub_ts) > 1e-9).mean()), 4),
    "q90_coverage_submission_order": round(float((est_sub_q90 >= dur).mean()), 4),
    "q90_coverage_completion_causal": round(float((est_c_q90 >= dur).mean()), 4),
    "tsafrir_coverage_submission_order": round(float((est_sub_ts >= dur).mean()), 4),
    "tsafrir_coverage_completion_causal": round(float((est_c_ts >= dur).mean()), 4),
    "tsafrir_cold_jobs_submission_order": int((np.abs(est_sub_ts - w["lim"].values) < 1e-9).sum()),
    "tsafrir_cold_jobs_completion_causal": int((np.abs(est_c_ts - w["lim"].values) < 1e-9).sum()),
}

# ---------- shifted-future test: perturb the unknown future of unfinished jobs ----------
rng = np.random.default_rng(0)
sample = rng.choice(n, 5000, replace=False)


def sub_est(usr, p, g_dur, k_ts, lim, kind):
    h = g_dur[max(0, p - (50 if kind == "q90" else 2)):p]
    if kind == "q90":
        return max(min(np.quantile(h, 0.9) * 1.1, lim) if len(h) >= 3 else lim, 60.0)
    return float(np.clip(np.mean(h) if len(h) >= 2 else lim, 60.0, max(lim, 60.0)))


def caus_est(g_ed, g_t, g_dur, t_i, lim, kind):
    o = np.argsort(g_ed, kind="stable")
    ed, ts, du = g_ed[o], g_t[o], g_dur[o]
    idx = int(np.searchsorted(ed, t_i, side="right"))
    m = ts[:idx] < t_i
    h = du[:idx][m][-(50 if kind == "q90" else 2):]
    if kind == "q90":
        return max(min(np.quantile(h, 0.9) * 1.1, lim) if len(h) >= 3 else lim, 60.0)
    return float(np.clip(np.mean(h) if len(h) >= 2 else lim, 60.0, max(lim, 60.0)))


chg = {"sub_q90": 0, "sub_ts": 0, "cau_q90": 0, "cau_ts": 0}
for i in sample:
    usr, ti, lim = w["usr"].iat[i], w["t"].iat[i], w["lim"].iat[i]
    g = rows_by_user[usr]
    p = pos[w["src_row"].iat[i]]
    ed, tt, sd, du = g["ed"].values.copy(), g["t"].values, g["sd"].values, g["dur"].values.copy()
    unfinished = ed > ti
    ed2, du2 = ed.copy(), du.copy()
    k = int(unfinished.sum())
    if k:
        new_ed = np.maximum(ed[unfinished], ti + 1) + rng.integers(1, 7 * 86400, size=k)   # a different future
        ed2[unfinished] = new_ed
        du2[unfinished] = np.maximum(new_ed - sd[unfinished], 1)
    for kind in ("q90", "ts"):
        a0, a1 = sub_est(usr, p, du, 0, lim, kind), sub_est(usr, p, du2, 0, lim, kind)
        b0, b1 = caus_est(ed, tt, du, ti, lim, kind), caus_est(ed2, tt, du2, ti, lim, kind)
        chg["sub_" + kind] += abs(a0 - a1) > 1e-9
        chg["cau_" + kind] += abs(b0 - b1) > 1e-9
res["shifted_future_test"] = {
    "sampled_jobs": int(len(sample)),
    "perturbation": "every job of the same user still unfinished at the decision time gets a new, later end time and runtime",
    "submission_order_q90_estimates_changed": int(chg["sub_q90"]),
    "submission_order_tsafrir_estimates_changed": int(chg["sub_ts"]),
    "completion_causal_q90_estimates_changed": int(chg["cau_q90"]),
    "completion_causal_tsafrir_estimates_changed": int(chg["cau_ts"]),
}
with open(os.path.join(out, "g3_fdata.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

"""Task B replays: EASY backfilling at one node count, one process. Only the walltime estimate changes between arms.
Usage: python3 -I replay_b.py CENTILE_REPO FORECAST_DIR G3_DIR OUT_DIR NODES
Estimates: clip(expm1(quantile forecast of log1p runtime), 60 s, declared limit); jobs with fewer than 3 finished jobs get the declared limit.
TTM carries quantiles through split conformal on the calibration set (log1p space).
"""
import sys, os, json, importlib
import numpy as np

repo, fdir, g3dir, out, NODES = sys.argv[1:6]
NODES = int(NODES)
os.environ["NODES"] = str(NODES)
sys.path.insert(0, os.path.join(repo, "pretrain"))
rb = importlib.import_module("run_backfill_replay")
os.makedirs(out, exist_ok=True)
d, w, t0 = rb.load()
sub = w["t"].values.astype(float); nn = w["nn"].values; dur = w["dur"].values.astype(float); lim = w["lim"].values.astype(float)
usr = w["usr"].values
cap = NODES
fit = nn <= cap
Q_IDX = {0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7}
Z = np.load(os.path.join(fdir, "contexts_c96.npz"))
assert np.array_equal(Z["dur"][Z["win_rows"]], dur)


def clip(e):
    return np.minimum(np.maximum(e, 60.0), np.maximum(lim, 60.0))


arms = {"user-elpl": np.maximum(lim, 60.0), "oracle": np.maximum(dur, 60.0),
        "classical:Tsafrir last-2 (completion-causal)": np.load(os.path.join(g3dir, "causal_tsafrir.npz"))["est"],
        "classical:history q90 (completion-causal)": np.load(os.path.join(g3dir, "causal_q90.npz"))["est"]}
for name, fn in (("Chronos-Bolt-small", "bolt"), ("Chronos-2", "chronos2"), ("Chronos-2 + covariates", "chronos2cov"), ("Moirai-2.0", "moirai2"), ("TimesFM-2.5", "timesfm")):
    F = np.load(os.path.join(fdir, f"F_{fn}.npz"))
    Q, m = F["win_Q"], F["win_forecast_mask"]
    for q, j in Q_IDX.items():
        e = lim.copy()
        e[m] = np.expm1(Q[m, j])
        arms[f"{name} q{q}"] = clip(e)
F = np.load(os.path.join(fdir, "F_ttm90.npz"))
res = np.sort((np.log1p(Z["dur"][Z["cal_rows"]]) - F["cal_P"])[F["cal_forecast_mask"]])
n_cal = len(res)
for q in Q_IDX:
    k = min(int(np.ceil((n_cal + 1) * q)) - 1, n_cal - 1)
    e = lim.copy()
    m = F["win_forecast_mask"]
    e[m] = np.expm1(F["win_P"][m] + res[k])
    arms[f"TTM-90 + conformal q{q}"] = clip(e)

if os.environ.get("ONLY_TIMESFM"):                      # add the TimesFM arms to an existing replay set; reference arms are not rerun
    arms = {k: v for k, v in arms.items() if k.startswith("TimesFM")}
SUFFIX = "_timesfm" if os.environ.get("ONLY_TIMESFM") else ""
fsub, fnn, fdur, flim, fusr = sub[fit], nn[fit], dur[fit], lim[fit], usr[fit]
summary = {"nodes": NODES, "dropped_too_wide": int((~fit).sum()), "arms": {}}
store = {"fit_index": np.where(fit)[0]}
for nm, est in arms.items():
    est = np.asarray(est, float)[fit]
    start, killed, nbf = rb.easy_sim(fsub, fnn, fdur, flim, est, cap)
    ok = start >= 0
    wait = start[ok] - fsub[ok]; run = np.minimum(fdur[ok], flim[ok])
    bs = np.maximum((wait + run) / np.maximum(run, 60.0), 1.0)
    mk = (start[ok] + run).max() - fsub.min()
    summary["arms"][nm] = {"mean_bs": float(bs.mean()), "p95_bs": float(np.quantile(bs, 0.95)),
                           "util_pct": float((fnn[ok] * run).sum() / (cap * mk) * 100), "backfilled": int(nbf),
                           "median_wait_s": float(np.median(wait)), "started_pct": float(ok.mean() * 100),
                           "killed": int(killed.sum())}
    full = np.full(len(fsub), np.nan, np.float32)
    full[ok] = bs
    store[nm] = full
    print(f"{NODES} {nm:50s} mean {bs.mean():8.2f} p95 {np.quantile(bs, 0.95):8.2f}", flush=True)
np.savez_compressed(os.path.join(out, f"replay_bs_{NODES}{SUFFIX}.npz"), usr=fusr, **{k.replace("/", "_"): v for k, v in store.items()})
with open(os.path.join(out, f"replay_summary_{NODES}{SUFFIX}.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("REPLAY_DONE")

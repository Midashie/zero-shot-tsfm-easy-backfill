"""PM100 EASY-backfilling replays. Only the walltime estimate changes between arms.
Usage: python3 -I replay_pm.py CENTILE_REPO CTX_NPZ FORECAST_DIR OUT_DIR NODES MODE      MODE: ref (user limits + true runtimes) or full
Estimates: clip(expm1(quantile forecast of log1p runtime), 60 s, kill limit); users with fewer than 3 finished jobs get the kill limit.
Kill limit = max(declared limit, true runtime), as in CENTILE's replay. TTM carries quantiles through split conformal (log1p space)."""
import sys, os, json, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pm_common import load_pm

repo, ctx_npz, fdir, out, NODES, MODE = sys.argv[1:7]
NODES = int(NODES); os.environ["NODES"] = str(NODES)
sys.path.insert(0, os.path.join(repo, "pretrain"))
rb = importlib.import_module("run_backfill_replay")
os.makedirs(out, exist_ok=True)
d = load_pm(); Z = np.load(ctx_npz); rows = Z["win_rows"]
assert np.array_equal(Z["dur"][rows], d["dur"][rows])
sub = d["ts"][rows].astype(float); nn = d["nn"][rows]; dur = d["dur"][rows]
lim = np.maximum(d["lim"][rows], dur); usr = d["usr"][rows]
assert (np.diff(sub) >= 0).all()
cap = NODES; fit = nn <= cap
clip = lambda e: np.minimum(np.maximum(e, 60.0), np.maximum(lim, 60.0))
arms = {"user-elpl": np.maximum(lim, 60.0), "oracle": np.maximum(dur, 60.0)}
if MODE == "full":
    # completion-causal classical arms, from the verified contexts (same definitions as g3_fdata.py)
    ctx, ln = Z["win_ctx"], Z["win_len"]; logd = None
    q90 = np.empty(len(rows)); tsa = np.empty(len(rows))
    for i in range(len(rows)):
        c = ctx[i][ctx[i] >= 0]; h = Z["dur"][c]
        h50 = h[-50:]; h2 = h[-2:]
        q90[i] = min(np.quantile(h50, 0.9) * 1.1, lim[i]) if len(h50) >= 3 else lim[i]
        tsa[i] = np.mean(h2) if len(h2) >= 2 else lim[i]
    arms["classical:Tsafrir last-2 (completion-causal)"] = clip(tsa)
    arms["classical:history q90 (completion-causal)"] = np.maximum(q90, 60.0)
    Q_IDX = {0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7}
    for name, fn in (("Chronos-Bolt-small", "bolt"), ("Chronos-2", "chronos2"), ("Chronos-2 + covariates", "chronos2cov"), ("Moirai-2.0", "moirai2"), ("TimesFM-2.5", "timesfm")):
        F = np.load(os.path.join(fdir, f"F_{fn}.npz")); Q, m = F["win_Q"], F["win_forecast_mask"]
        for q, j in Q_IDX.items():
            e = lim.copy(); e[m] = np.expm1(Q[m, j]); arms[f"{name} q{q}"] = clip(e)
    F = np.load(os.path.join(fdir, "F_ttm90.npz"))
    res = np.sort((np.log1p(Z["dur"][Z["cal_rows"]]) - F["cal_P"])[F["cal_forecast_mask"]]); n_cal = len(res)
    for q in Q_IDX:
        k = min(int(np.ceil((n_cal + 1) * q)) - 1, n_cal - 1)
        e = lim.copy(); m = F["win_forecast_mask"]; e[m] = np.expm1(F["win_P"][m] + res[k]); arms[f"TTM-90 + conformal q{q}"] = clip(e)
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
    summary["arms"][nm] = {"mean_bs": float(bs.mean()), "p95_bs": float(np.quantile(bs, 0.95)), "util_pct": float((fnn[ok] * run).sum() / (cap * mk) * 100),
                           "backfilled": int(nbf), "median_wait_s": float(np.median(wait)), "started_pct": float(ok.mean() * 100), "killed": int(killed.sum())}
    full = np.full(len(fsub), np.nan, np.float32); full[ok] = bs; store[nm] = full
    print(f"{NODES} {nm:50s} mean {bs.mean():8.2f} p95 {np.quantile(bs, 0.95):8.2f} util {summary['arms'][nm]['util_pct']:.1f}", flush=True)
suffix = "" if MODE == "full" else "_ref"
np.savez_compressed(os.path.join(out, f"replay_bs_{NODES}{suffix}.npz"), usr=fusr, **{k.replace("/", "_"): v for k, v in store.items()})
json.dump(summary, open(os.path.join(out, f"replay_summary_{NODES}{suffix}.json"), "w"), indent=1)
print("REPLAY_DONE")

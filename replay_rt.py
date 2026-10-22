"""EASY replays with replay-timeline contexts. Usage: python3 -I replay_rt.py CENTILE_REPO TRACE NODES RT_DIR
Arms: user limits, true runtimes, completion-causal Tsafrir last-2 and history q90 (rebuilt on the replay-timeline contexts), and four models
(Chronos-Bolt, Chronos-2, Moirai-2.0, TTM-90 + conformal) at q 0.5 to 0.8. Same estimate rule as replay_b.py / replay_pm.py."""
import sys, os, json, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rt_common import get_trace, OUT

repo, trace, NODES, rt = sys.argv[1:5]; NODES = int(NODES); os.environ["NODES"] = str(NODES)
sys.path.insert(0, os.path.join(repo, "pretrain"))
rb = importlib.import_module("run_backfill_replay")
orig = f"{OUT}/{'taskb' if trace == 'fdata' else 'pm100'}"
Zo, usr_all, _ = get_trace(trace)
Zr = np.load(os.path.join(rt, "contexts_c96.npz")); rows = Zr["win_rows"]
assert np.array_equal(rows, Zo["win_rows"])
sub = Zo["ts"][rows].astype(float); nn = Zo["nn"][rows]; dur = Zo["dur"][rows]; lim = np.maximum(Zo["lim"][rows], dur); usr = usr_all[rows]
fit = nn <= NODES
clip = lambda e: np.minimum(np.maximum(e, 60.0), np.maximum(lim, 60.0))
ctx = Zr["win_ctx"]; q90 = np.empty(len(rows)); tsa = np.empty(len(rows))
for i in range(len(rows)):
    c = ctx[i][ctx[i] >= 0]; h = Zo["dur"][c]; h50, h2 = h[-50:], h[-2:]
    q90[i] = min(np.quantile(h50, 0.9) * 1.1, lim[i]) if len(h50) >= 3 else lim[i]
    tsa[i] = np.mean(h2) if len(h2) >= 2 else lim[i]
arms = {"user-elpl": np.maximum(lim, 60.0), "oracle": np.maximum(dur, 60.0),
        "classical:Tsafrir last-2 (completion-causal)": clip(tsa), "classical:history q90 (completion-causal)": np.maximum(q90, 60.0)}
Q_IDX = {0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7}
for name, fn in (("Chronos-Bolt-small", "bolt"), ("Chronos-2", "chronos2"), ("Moirai-2.0", "moirai2")):
    F = np.load(os.path.join(rt, f"F_{fn}.npz")); Q, m = F["win_Q"], F["win_forecast_mask"]
    for q, j in Q_IDX.items():
        e = lim.copy(); e[m] = np.expm1(Q[m, j]); arms[f"{name} q{q}"] = clip(e)
F = np.load(os.path.join(rt, "F_ttm90.npz")); Fo = np.load(os.path.join(orig, "F_ttm90.npz"))
res = np.sort((np.log1p(Zo["dur"][Zo["cal_rows"]]) - Fo["cal_P"])[Fo["cal_forecast_mask"]]); n_cal = len(res)     # calibration set is unchanged (pre-window jobs)
for q in Q_IDX:
    k = min(int(np.ceil((n_cal + 1) * q)) - 1, n_cal - 1)
    e = lim.copy(); m = F["win_forecast_mask"]; e[m] = np.expm1(F["win_P"][m] + res[k]); arms[f"TTM-90 + conformal q{q}"] = clip(e)
fsub, fnn, fdur, flim, fusr = sub[fit], nn[fit], dur[fit], lim[fit], usr[fit]
summary = {"nodes": NODES, "dropped_too_wide": int((~fit).sum()), "arms": {}}; store = {"fit_index": np.where(fit)[0]}
for nm, est in arms.items():
    est = np.asarray(est, float)[fit]
    start, killed, nbf = rb.easy_sim(fsub, fnn, fdur, flim, est, NODES)
    ok = start >= 0; wait = start[ok] - fsub[ok]; run = np.minimum(fdur[ok], flim[ok])
    bs = np.maximum((wait + run) / np.maximum(run, 60.0), 1.0); mk = (start[ok] + run).max() - fsub.min()
    summary["arms"][nm] = {"mean_bs": float(bs.mean()), "p95_bs": float(np.quantile(bs, 0.95)), "util_pct": float((fnn[ok] * run).sum() / (NODES * mk) * 100),
                           "backfilled": int(nbf), "median_wait_s": float(np.median(wait)), "started_pct": float(ok.mean() * 100), "killed": int(killed.sum())}
    full = np.full(len(fsub), np.nan, np.float32); full[ok] = bs; store[nm] = full
    print(f"{NODES} {nm:50s} mean {bs.mean():8.2f} p95 {np.quantile(bs, 0.95):8.2f}", flush=True)
np.savez_compressed(os.path.join(rt, f"replay_bs_{NODES}.npz"), usr=fusr, **{k.replace("/", "_"): v for k, v in store.items()})
json.dump(summary, open(os.path.join(rt, f"replay_summary_{NODES}.json"), "w"), indent=1); print("REPLAY_DONE")

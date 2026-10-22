"""Replay-timeline contexts. Usage: python3 -I build_rt.py CENTILE_REPO TRACE NODES OUT_DIR   TRACE: fdata or pm100
A window job's context = the user's jobs submitted earlier AND finished by its submit time, where a replay-window job finishes at
(its start time in the user-limit replay) + min(runtime, kill limit), and every other job (history before the window, jobs the replay drops
for being wider than the machine) finishes at its recorded end time. The user-limit replay does not depend on any model."""
import sys, os, json, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from rt_common import get_trace

repo, trace, NODES, out = sys.argv[1:5]; NODES = int(NODES); C = 96
os.environ["NODES"] = str(NODES); sys.path.insert(0, os.path.join(repo, "pretrain"))
rb = importlib.import_module("run_backfill_replay")
os.makedirs(out, exist_ok=True)
Z, usr, ed = get_trace(trace)
rows = Z["win_rows"]; ts, dur = Z["ts"], Z["dur"]
lim_k = np.maximum(Z["lim"], dur)
sub, nn, du, lm = ts[rows].astype(float), Z["nn"][rows], dur[rows], lim_k[rows]
fit = nn <= NODES
start, killed, nbf = rb.easy_sim(sub[fit], nn[fit], du[fit], lm[fit], np.maximum(lm[fit], 60.0), NODES)   # user-limit estimates
assert (start >= 0).all()
f = ed.astype(float).copy()
f[rows[fit]] = start + np.minimum(du[fit], lm[fit])
shift = f[rows[fit]] - ed[rows[fit]]
by = {}
idx = np.argsort(usr, kind="stable"); us = usr[idx]
cuts = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
for a, b in zip(cuts[:-1], cuts[1:]):
    r = idx[a:b]; o = np.argsort(f[r], kind="stable"); by[us[a]] = (r[o], f[r][o], ts[r][o])
M = -np.ones((len(rows), C), np.int64); L = np.zeros(len(rows), np.int64)
for i, rw in enumerate(rows):
    rr, ff, tt = by[usr[rw]]; t_i = ts[rw]
    k = int(np.searchsorted(ff, t_i, side="right")); m = tt[:k] < t_i
    c = rr[:k][m][-C:]; L[i] = len(c)
    if len(c): M[i, C - len(c):] = c
    assert (f[c] <= t_i).all() and (ts[c] < t_i).all()
old = Z["win_ctx"]
same_rows = np.array([np.array_equal(old[i][old[i] >= 0], M[i][M[i] >= 0]) for i in range(len(rows))])
Zd = {k: Z[k] for k in Z.files}; Zd["win_ctx"] = M; Zd["win_len"] = L
np.savez_compressed(os.path.join(out, "contexts_c96.npz"), **Zd)
info = {"trace": trace, "nodes": NODES, "window_jobs": int(len(rows)), "dropped_too_wide": int((~fit).sum()),
        "median_shift_s_replay_finish_minus_recorded_end": float(np.median(shift)), "mean_abs_shift_s": float(np.abs(shift).mean()),
        "share_jobs_finishing_later_in_replay": float((shift > 1).mean()), "share_jobs_finishing_earlier_in_replay": float((shift < -1).mean()),
        "share_contexts_identical_to_recorded_time": float(same_rows.mean()), "jobs_with_fewer_than_3_context": int((L < 3).sum()),
        "jobs_with_fewer_than_3_context_recorded": int((Z["win_len"] < 3).sum())}
json.dump(info, open(os.path.join(out, "build_rt.json"), "w"), indent=1); print(json.dumps(info, indent=1))

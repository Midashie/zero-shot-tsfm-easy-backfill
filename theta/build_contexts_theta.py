"""Theta 2017 contexts, same rules as build_contexts_pm.py. Usage: python3 -I build_contexts_theta.py OUT_NPZ [C]"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from theta_common import load_theta

out = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 96
d = load_theta(int(os.environ['THETA_YEAR']) if os.environ.get('THETA_YEAR') else None)
ts, ed, dur, usr = d["ts"], d["ed"], d["dur"], d["usr"]
rows = d["win_rows"]
by_user = {}
idx = np.argsort(usr, kind="stable")
us = usr[idx]
cuts = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
for a, b in zip(cuts[:-1], cuts[1:]):
    r = idx[a:b]
    o = np.argsort(ed[r], kind="stable")
    by_user[us[a]] = (r[o], ed[r][o], ts[r][o])


def context(u, t_i):
    rr, ee, tt = by_user[u]
    k = int(np.searchsorted(ee, t_i, side="right"))
    m = tt[:k] < t_i
    return rr[:k][m][-C:]


def build(job_rows, tag):
    n = len(job_rows); M = -np.ones((n, C), np.int64); L = np.zeros(n, np.int64)
    for i, rw in enumerate(job_rows):
        c = context(usr[rw], ts[rw]); L[i] = len(c)
        if len(c): M[i, C - len(c):] = c
        assert (ed[c] <= ts[rw]).all() and (ts[c] < ts[rw]).all()
    return {f"{tag}_rows": np.asarray(job_rows), f"{tag}_ctx": M, f"{tag}_len": L}


res = build(rows, "win")
t0 = int(ts[rows].min()); res["replay_start_unix"] = np.int64(t0)
rng = np.random.default_rng(3)
cand = np.where((ts >= t0 - 7 * 86400) & (ts < t0) & (ed <= t0))[0]
cand = np.array([c for c in cand if len(context(usr[c], ts[c])) >= 3])
cal = np.sort(rng.choice(cand, min(5000, len(cand)), replace=False))
res.update(build(cal, "cal"))
res.update({"ts": ts, "dur": dur, "lim": d["lim"], "nn": d["nn"], "gap": d["gap"]})
np.savez_compressed(out, **res)
info = {"C": C, "window_jobs": int(len(rows)), "window_users": int(len(set(usr[rows]))), "window_jobs_with_fewer_than_3_context": int((res["win_len"] < 3).sum()),
        "share_full_context": float((res["win_len"] == C).mean()), "median_context_len": float(np.median(res["win_len"])),
        "calibration_candidates": int(len(cand)), "calibration_n": int(len(cal)), "replay_start_unix": t0}
print(json.dumps(info, indent=1))

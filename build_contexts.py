"""Task B inputs. For every replayed F-DATA job, the context is the runtimes of the same user's last C jobs that had FINISHED
at the job's submit time (recorded end time at or before the submission, submitted strictly earlier, ordered by end time).
Also builds a calibration set (warm-up jobs whose end time is before the replay starts) for split conformal.

Usage: python3 -I build_contexts.py PARQUET MANIFEST_CSV OUT_NPZ [C]
"""
import sys, json
import numpy as np, pandas as pd

pq, man_csv, out = sys.argv[1:4]
C = int(sys.argv[4]) if len(sys.argv) > 4 else 96

raw = pd.read_parquet(pq, columns=["usr", "adt", "edt", "nnumr", "elpl", "duration"])
raw["row"] = np.arange(len(raw))
u = lambda s: pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
raw["ts"], raw["ed"] = u(raw["adt"]), u(raw["edt"])
raw["dur"] = pd.to_numeric(raw["duration"], errors="coerce").clip(lower=1)
raw["lim"] = pd.to_numeric(raw["elpl"], errors="coerce")
raw["nn"] = pd.to_numeric(raw["nnumr"], errors="coerce").fillna(1).clip(lower=1)
ok = raw.dropna(subset=["dur", "lim"]).copy()
assert len(ok) == len(raw)

# inter-submission gap per job: seconds since the same user's previous submission (0 for the first)
ok = ok.sort_values(["usr", "ts"], kind="stable")
ok["gap"] = ok.groupby("usr")["ts"].diff().fillna(0).clip(lower=0)
ok = ok.sort_values("row")
gap = ok["gap"].values.astype(np.float64)
ts, ed, dur, lim, nn = ok["ts"].values, ok["ed"].values, ok["dur"].values.astype(np.float64), ok["lim"].values.astype(np.float64), ok["nn"].values.astype(np.float64)

man = pd.read_csv(man_csv)
rows = man["parquet_row"].values
assert np.array_equal(dur[rows], man["true_runtime_s"].values) and np.array_equal(ts[rows], man["submit_unix"].values)

by_user = {}
for usr, g in ok.groupby("usr", sort=False):
    o = np.argsort(g["ed"].values, kind="stable")
    by_user[usr] = (g["row"].values[o], g["ed"].values[o], g["ts"].values[o])


def context(usr, t_i):
    rr, ee, tt = by_user[usr]
    k = int(np.searchsorted(ee, t_i, side="right"))
    m = tt[:k] < t_i
    return rr[:k][m][-C:]


def build(job_rows, tag):
    n = len(job_rows)
    M = -np.ones((n, C), np.int64)
    L = np.zeros(n, np.int64)
    for i, rw in enumerate(job_rows):
        c = context(ok["usr"].values[rw], ts[rw])
        L[i] = len(c)
        if len(c):
            M[i, C - len(c):] = c                      # left padded, most recent last
        # the context must be strictly earlier and finished
        assert (ed[c] <= ts[rw]).all() and (ts[c] < ts[rw]).all()
    return {f"{tag}_rows": np.asarray(job_rows), f"{tag}_ctx": M, f"{tag}_len": L}


res = build(rows, "win")
t0 = int(ts[rows].min())
res["replay_start_unix"] = np.int64(t0)
# calibration: warm-up jobs submitted in the 7 days before the replay whose end time is at or before the replay start
rng = np.random.default_rng(3)
cand = np.where((ts >= t0 - 7 * 86400) & (ts < t0) & (ed <= t0))[0]
cand = np.array([c for c in cand if len(context(ok["usr"].values[c], ts[c])) >= 3])
cal = np.sort(rng.choice(cand, min(5000, len(cand)), replace=False))
res.update(build(cal, "cal"))
res.update({"ts": ts, "dur": dur, "lim": lim, "nn": nn, "gap": gap})
np.savez_compressed(out, **res)
info = {"C": C, "window_jobs": int(len(rows)), "window_jobs_with_fewer_than_3_context": int((res["win_len"] < 3).sum()),
        "share_full_context": float((res["win_len"] == C).mean()), "median_context_len": float(np.median(res["win_len"])),
        "calibration_candidates": int(len(cand)), "calibration_n": int(len(cal)), "replay_start_unix": t0}
print(json.dumps(info, indent=1))

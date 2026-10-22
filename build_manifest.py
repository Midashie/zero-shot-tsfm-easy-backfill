"""Build the frozen splits and window-id manifest for the zero-shot TSFM pilot.

Reproduces CENTILE's window selection exactly, cross-checks against CENTILE's own functions,
and writes CSV/JSON files. No model is run here.

Usage: python3 -I build_manifest.py CENTILE_REPO NPZ PARQUET OUT_DIR
"""
import sys, os, json, hashlib, importlib, datetime, platform
import numpy as np, pandas as pd

repo, npz_path, pq_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.join(repo, "pretrain"))
sys.path.insert(0, os.path.join(repo, "common"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iso(u):
    return datetime.datetime.fromtimestamp(int(u), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


man = {
    "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "inputs": {"cesnet_hourly.npz": sha256(npz_path), "21_04.parquet": sha256(pq_path)},
}

# ------------------------------------------------------------------ CESNET
SPLIT, HMAX = 168, 24
N = SPLIT + HMAX
STRIDE = 24
EVALMAX = 3000
RECENT_BINS = 28 * 24      # my choice: "most recent weeks" = last 4 weeks of data
CAL_BINS = 28 * 24         # my choice: calibration region = last 4 weeks of the training region
z = np.load(npz_path)
r, v, ts, sid = z["r"], z["valid"], z["ts"], z["sid"]
S, T = r.shape
cut = int(T * 0.8)


def windows_idx(lo, hi, stride):
    """(series row, start bin) of every fully valid window, same order as CENTILE's windows()."""
    rows, starts = [], []
    for i in range(S):
        for s in range(lo, hi - N + 1, stride):
            if v[i, s:s + N].min() > 0:
                rows.append(i)
                starts.append(s)
    return np.array(rows), np.array(starts)


def sample(rows, starts, seed, k):
    if len(rows) > k:
        pick = np.random.default_rng(seed).choice(len(rows), k, replace=False)
        return rows[pick], starts[pick], pick
    return rows, starts, np.arange(len(rows))


def write_windows(name, rows, starts):
    df = pd.DataFrame({
        "window_id": [f"{name}-{k:05d}" for k in range(len(rows))],
        "institution_id": sid[rows], "series_row": rows, "start_bin": starts,
        "start_time_utc": [iso(ts[s]) for s in starts],
        "decision_time_utc": [iso(ts[s + SPLIT]) for s in starts],
    })
    df.to_csv(os.path.join(out, f"cesnet_{name}_windows.csv"), index=False)
    return df


pr, ps = windows_idx(cut, T, STRIDE)
n_primary_pool = len(pr)
pr, ps, _ = sample(pr, ps, 0, EVALMAX)
d_primary = write_windows("primary", pr, ps)

rr, rs = windows_idx(T - RECENT_BINS, T, STRIDE)
n_recent_pool = len(rr)
rr, rs, _ = sample(rr, rs, 1, EVALMAX)
d_recent = write_windows("recent", rr, rs)

cr, cs = windows_idx(cut - CAL_BINS, cut, STRIDE)
n_cal_pool = len(cr)
cr, cs, _ = sample(cr, cs, 2, EVALMAX)
d_cal = write_windows("calibration", cr, cs)

# cross-check the primary sample against CENTILE's own windows() and sampler
cc = importlib.import_module("run_cesnet_chronos")
Ze, SIDe = cc.windows(r, v, ts, cut, T, STRIDE)
k = np.random.default_rng(0).choice(len(Ze), EVALMAX, replace=False)
Ze, SIDe = Ze[k], SIDe[k]
mine = np.stack([r[i, s:s + N] for i, s in zip(pr, ps)])
assert Ze.shape == mine.shape and np.array_equal(Ze, mine) and np.array_equal(SIDe, pr), "primary sample differs from CENTILE"

key = lambda a, b: set(zip(a.tolist(), b.tolist()))
man["cesnet"] = {
    "series_kept": int(S), "hourly_bins": int(T), "first_bin_utc": iso(ts[0]), "last_bin_utc": iso(ts[-1]),
    "train_eval_cut_bin": int(cut), "cut_time_utc": iso(ts[cut]),
    "context_hours": SPLIT, "max_horizon_hours": HMAX, "window_len": N, "stride_hours": STRIDE, "target": "log1p(n_bytes), as stored in the npz",
    "primary": {"rule": "CENTILE protocol: fully valid windows in [cut, T), stride 24, rng(0).choice(pool, 3000)", "pool": n_primary_pool, "n": len(pr), "seed": 0,
                "matches_centile_function": True},
    "recent": {"rule": f"last {RECENT_BINS} bins ({RECENT_BINS // 24} days), fully valid windows, stride 24, rng(1).choice(pool, 3000). My addition, not in CENTILE",
               "pool": n_recent_pool, "n": len(rr), "seed": 1, "overlap_with_primary": len(key(rr, rs) & key(pr, ps))},
    "calibration": {"rule": f"fully valid windows inside [cut-{CAL_BINS}, cut), stride 24, rng(2).choice(pool, 3000). My addition; used only for split conformal",
                    "pool": n_cal_pool, "n": len(cr), "seed": 2, "last_bin_used": int(cs.max() + N - 1),
                    "ends_before_first_eval_bin": bool(cs.max() + N - 1 < cut)},
}
assert man["cesnet"]["calibration"]["ends_before_first_eval_bin"]
assert pr.min() >= 0 and ps.min() >= cut and rs.min() >= T - RECENT_BINS

# ------------------------------------------------------------------ F-DATA
rb = importlib.import_module("run_backfill_replay")
os.environ["HOME"] = os.environ.get("HOME", "/root")
d0, w0, t0_ref = rb.load()  # CENTILE's own loader (reads ~/nfm/data/fdata)
cols = ["jid", "usr", "adt", "nnumr", "elpl", "duration"]
d = pd.read_parquet(pq_path, columns=cols)
d["src_row"] = np.arange(len(d))
d["adt"] = pd.to_datetime(d["adt"], utc=True, errors="coerce")
d = d.dropna(subset=["adt"]).sort_values("adt")
d["t"] = d["adt"].astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
d["dur"] = pd.to_numeric(d["duration"], errors="coerce").clip(lower=1)
d["lim"] = pd.to_numeric(d["elpl"], errors="coerce")
d["nn"] = pd.to_numeric(d["nnumr"], errors="coerce").fillna(1).clip(lower=1).astype(int)
d = d.dropna(subset=["dur", "lim"])
d["lim"] = np.maximum(d["lim"], d["dur"])
t0 = d["t"].iloc[0] + rb.SKIPD * 86400
w = d[(d["t"] >= t0) & (d["t"] < t0 + rb.DAYS * 86400)].head(rb.MAXJ).reset_index(drop=True)
for c in ["t", "nn", "dur", "lim"]:
    assert np.array_equal(w[c].values, w0[c].values), f"F-DATA window column {c} differs from CENTILE"
assert (w["usr"].values == w0["usr"].values).all()
jid_dups_in_window = int(w["jid"].duplicated(keep=False).sum())

w_out = pd.DataFrame({"replay_order": np.arange(len(w)), "parquet_row": w["src_row"], "jid": w["jid"], "usr": w["usr"], "submit_unix": w["t"],
                      "submit_utc": [iso(x) for x in w["t"]], "nodes_requested": w["nn"],
                      "declared_limit_s": w["lim"], "true_runtime_s": w["dur"]})
w_out.to_csv(os.path.join(out, "fdata_replay_jobs.csv"), index=False)

past = d[d["t"] < w["t"].iloc[0]]
cnt = past.groupby("usr").size()
first_idx = w.groupby("usr").head(1)
prior_at_first = first_idx["usr"].map(cnt).fillna(0).astype(int)
sub = w["t"].values.astype(float)
nn = w["nn"].values
dur = w["dur"].values.astype(float)
span = float(sub.max() - sub.min()) + 1.0
caps = {}
for cap in (20000, 23000):
    fit = nn <= cap
    off = float((nn[fit].astype(np.float64) * dur[fit]).sum()) / span
    caps[str(cap)] = {"dropped_too_wide": int((~fit).sum()), "offered_nodes_avg": round(off, 1), "offered_to_capacity": round(off / cap, 4)}
man["fdata"] = {
    "file": "21_04.parquet", "rows_in_file": int(len(pd.read_parquet(pq_path, columns=["jid"]))), "rows_after_cleaning": int(len(d)),
    "rule": "CENTILE loader: sort by submit time, drop rows with missing submit/runtime/limit, limit=max(limit, runtime), runtime>=1 s; "
            "window starts 7 days after first job and lasts 14 days; history = every earlier job in the file",
    "first_job_utc": iso(d["t"].iloc[0]), "replay_start_utc": iso(t0), "replay_end_utc": iso(t0 + rb.DAYS * 86400),
    "warmup_days": rb.SKIPD, "replay_days": rb.DAYS, "jobs_before_replay": int(len(past)),
    "replay_jobs": int(len(w)), "replay_users": int(w["usr"].nunique()),
    "users_with_fewer_than_3_prior_jobs_at_first_replay_job": int((prior_at_first < 3).sum()),
    "capacities": caps, "span_days": round(span / 86400, 3),
    "jid_is_not_a_row_key": {"rows_in_file": int(len(d)), "distinct_jid_in_file": int(d["jid"].nunique()),
        "rows_in_replay_window": int(len(w)), "distinct_jid_in_window": int(w["jid"].nunique()), "window_rows_sharing_a_jid": jid_dups_in_window,
        "window_exact_duplicate_rows": int(w_out.duplicated(["jid", "usr", "submit_unix", "nodes_requested", "declared_limit_s", "true_runtime_s"]).sum()),
        "window_rows_same_jid_user_nodes_limit_but_different_time_or_runtime": int(w_out.duplicated(["jid", "usr", "nodes_requested", "declared_limit_s"]).sum() - w_out.duplicated(["jid", "usr", "submit_unix", "nodes_requested", "declared_limit_s", "true_runtime_s"]).sum()),
        "note": "the replay treats each row as one job and keeps all duplicates, as CENTILE does, so rows are keyed by parquet_row. I have not established why a jid repeats"},
    "matches_centile_loader": True, "replay_is_deterministic": True,
    "sort_note": "sort_values('adt') is not a stable sort; replay_order in fdata_replay_jobs.csv freezes the tie order that produced the published numbers",
}
with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(man, f, indent=2)
print(json.dumps(man, indent=2))

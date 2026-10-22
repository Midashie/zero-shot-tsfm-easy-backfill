"""Theta (ALCF) loader, 2017 file. Same array layout as pm_common.load_pm; every array is aligned to the table order (submit time, job id).
user = USERNAME_GENID; submit = QUEUED_TIMESTAMP; end = END_TIMESTAMP; nodes = NODES_REQUESTED; runtime = RUNTIME_SECONDS floored at 1;
declared limit = WALLTIME_SECONDS. Timestamps are read as logged (no time zone shift)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, pandas as pd

UP = _IN + "/ANL-ALCF-DJC-THETA_"
FILES = {2017: UP + "20170701_20171231.csv", 2018: UP + "20180101_20181231.csv", 2019: UP + "20190101_20191231.csv", 2020: UP + "20200101_20201231.csv"}
CSV = FILES[2017]
WIN_START = "2017-08-01T00:00:00Z"     # year=None: the 2017 pre-registration
WIN_END = "2017-12-20T00:00:00Z"


def load_theta(year=None):
    """year=None: the 2017 file alone, window 2017-08-01 to 2017-12-20. year=Y (2018 to 2020): files Y-1 and Y with duplicate job IDs removed,
    window Y-04-01 to Y-07-01 (2018 to 2020 pre-registration)."""
    cols = ["COBALT_JOBID", "USERNAME_GENID", "QUEUED_TIMESTAMP", "START_TIMESTAMP", "END_TIMESTAMP", "WALLTIME_SECONDS", "RUNTIME_SECONDS", "NODES_REQUESTED"]
    if year is None:
        raw = pd.read_csv(CSV, usecols=cols)
        ws, we = WIN_START, WIN_END
    else:
        raw = pd.concat([pd.read_csv(FILES[y], usecols=cols) for y in (year - 1, year)], ignore_index=True).drop_duplicates("COBALT_JOBID", keep="last")
        ws, we = f"{year}-04-01T00:00:00Z", f"{year}-07-01T00:00:00Z"
    for c in ["QUEUED_TIMESTAMP", "START_TIMESTAMP", "END_TIMESTAMP"]:
        raw[c] = pd.to_datetime(raw[c], utc=True)
    raw = raw.sort_values(["QUEUED_TIMESTAMP", "COBALT_JOBID"], kind="stable").reset_index(drop=True)
    u = lambda s: s.astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
    d = {"usr": raw["USERNAME_GENID"].values, "ts": u(raw["QUEUED_TIMESTAMP"]).values.astype(np.int64),
         "sd": u(raw["START_TIMESTAMP"]).values.astype(np.int64), "ed": u(raw["END_TIMESTAMP"]).values.astype(np.int64),
         "dur": np.maximum(raw["RUNTIME_SECONDS"].values.astype(np.float64), 1.0),
         "lim": raw["WALLTIME_SECONDS"].values.astype(np.float64), "nn": np.maximum(raw["NODES_REQUESTED"].values.astype(np.float64), 1.0),
         "job_id": raw["COBALT_JOBID"].values}
    order = np.lexsort((np.arange(len(raw)), d["usr"]))
    gap = np.zeros(len(raw))
    us, ts = d["usr"][order], d["ts"][order]
    same = us[1:] == us[:-1]
    g = np.zeros(len(raw)); g[1:] = np.where(same, ts[1:] - ts[:-1], 0)
    gap[order] = np.maximum(g, 0)
    d["gap"] = gap
    t0 = int(pd.Timestamp(ws).timestamp()); t1 = int(pd.Timestamp(we).timestamp())
    d["win_rows"] = np.where((d["ts"] >= t0) & (d["ts"] < t1))[0]
    d["t0"], d["t1"] = t0, t1
    return d

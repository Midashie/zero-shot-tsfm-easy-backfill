"""PM100 loader. Table sorted by (submit time, job id); every array below is aligned to that order.
user = user_id; submit = submit_time; start = start_time; end = end_time; nodes = num_nodes_alloc;
runtime = run_time seconds floored at 1; declared limit = time_limit (minutes) * 60."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, pandas as pd

PQ = _IN + "/ZeroShotScheduleLocal/job_table.parquet"
WIN_START = "2020-07-01T00:00:00Z"
WIN_END = "2020-07-15T00:00:00Z"


def load_pm():
    raw = pd.read_parquet(PQ, columns=["job_id", "user_id", "submit_time", "start_time", "end_time", "run_time", "time_limit", "num_nodes_alloc"])
    raw = raw.sort_values(["submit_time", "job_id"], kind="stable").reset_index(drop=True)
    u = lambda s: s.astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
    d = {"usr": raw["user_id"].values, "ts": u(raw["submit_time"]).values.astype(np.int64), "sd": u(raw["start_time"]).values.astype(np.int64),
         "ed": u(raw["end_time"]).values.astype(np.int64), "dur": np.maximum(raw["run_time"].values.astype(np.float64), 1.0),
         "lim": raw["time_limit"].values.astype(np.float64) * 60.0, "nn": np.maximum(raw["num_nodes_alloc"].values.astype(np.float64), 1.0),
         "job_id": raw["job_id"].values}
    # gap: seconds since the same user's previous submission (0 for the first)
    order = np.lexsort((np.arange(len(raw)), d["usr"]))
    gap = np.zeros(len(raw))
    us, ts = d["usr"][order], d["ts"][order]
    same = us[1:] == us[:-1]
    g = np.zeros(len(raw)); g[1:] = np.where(same, ts[1:] - ts[:-1], 0)
    gap[order] = np.maximum(g, 0)
    d["gap"] = gap
    t0 = int(pd.Timestamp(WIN_START).timestamp()); t1 = int(pd.Timestamp(WIN_END).timestamp())
    d["win_rows"] = np.where((d["ts"] >= t0) & (d["ts"] < t1))[0]
    d["t0"], d["t1"] = t0, t1
    return d

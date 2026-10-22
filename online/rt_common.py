import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

OUT = _OUT
FD_PQ = _IN + "/ZeroShotSchedule/21_04.parquet"


def get_trace(trace):
    """Row-aligned arrays for the whole table plus the replay window, from the verified recorded-time contexts."""
    Z = np.load(f"{OUT}/{ {'fdata': 'taskb', 'pm100': 'pm100'}.get(trace, trace) }/contexts_c96.npz")
    if trace == "fdata":
        raw = pd.read_parquet(FD_PQ, columns=["usr", "edt"])
        usr = raw["usr"].values
        ed = pd.to_datetime(raw["edt"], utc=True).astype("datetime64[ns, UTC]").astype(np.int64).values // 10 ** 9
    elif trace.startswith("theta"):
        from theta_common import load_theta
        d = load_theta(int(trace[5:]) if trace[5:] else None); usr, ed = d["usr"], d["ed"]
    else:
        from pm_common import load_pm
        d = load_pm(); usr, ed = d["usr"], d["ed"]
    assert len(usr) == len(Z["ts"])
    return Z, usr, ed.astype(np.int64)

"""Shared window builder for Task A. Every window comes from the frozen manifest CSVs.
Context = the first 168 bins of the 192-bin window (log1p bytes); the decision is at bin 168; the target is bin 168 + h - 1.
Extended contexts (for TTM 180 and 512) take extra bins from BEFORE the window start, so they are still earlier than the decision.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import os
import numpy as np, pandas as pd

NPZ = _IN + "/ZeroShotSchedule/derived/cesnet_hourly.npz"
MAN = _OUT + "/manifest"
SPLIT, HMAX, N = 168, 24, 192
SETS = {"primary": "cesnet_primary_windows.csv", "recent": "cesnet_recent_windows.csv", "calibration": "cesnet_calibration_windows.csv"}


def load_set(name):
    z = np.load(NPZ)
    r, v = z["r"], z["valid"]
    df = pd.read_csv(os.path.join(MAN, SETS[name]))
    sid = df["series_row"].values.astype(int)
    st = df["start_bin"].values.astype(int)
    Z = np.stack([r[i, s:s + N] for i, s in zip(sid, st)]).astype(np.float32)
    assert all(v[i, s:s + N].min() > 0 for i, s in zip(sid, st))
    return {"r": r, "v": v, "sid": sid, "st": st, "Z": Z, "n": len(df)}


def extended_context(d, L):
    """Last L bins before the decision bin, with an observed mask (True where the bin is valid). Never reads bin >= decision."""
    r, v, sid, st = d["r"], d["v"], d["sid"], d["st"]
    X = np.zeros((d["n"], L), np.float32)
    M = np.zeros((d["n"], L), bool)
    for k, (i, s) in enumerate(zip(sid, st)):
        dec = s + SPLIT
        lo = dec - L
        assert lo >= 0 and dec <= r.shape[1]
        X[k] = r[i, lo:dec]
        M[k] = v[i, lo:dec] > 0
    return X, M

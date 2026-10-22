"""SafeWall-style baseline (reconstructed from arXiv 2610.26577 text). Features, training and buffer; pre-registered in the doc."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import os, pickle, json
import numpy as np

OUTD = _OUT + "/swb"
C = 96; TAU = 0.10; KAPPA = 10.0
NAMES = ["own_lim", "own_nodes", "lag1", "lag2", "lag_mean", "acc_mean", "acc_max", "acc_mean10", "acc_max10",
         "run_mean", "run_max", "run_mean10", "run_max10", "run_p25", "n_ctx"]


def feat_one(y, w, own_lim, own_nn):
    a = np.minimum(w / y, y / w)
    return [own_lim, own_nn, y[-1], y[-2], 0.5 * (y[-1] + y[-2]), a.mean(), a.max(), a[-10:].mean(), a[-10:].max(),
            y.mean(), y.max(), y[-10:].mean(), y[-10:].max(), float(np.percentile(y, 25)), float(len(y))]


def feats(ctx_list, own_rows, dur, lim, nn):
    """ctx_list: list of arrays of table rows (oldest to newest, at least 3 each); own_rows: the jobs' own table rows."""
    return np.array([feat_one(dur[r], lim[r], lim[o], nn[o]) for r, o in zip(ctx_list, own_rows)], dtype=np.float64)


def buffers(user, resid):
    """Global and per-user buffers from positive training residuals (paper Eq. 8 and 9). Returns (b_g, {user: (n_u, b_u)})."""
    pos = resid > 0
    bg = float(np.quantile(resid[pos], 1 - TAU))
    out = {}
    order = np.argsort(user[pos], kind="stable"); us = user[pos][order]; rs = resid[pos][order]
    cuts = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        out[us[a]] = (int(b - a), float(np.quantile(rs[a:b], 1 - TAU)))
    return bg, out


def user_buffer(u, bg, per):
    n, bu = per.get(u, (0, 0.0))
    return (n / (n + KAPPA)) * bu + (KAPPA / (n + KAPPA)) * bg


def load_swb(trace):
    import xgboost as xgb
    m = xgb.XGBRegressor(); m.load_model(f"{OUTD}/{trace}_model.json")
    P = pickle.load(open(f"{OUTD}/{trace}_buf.pkl", "rb"))
    return m, P["bg"], P["per"]

"""Tuned classical arms for Task A. Everything is chosen on the 3,000 calibration windows only (they end before the evaluation period).
The primary and recent windows are never read here.

Selection score for a point forecaster: over-provisioning (percent of demand) at a matched violation rate, averaged over the 20% and 30%
operating points, using 2-fold cross-conformal on the calibration windows (residuals from one half cap the other half). Lower is better.
Candidates:
  EWMA        flat forecast, smoothing weight a in {0.001 ... 0.5}, the grid extended after the first try put the best value at its edge
  Seasonal    same-hour-of-day profile over the last K days (mean or median), optional level correction from the last 3 hours
  Ridge       pooled ridge regression on 24 recent bins, 7 same-hour lags, an EWMA, the 168-bin mean and the same hour one week earlier;
              one model per reported horizon (1, 4, 24); penalty chosen by the same score
All values are log1p bytes, as in analyze.py. Contexts are the first 168 bins of each window, so no input is at or after the decision bin.
"""
import numpy as np

SPLIT, HMAX = 168, 24
CLV = np.round(np.arange(0.05, 0.9901, 0.01), 2)
HS = (1, 4, 24)


def ewma(Z, a):
    e = Z[:, 0].copy()
    for k in range(1, SPLIT):
        e = a * Z[:, k] + (1 - a) * e
    return e


def seasonal(Z, K, stat="mean", beta=0.0):
    """Point forecast (n, 24). Target bin for step h is SPLIT+h-1; lag 24k bins back stays inside the context for k>=1."""
    out = np.zeros((len(Z), HMAX), np.float32)
    f = np.mean if stat == "mean" else np.median
    for h in range(1, HMAX + 1):
        out[:, h - 1] = f(np.stack([Z[:, SPLIT + h - 1 - 24 * k] for k in range(1, K + 1)], 1), axis=1)
    if beta:
        last3 = Z[:, SPLIT - 3:SPLIT].mean(1)
        al = np.stack([f(np.stack([Z[:, SPLIT - j - 24 * k] for k in range(1, K + 1)], 1), axis=1) for j in (1, 2, 3)], 1).mean(1)
        out = out + beta * (last3 - al)[:, None]
    return out


def feats(Z, h):
    cols = [Z[:, SPLIT - 24:SPLIT]]
    cols.append(np.stack([Z[:, SPLIT + h - 1 - 24 * k] for k in range(1, 8) if SPLIT + h - 1 - 24 * k >= 0], 1))
    cols.append(ewma(Z, 0.1)[:, None])
    cols.append(Z[:, :SPLIT].mean(1, keepdims=True))
    cols.append(Z[:, h - 1:h])                      # same hour one week earlier
    return np.concatenate(cols, 1).astype(np.float64)


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def _over_viol(cap_log, y_log):
    cap, act = np.expm1(cap_log), np.expm1(y_log)
    return np.maximum(cap - act, 0), (act > cap).astype(float), act


def cv_score(pred, y, folds, targets=(20.0, 30.0)):
    """Mean over-provisioning at matched violation, 2-fold cross-conformal. pred: (n,) point forecasts made without using y."""
    n = len(y)
    over, viol = [], []
    for t in CLV:
        o = v = a = 0.0
        for f in (0, 1):
            te, tr = folds == f, folds != f
            res = np.sort(y[tr] - pred[tr])
            k = min(int(np.ceil((tr.sum() + 1) * t)) - 1, tr.sum() - 1)
            oo, vv, aa = _over_viol(pred[te] + res[k], y[te])
            o += oo.sum(); v += vv.sum(); a += aa.sum()
        over.append(o / a * 100); viol.append(v / n * 100)
    over, viol = np.array(over), np.array(viol)
    order = np.argsort(viol)
    xs, ys = viol[order], over[order]
    vals = []
    for T in targets:
        vals.append(np.interp(T, xs, ys) if xs[0] <= T <= xs[-1] else np.inf)
    return float(np.mean(vals)), [float(x) for x in vals]


def conformal_caps(P_cal_h1, y_cal, P_eval_h1):
    res = np.sort(y_cal - P_cal_h1)
    n = len(res)
    caps = [res[min(int(np.ceil((n + 1) * t)) - 1, n - 1)] for t in CLV]
    return P_eval_h1[:, None] + np.array(caps)[None, :]


def build_tuned(Zcal, Zeval):
    """Zcal: calibration contexts+targets (n,192). Zeval: dict set -> (n,192). Returns (arms dict, report dict)."""
    ycal = Zcal[:, SPLIT]
    rng = np.random.default_rng(7)
    folds = (rng.permutation(len(ycal)) % 2).astype(int)
    rep = {"candidates": {}}
    # ---- EWMA
    sc_e = {}
    for a in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        s, parts = cv_score(ewma(Zcal, a), ycal, folds)
        sc_e[a] = s; rep["candidates"][f"EWMA a={a}"] = {"score": s, "over_at_20_30": parts}
    a_best = min(sc_e, key=sc_e.get)
    # ---- seasonal
    sc_s = {}
    for K in (1, 2, 3, 5, 7):
        for stat in ("mean", "median"):
            if K == 1 and stat == "median":
                continue
            for beta in (0.0, 0.25, 0.5, 1.0):
                P = seasonal(Zcal, K, stat, beta)[:, 0]
                s, parts = cv_score(P, ycal, folds)
                key = (K, stat, beta); sc_s[key] = s
                rep["candidates"][f"Seasonal K={K} {stat} beta={beta}"] = {"score": s, "over_at_20_30": parts}
    s_best = min(sc_s, key=sc_s.get)
    # ---- ridge, one model per horizon in HS; cross-fitted calibration predictions
    Xc = {h: feats(Zcal, h) for h in HS}
    yh = {h: Zcal[:, SPLIT + h - 1].astype(np.float64) for h in HS}
    sc_r = {}
    for lam in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        pred = np.zeros(len(ycal))
        for f in (0, 1):
            m = ridge_fit(Xc[1][folds != f], yh[1][folds != f], lam)
            pred[folds == f] = m(Xc[1][folds == f])
        s, parts = cv_score(pred, ycal, folds)
        sc_r[lam] = s; rep["candidates"][f"Ridge lam={lam}"] = {"score": s, "over_at_20_30": parts}
    lam_best = min(sc_r, key=sc_r.get)
    rep["chosen"] = {"EWMA": {"a": a_best, "score": sc_e[a_best]},
                     "Seasonal": {"K": s_best[0], "stat": s_best[1], "beta": s_best[2], "score": sc_s[s_best]},
                     "Ridge": {"lambda": lam_best, "score": sc_r[lam_best]},
                     "untuned_reference_EWMA_a=0.3_score": sc_e[0.3]}
    # ---- cross-fitted calibration predictions for ridge (5-fold, h=1 residuals used for conformal)
    k5 = (np.random.default_rng(11).permutation(len(ycal)) % 5).astype(int)
    pcal_r = np.zeros(len(ycal))
    for f in range(5):
        m = ridge_fit(Xc[1][k5 != f], yh[1][k5 != f], lam_best)
        pcal_r[k5 == f] = m(Xc[1][k5 == f])
    models = {h: ridge_fit(Xc[h], yh[h], lam_best) for h in HS}
    arms = {}
    pcal = {"Constant cap+conformal": np.full(len(ycal), ycal.mean()), "EWMA-tuned+conformal": ewma(Zcal, a_best), "Seasonal-tuned+conformal": seasonal(Zcal, *s_best)[:, 0], "Ridge+conformal": pcal_r}
    for nm in pcal:
        arms[nm] = {"levels": CLV, "caps": {}, "point": {}, "q": {}, "native": False}
    for s_, Z in Zeval.items():
        pe = {"Constant cap+conformal": np.full((len(Z), HMAX), ycal.mean(), np.float32), "EWMA-tuned+conformal": np.repeat(ewma(Z, a_best)[:, None], HMAX, 1), "Seasonal-tuned+conformal": seasonal(Z, *s_best)}
        R = np.full((len(Z), HMAX), np.nan, np.float32)
        for h in HS:
            R[:, h - 1] = models[h](feats(Z, h))
        pe["Ridge+conformal"] = R
        for nm in pcal:
            arms[nm]["caps"][s_] = conformal_caps(pcal[nm], ycal, pe[nm][:, 0])
            arms[nm]["point"][s_] = pe[nm]
            arms[nm]["q"][s_] = None
    return arms, rep

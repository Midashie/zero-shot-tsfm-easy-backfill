"""Task A analysis. Reads forecasts from run_models.py, builds every arm, scores decisions and forecasts,
runs paired bootstrap tests and rank agreement.

Usage: python3 -I analyze_tuned.py FORECAST_DIR OUT_DIR [B]
Pre-set choices (fixed before any full-run number existed):
  - operating points for tests: violation target 20% and 30% (interpolated along each arm's tau sweep, no extrapolation)
  - bootstrap: B=10,000 resamples of windows, shared across arms; Holm within each family
  - family 1: five zero-shot TSFMs vs the best classical arm at each operating point (10 tests)
  - family 2: all pairs among the five TSFMs at each operating point (20 tests)
  - TTM primary arm is the 90-hour model; the 180-hour and 512-hour models are sensitivity arms, outside both families
  - classical arms and TTM carry quantiles through split conformal on the calibration windows (log1p space, h=1)
"""
import sys, os, json, itertools
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_a import load_set, SPLIT, HMAX

fdir, out = sys.argv[1], sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(12345)

LV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
# conformal arms can serve any level, so they get a dense grid; native-quantile arms keep their own levels.
# (Added after the first pass: coarse linear interpolation overstated the cost of the conformal arms, whose cost curve is steep.)
CLV = np.round(np.arange(0.05, 0.9901, 0.01), 2)
HS = (1, 4, 24)
TARGETS = (20.0, 30.0, 10.0)           # 20 and 30 carry tests; 10 is descriptive and often out of range
TEST_V = (20.0, 30.0)
TSFM = ["Chronos-Bolt-small", "Chronos-2", "TimesFM-2.5", "Moirai-2.0", "TTM-90"]
CLASSICAL = ["persistence+conformal", "EWMA+conformal", "lastweek+conformal", "EWMA-tuned+conformal", "Seasonal-tuned+conformal", "Ridge+conformal", "Constant cap+conformal"]
CLASSICAL_SA = [c for c in CLASSICAL if not c.startswith("Constant")]   # series-aware classical arms
SENS = ["TTM-180", "TTM-512"]

D = {s: load_set(s) for s in ("primary", "recent", "calibration")}
r, v = D["primary"]["r"], D["primary"]["v"]
cut = int(r.shape[1] * 0.8)
scale = {i: float(np.mean(np.abs(np.diff(r[i, :cut][v[i, :cut] > 0])))) for i in range(r.shape[0])}
med = np.median([x for x in scale.values() if np.isfinite(x) and x > 0])


def sc(sid):
    return np.array([scale[i] if np.isfinite(scale[i]) and scale[i] >= 1e-3 * med else med for i in sid])


def classical_point(Z):
    last = Z[:, SPLIT - 1]
    ew = Z[:, 0].copy()
    for k in range(1, SPLIT):
        ew = 0.3 * Z[:, k] + 0.7 * ew
    P = {"persistence": np.repeat(last[:, None], HMAX, 1), "EWMA": np.repeat(ew[:, None], HMAX, 1)}
    P["lastweek"] = np.stack([Z[:, SPLIT + h - 1 - 168] if SPLIT + h - 1 - 168 >= 0 else Z[:, 0] for h in range(1, HMAX + 1)], 1)
    return P


def conformal_caps(P_cal_h1, y_cal, P_eval_h1):
    res = np.sort(y_cal - P_cal_h1)
    n = len(res)
    caps = []
    for t in CLV:
        k = int(np.ceil((n + 1) * t)) - 1
        caps.append(res[min(k, n - 1)])
    return P_eval_h1[:, None] + np.array(caps)[None, :]


# ---------------- build arms ----------------
arms = {}      # name -> dict(levels, caps{set}, point{set} (n,24), q9{set} or None, native bool)
ycal = D["calibration"]["Z"][:, SPLIT]
Pcal = classical_point(D["calibration"]["Z"])
for s_ in ("primary", "recent"):
    Pe = classical_point(D[s_]["Z"])
    for nm in ("persistence", "EWMA", "lastweek"):
        a = arms.setdefault(nm + "+conformal", {"levels": CLV, "caps": {}, "point": {}, "q": {}, "native": False})
        a["caps"][s_] = conformal_caps(Pcal[nm][:, 0], ycal, Pe[nm][:, 0])
        a["point"][s_] = Pe[nm]
        a["q"][s_] = None


def load(name, s_):
    return np.load(os.path.join(fdir, f"F_{name}_{s_}.npz"))


native = {"Chronos-Bolt-small": ("bolt", 9), "Chronos-2": ("chronos2", 11), "TimesFM-2.5": ("timesfm", 9), "Moirai-2.0": ("moirai2", 9)}
for nm, (fn, K) in native.items():
    a = {"levels": LV[:K], "caps": {}, "point": {}, "q": {}, "native": True}
    for s_ in ("primary", "recent"):
        Q = load(fn, s_)["Q"][:, :, :K]
        a["caps"][s_] = Q[:, 0, :]
        a["point"][s_] = Q[:, :, 4]
        a["q"][s_] = Q[:, :, :9]
    arms[nm] = a
for nm, fn in (("TTM-90", "ttm90"), ("TTM-180", "ttm180"), ("TTM-512", "ttm512")):
    Pc = np.load(os.path.join(fdir, f"F_{fn}_calibration.npz"))["P"]
    a = {"levels": CLV, "caps": {}, "point": {}, "q": {}, "native": False}
    for s_ in ("primary", "recent"):
        P = np.load(os.path.join(fdir, f"F_{fn}_{s_}.npz"))["P"]
        a["caps"][s_] = conformal_caps(Pc[:, 0], ycal, P[:, 0])
        a["point"][s_] = P
        a["q"][s_] = None
    arms[nm] = a

# tuned classical arms (chosen on calibration windows only)
from tune_a import build_tuned
_t, TUNE_REP = build_tuned(D["calibration"]["Z"], {k: D[k]["Z"] for k in ("primary", "recent")})
arms.update(_t)
json.dump(TUNE_REP, open(os.path.join(out, "tuning_report.json"), "w"), indent=1)
print("TUNED", TUNE_REP["chosen"], flush=True)

# operator dots
dots = {}
for s_ in ("primary", "recent"):
    Z = D[s_]["Z"]
    last = Z[:, SPLIT - 1]
    wk = Z[:, 0]
    ew = classical_point(Z)["EWMA"][:, 0]
    dots[s_] = {"rule:lastweek x1.0": wk, "rule:lastweek x1.2": np.log1p(1.2 * np.expm1(wk)),
                "rule:EWMA x1.2": np.log1p(1.2 * np.expm1(ew)), "rule:last value x1.0": last,
                "rule:last value x1.2": np.log1p(1.2 * np.expm1(last))}


def prov(cap_log, y_log):
    cap, act = np.expm1(cap_log), np.expm1(y_log)
    return np.maximum(cap - act, 0), (act > cap).astype(float), act


# ---------------- bootstrap machinery ----------------
def counts(n, B_):
    for i in range(0, B_, 2000):
        m = min(2000, B_ - i)
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float32)


def interp_at(viol, over, V):
    """viol, over: (K,) for one resample; no extrapolation."""
    o = np.argsort(viol)
    x, y = viol[o], over[o]
    if V < x[0] or V > x[-1]:
        return np.nan
    return float(np.interp(V, x, y))


R = {"design": {"B": B, "operating_points_violation_pct": list(TARGETS), "test_points": list(TEST_V),
                "families": "see analyze.py header", "windows_primary": D["primary"]["n"]},
     "arms": {}, "tests": {}, "tests_series_aware": {}, "rank": {}, "calibration": {}, "forecast": {}, "operator_dots": {}}

# ---------- point estimates and bootstrap for provisioning on each window set ----------
for s_ in ("primary", "recent"):
    y = D[s_]["Z"][:, SPLIT]
    names = list(arms)
    # stack per-window quantities: for each arm and tau -> over_num, viol, plus act
    cols, keys = [], []
    for nm in names:
        for j, t in enumerate(arms[nm]["levels"]):
            o_, v_, a_ = prov(arms[nm]["caps"][s_][:, j], y)
            cols.append(np.stack([o_, v_, a_], 1)); keys.append((nm, j))
    for nm, cap in dots[s_].items():
        o_, v_, a_ = prov(cap, y)
        cols.append(np.stack([o_, v_, a_], 1)); keys.append((nm, -1))
    M = np.stack(cols, 0)                          # (C, n, 3)
    Cn = M.shape[0]
    bo = np.zeros((B, Cn), np.float32); bv = np.zeros((B, Cn), np.float32)
    i0 = 0
    for Cb in counts(len(y), B):
        m = Cb.shape[0]
        num = Cb @ M[:, :, 0].T; vi = Cb @ M[:, :, 1].T; ac = Cb @ M[:, :, 2].T
        bo[i0:i0 + m] = num / ac * 100; bv[i0:i0 + m] = vi / len(y) * 100
        i0 += m
    point_o = M[:, :, 0].sum(1) / M[:, :, 2].sum(1) * 100
    point_v = M[:, :, 1].mean(1) * 100
    idx = {}
    for c, (nm, j) in enumerate(keys):
        idx.setdefault(nm, []).append((j, c))
    store = {}
    for nm, lst in idx.items():
        lst.sort()
        cs = [c for _, c in lst]
        if nm.startswith("rule:"):
            R["operator_dots"].setdefault(s_, {})[nm] = {
                "over_pct": float(point_o[cs[0]]), "over_ci95": np.percentile(bo[:, cs[0]], [2.5, 97.5]).tolist(),
                "viol_pct": float(point_v[cs[0]]), "viol_ci95": np.percentile(bv[:, cs[0]], [2.5, 97.5]).tolist()}
            continue
        levels = arms[nm]["levels"]
        rows = []
        for (j, c) in lst:
            rows.append({"tau": float(levels[j]), "over_pct": float(point_o[c]), "over_ci95": np.percentile(bo[:, c], [2.5, 97.5]).tolist(),
                         "viol_pct": float(point_v[c]), "viol_ci95": np.percentile(bv[:, c], [2.5, 97.5]).tolist()})
        R["arms"].setdefault(nm, {})[s_] = {"sweep": rows}
        store[nm] = (cs, levels)
    # matched-violation frontier, point and bootstrap
    front = {}
    for nm, (cs, levels) in store.items():
        pts = {}
        for V in TARGETS:
            pe = interp_at(point_v[cs], point_o[cs], V)
            bs = np.array([interp_at(bv[b, cs], bo[b, cs], V) for b in range(B)])
            ok = ~np.isnan(bs)
            ci = np.percentile(bs[ok], [2.5, 97.5]).tolist() if ok.mean() > 0.95 else None
            pts[str(V)] = {"over_at_violation_pct": None if np.isnan(pe) else pe, "ci95": ci, "share_resamples_in_range": float(ok.mean())}
        front[nm] = {"point": pts, "boot": {str(V): np.array([interp_at(bv[b, cs], bo[b, cs], V) for b in range(B)]) for V in TARGETS}}
        R["arms"][nm][s_]["matched_violation"] = pts
    if s_ == "primary":
        FRONT = front

# ---------- paired tests on the primary windows ----------
def pboot(a, b):
    d = a - b
    ok = ~np.isnan(d)
    if ok.mean() < 0.95:
        return None
    d = d[ok]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff_over_pct": float(np.mean(d)), "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_boot": float(max(p, 1.0 / (len(d) + 1)))}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


for V in TEST_V:
    best = None
    for nm in CLASSICAL:
        pe = FRONT[nm]["point"][str(V)]["over_at_violation_pct"]
        if pe is not None and (best is None or pe < best[1]):
            best = (nm, pe)
    R["tests"][str(V)] = {"best_classical": best[0] if best else None, "vs_best_classical": {}, "pairwise": {}}
    fam1 = []
    for nm in TSFM:
        res = pboot(FRONT[nm]["boot"][str(V)], FRONT[best[0]]["boot"][str(V)]) if best else None
        fam1.append((nm, res))
    R["tests"][str(V)]["vs_best_classical"] = {k: v_ for k, v_ in fam1}
    R["tests"][str(V)]["pairwise"] = {f"{a} minus {b}": pboot(FRONT[a]["boot"][str(V)], FRONT[b]["boot"][str(V)]) for a, b in itertools.combinations(TSFM, 2)}
fam1_all = [(V, k, x) for V in TEST_V for k, x in R["tests"][str(V)]["vs_best_classical"].items() if x]
adj = holm([x["p_boot"] for _, _, x in fam1_all]) if fam1_all else []
for (V, k, x), a in zip(fam1_all, adj):
    x["p_holm_family1"] = float(a)
fam2_all = [(V, k, x) for V in TEST_V for k, x in R["tests"][str(V)]["pairwise"].items() if x]
adj = holm([x["p_boot"] for _, _, x in fam2_all]) if fam2_all else []
for (V, k, x), a in zip(fam2_all, adj):
    x["p_holm_family2"] = float(a)

# same tests against the best series-aware classical arm (post hoc: the constant-cap reference is excluded)
for V in TEST_V:
    best = None
    for nm in CLASSICAL_SA:
        pe = FRONT[nm]["point"][str(V)]["over_at_violation_pct"]
        if pe is not None and (best is None or pe < best[1]):
            best = (nm, pe)
    R["tests_series_aware"][str(V)] = {"best_classical": best[0] if best else None, "vs_best_classical": {}, "pairwise": {}}
    fam1 = []
    for nm in TSFM:
        res = pboot(FRONT[nm]["boot"][str(V)], FRONT[best[0]]["boot"][str(V)]) if best else None
        fam1.append((nm, res))
    R["tests_series_aware"][str(V)]["vs_best_classical"] = {k: v_ for k, v_ in fam1}
    R["tests_series_aware"][str(V)]["pairwise"] = {f"{a} minus {b}": pboot(FRONT[a]["boot"][str(V)], FRONT[b]["boot"][str(V)]) for a, b in itertools.combinations(TSFM, 2)}
fam1_all = [(V, k, x) for V in TEST_V for k, x in R["tests_series_aware"][str(V)]["vs_best_classical"].items() if x]
adj = holm([x["p_boot"] for _, _, x in fam1_all]) if fam1_all else []
for (V, k, x), a in zip(fam1_all, adj):
    x["p_holm_family1"] = float(a)
fam2_all = [(V, k, x) for V in TEST_V for k, x in R["tests_series_aware"][str(V)]["pairwise"].items() if x]
adj = holm([x["p_boot"] for _, _, x in fam2_all]) if fam2_all else []
for (V, k, x), a in zip(fam2_all, adj):
    x["p_holm_family2"] = float(a)

# ---------- forecast metrics, calibration ----------
def pin(y, q, t):
    e = y - q
    return np.maximum(t * e, (t - 1) * e)


for s_ in ("primary", "recent"):
    Z = D[s_]["Z"]; sid = D[s_]["sid"]; scl = sc(sid)
    R["forecast"][s_] = {}
    for nm, a in arms.items():
        R["forecast"][s_][nm] = {}
        for h in HS:
            y = Z[:, SPLIT + h - 1]
            P = a["point"][s_][:, h - 1]
            per = {}
            for sd in np.unique(sid):
                m_ = sid == sd
                per[sd] = np.abs(P[m_] - y[m_]).mean() / scl[m_][0]
            ent = {"MASE": float(np.mean(list(per.values())))}
            if a["q"][s_] is not None:
                q9 = a["q"][s_][:, h - 1, :]
                ent["CRPS9"] = float(np.mean([2 * pin(y, q9[:, j], t).mean() for j, t in enumerate(LV[:9])]))
            else:
                ent["CRPS9"] = float(np.abs(P - y).mean())      # point forecast: CRPS equals MAE
                ent["CRPS9_note"] = "point forecast"
            if a["native"] and len(a["levels"]) == 11:
                ent["pinball99"] = float(pin(y, a["caps"][s_][:, 10] if h == 1 else a["q"][s_][:, h - 1, 8], 0.99).mean()) if h == 1 else None
            R["forecast"][s_][nm][f"h{h}"] = ent
    y = Z[:, SPLIT]
    R["calibration"][s_] = {}
    for nm, a in arms.items():
        R["calibration"][s_][nm] = {str(float(t)): float((y <= a["caps"][s_][:, j]).mean()) for j, t in enumerate(a["levels"])}

# ---------- rank agreement on the primary windows ----------
main = TSFM + CLASSICAL
y = D["primary"]["Z"][:, SPLIT]; sid = D["primary"]["sid"]; scl = sc(sid)
Vr = 20.0
mase_w, crps_w = {}, {}
for nm in main:
    a = arms[nm]
    mase_w[nm] = np.abs(a["point"]["primary"][:, 0] - y) / scl
    if a["q"]["primary"] is not None:
        q9 = a["q"]["primary"][:, 0, :]
        crps_w[nm] = np.mean([2 * pin(y, q9[:, j], t) for j, t in enumerate(LV[:9])], 0)
    else:
        crps_w[nm] = np.abs(a["point"]["primary"][:, 0] - y)
over_pt = {nm: FRONT[nm]["point"][str(Vr)]["over_at_violation_pct"] for nm in main}
usable = [nm for nm in main if over_pt[nm] is not None]
R["rank"]["arms_in_ranking"] = usable
R["rank"]["decision_metric"] = f"over-provisioning at {Vr}% violation (lower is better)"
mase_pt = {nm: float(mase_w[nm].mean()) for nm in usable}
crps_pt = {nm: float(crps_w[nm].mean()) for nm in usable}
R["rank"]["point"] = {"over_at_20": {k: over_pt[k] for k in usable}, "MASE_h1_window_mean": mase_pt, "CRPS9_h1": crps_pt}
tau_m = kendalltau([over_pt[k] for k in usable], [mase_pt[k] for k in usable])[0]
tau_c = kendalltau([over_pt[k] for k in usable], [crps_pt[k] for k in usable])[0]
R["rank"]["kendall_point"] = {"decision_vs_MASE": float(tau_m), "decision_vs_CRPS": float(tau_c)}
# bootstrap kendall and pairwise inversions
Cb_all = np.concatenate(list(counts(len(y), B)), 0)
mb = {k: (Cb_all @ mase_w[k]) / len(y) for k in usable}
cb = {k: (Cb_all @ crps_w[k]) / len(y) for k in usable}
ob = {k: FRONT[k]["boot"][str(Vr)] for k in usable}
tm, tc = [], []
for b in range(B):
    o_ = [ob[k][b] for k in usable]
    if np.isnan(o_).any():
        continue
    tm.append(kendalltau(o_, [mb[k][b] for k in usable])[0]); tc.append(kendalltau(o_, [cb[k][b] for k in usable])[0])
R["rank"]["kendall_boot"] = {"decision_vs_MASE_ci95": np.nanpercentile(tm, [2.5, 97.5]).tolist(), "decision_vs_CRPS_ci95": np.nanpercentile(tc, [2.5, 97.5]).tolist(),
                             "resamples_used": len(tm)}
inv = []
for a_, b_ in itertools.combinations(usable, 2):
    do = ob[a_] - ob[b_]; dm = mb[a_] - mb[b_]; dc = cb[a_] - cb[b_]
    ok = ~np.isnan(do)
    ci = lambda x: np.percentile(x[ok], [2.5, 97.5])
    co, cm, cc = ci(do), ci(dm), ci(dc)
    sig = lambda c: (c[0] > 0) or (c[1] < 0)
    for nm_, c_, d_ in (("MASE", cm, dm), ("CRPS", cc, dc)):
        if sig(co) and sig(c_) and np.sign(co.mean()) != np.sign(c_.mean()):
            inv.append({"pair": f"{a_} vs {b_}", "forecast_metric": nm_, "decision_diff_over_pct_ci95": co.tolist(), "forecast_diff_ci95": c_.tolist()})
R["rank"]["inversions_with_ci_excluding_zero_on_both_metrics"] = inv

# ---------- latency ----------
lat = {}
for fn in ("bolt", "chronos2", "timesfm", "moirai2", "ttm90", "ttm180", "ttm512"):
    p = os.path.join(fdir, f"lat_{fn}.json")
    if os.path.exists(p):
        lat[fn] = json.load(open(p))
R["latency_cpu_2_threads"] = lat

with open(os.path.join(out, "task_a_full.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("ANALYSIS_DONE")

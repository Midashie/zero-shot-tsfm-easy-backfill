"""Task A, byte-weighted decision metric with paired bootstrap. Added after the constant-cap finding, so it is NOT pre-registered; stated in the doc.
Metric: over-provisioning (percent of demand, bytes) at matched byte shortfall, shortfall = sum(max(act-cap,0))/sum(act) in percent.
Operating points 5 and 10 percent. Interpolation along each arm's tau sweep, no extrapolation. 10,000 paired bootstrap resamples of windows.
Tests: five zero-shot models against the best classical arm (lowest point estimate on the primary windows; classical = every non-zero-shot arm, including the constant cap),
Holm over the 10 tests. Also against the best series-aware classical arm (constant cap excluded).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, json, itertools
src = open(_ROOT + "/taska/analyze_tuned.py").read().split("# operator dots")[0]
sys.argv = ["x", _OUT + "/taska_full", _OUT + "/taska_tuned", "10"]
exec(compile(src, "head", "exec"))
B = 10000
rng2 = np.random.default_rng(2026)
y = D["primary"]["Z"][:, SPLIT]; act = np.expm1(y).astype(np.float64); n = len(y)
names = [k for k in arms]
cols, keys = [], []
for nm in names:
    cap = np.expm1(arms[nm]["caps"]["primary"].astype(np.float64))
    for j in range(cap.shape[1]):
        cols.append(np.stack([np.maximum(cap[:, j] - act, 0), np.maximum(act - cap[:, j], 0)], 1)); keys.append((nm, j))
M = np.stack(cols, 0)                                     # (C, n, 2)
den = act.sum()
Cn = M.shape[0]
bo = np.zeros((B, Cn), np.float32); bs = np.zeros((B, Cn), np.float32)
i0 = 0
while i0 < B:
    m = min(1000, B - i0)
    cnt = rng2.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float64)
    d = cnt @ act
    bo[i0:i0 + m] = (cnt @ M[:, :, 0].T) / d[:, None] * 100
    bs[i0:i0 + m] = (cnt @ M[:, :, 1].T) / d[:, None] * 100
    i0 += m
po = M[:, :, 0].sum(1) / den * 100; ps = M[:, :, 1].sum(1) / den * 100
idx = {}
for c, (nm, j) in enumerate(keys): idx.setdefault(nm, []).append(c)
def interp(x, v, T):
    o = np.argsort(x); xs, vs = x[o], v[o]
    return float(np.interp(T, xs, vs)) if xs[0] <= T <= xs[-1] else np.nan
OPS = (5.0, 10.0)
res, boot = {}, {}
for nm, cs in idx.items():
    res[nm] = {}; boot[nm] = {}
    for T in OPS:
        pe = interp(ps[cs], po[cs], T)
        bb = np.array([interp(bs[b, cs], bo[b, cs], T) for b in range(B)])
        ok = ~np.isnan(bb)
        res[nm][str(T)] = {"point": None if np.isnan(pe) else pe, "ci95": np.percentile(bb[ok], [2.5, 97.5]).tolist() if ok.mean() > 0.95 else None, "share_in_range": float(ok.mean())}
        boot[nm][str(T)] = bb
def holm(ps_):
    o = np.argsort(ps_); m = len(ps_); adj = np.empty(m); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, min(1.0, (m - r_) * ps_[i])); adj[i] = run
    return adj
def tests(base_names):
    out, flat = {}, []
    for T in OPS:
        cands = [(nm, res[nm][str(T)]["point"]) for nm in base_names if res[nm][str(T)]["point"] is not None]
        best = min(cands, key=lambda x: x[1])[0]
        out[str(T)] = {"best": best, "vs": {}}
        for nm in TSFM:
            d = boot[nm][str(T)] - boot[best][str(T)]; ok = ~np.isnan(d)
            if ok.mean() < 0.95:
                out[str(T)]["vs"][nm] = None; continue
            d = d[ok]; p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (len(d) + 1))
            e = {"diff": float(d.mean()), "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p": float(p), "share_in_range": float(ok.mean())}
            out[str(T)]["vs"][nm] = e; flat.append(e)
    adj = holm([e["p"] for e in flat]) if flat else []
    for e, a in zip(flat, adj): e["p_holm"] = float(a)
    return out
CL_ALL = [k for k in CLASSICAL]
CL_SA = [k for k in CLASSICAL_SA]
R2 = {"operating_points_byte_shortfall_pct": OPS, "B": B, "arms": res, "tests_vs_best_classical": tests(CL_ALL), "tests_vs_best_series_aware": tests(CL_SA)}
json.dump(R2, open(_OUT + "/taska_tuned/bytes_metric.json", "w"), indent=1)
for nm in names:
    print(f"{nm:28s}", {T: (None if res[nm][T]['point'] is None else round(res[nm][T]['point'], 1), None if res[nm][T]['ci95'] is None else [round(x, 1) for x in res[nm][T]['ci95']]) for T in res[nm]})
for k in ("tests_vs_best_classical", "tests_vs_best_series_aware"):
    print(k)
    for T, t in R2[k].items():
        print(" T", T, "best", t["best"])
        for nm, e in t["vs"].items():
            print("   ", nm, None if e is None else (round(e["diff"], 1), [round(x, 1) for x in e["ci95"]], round(e["p_holm"], 3)))

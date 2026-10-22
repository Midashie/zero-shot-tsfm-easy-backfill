"""Compare recorded-time and replay-timeline contexts. Usage: python3 -I analyze_rt.py OUT_JSON [B]
Per setting (F-DATA 20K, F-DATA 23K, PM100 512): (1) mean slowdown, 95% interval over users, for each arm under both context rules;
(2) paired difference replay-timeline minus recorded-time per arm, same user resamples; (3) four models vs the best classical arm under each rule,
Holm inside a family of 4 per setting and rule. q = 0.5 primary."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json
import numpy as np
OUT = _OUT; B = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
rng = np.random.default_rng(4242)
MODELS = ["Chronos-Bolt-small", "Chronos-2", "Moirai-2.0", "TTM-90 + conformal"]
CL = ["user-elpl", "classical:Tsafrir last-2 (completion-causal)", "classical:history q90 (completion-causal)"]
SET = [("fdata_20000", f"{OUT}/taskb/replay_bs_20000.npz", f"{OUT}/rt_fdata_20000/replay_bs_20000.npz"),
       ("fdata_23000", f"{OUT}/taskb/replay_bs_23000.npz", f"{OUT}/rt_fdata_23000/replay_bs_23000.npz"),
       ("pm100_512", f"{OUT}/pm100/replay_bs_512.npz", f"{OUT}/rt_pm100_512/replay_bs_512.npz")]


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def load(p, arms):
    Z = np.load(p, allow_pickle=True); usr = Z["usr"]; uniq, inv = np.unique(usr, return_inverse=True)
    S = np.zeros((len(arms), len(uniq))); C = np.zeros_like(S)
    for a, nm in enumerate(arms):
        x = Z[nm]; m = ~np.isnan(x); np.add.at(S[a], inv[m], x[m].astype(np.float64)); np.add.at(C[a], inv[m], 1.0)
    return uniq, S, C


R = {"B": B, "settings": {}}
for sname, old_p, new_p in SET:
    arms = ["user-elpl", "oracle"] + CL[1:] + [f"{m} q{q}" for m in MODELS for q in (0.5, 0.6, 0.7, 0.8)]
    uo, So, Co = load(old_p, arms); un, Sn, Cn = load(new_p, arms)
    assert np.array_equal(uo, un)
    cnts = np.concatenate([rng.multinomial(len(uo), np.full(len(uo), 1.0 / len(uo)), size=2000).astype(np.float64) for _ in range(B // 2000)], 0)
    bo = (cnts @ So.T) / (cnts @ Co.T); bn = (cnts @ Sn.T) / (cnts @ Cn.T)
    po, pn = So.sum(1) / Co.sum(1), Sn.sum(1) / Cn.sum(1)
    ix = {a: i for i, a in enumerate(arms)}
    out = {"arms": {}, "tests": {}}
    for a in arms:
        i = ix[a]; d = bn[:, i] - bo[:, i]
        out["arms"][a] = {"recorded": float(po[i]), "recorded_ci95": np.percentile(bo[:, i], [2.5, 97.5]).tolist(), "replay_timeline": float(pn[i]),
                          "replay_timeline_ci95": np.percentile(bn[:, i], [2.5, 97.5]).tolist(), "diff": float(pn[i] - po[i]), "diff_ci95": np.percentile(d, [2.5, 97.5]).tolist(),
                          "diff_p": float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1)))}
    for rule, P, Bt in (("recorded", po, bo), ("replay_timeline", pn, bn)):
        best = min(CL, key=lambda c: P[ix[c]]); es = []
        for m in MODELS:
            d = Bt[:, ix[f"{m} q0.5"]] - Bt[:, ix[best]]
            es.append({"model": m, "diff": float(P[ix[f"{m} q0.5"]] - P[ix[best]]), "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
                       "p_boot": float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (B + 1)))})
        for e, a in zip(es, holm([e["p_boot"] for e in es])): e["p_holm_family4"] = float(a)
        out["tests"][rule] = {"best_classical": best, "vs_best": es}
    R["settings"][sname] = out
json.dump(R, open(sys.argv[1], "w"), indent=1, default=float); print("ANALYSIS_RT_DONE")

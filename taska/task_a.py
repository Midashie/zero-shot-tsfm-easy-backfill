"""Task A: CESNET next-hour capacity provisioning on the frozen primary windows.

Usage: python3 -I task_a.py CENTILE_REPO NPZ MANIFEST_DIR OUT_DIR MODEL_NAME MODEL_DIR
MODEL_NAME: bolt | chronos2 | baselines
Context = first 168 hourly bins of each window (log1p bytes), decision at bin 168, target = bin 168 (h=1).
Everything the model sees ends before the decision time (G3 passed for this path).
"""
import sys, os, json, time, importlib
import numpy as np, pandas as pd, torch

repo, npz_path, man_dir, out, name = sys.argv[1:6]
mdir = sys.argv[6] if len(sys.argv) > 6 else None
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.join(repo, "common"))
import metrics_v2 as MET

SPLIT, HMAX, N = 168, 24, 192
HS = [1, 4, 24]
QS9 = tuple(np.linspace(0.1, 0.9, 9))
QLEV = list(QS9) + [0.95, 0.99]
TAUS = (0.5, 0.8, 0.9, 0.95, 0.99)
CAPCOL = {0.5: 4, 0.8: 7, 0.9: 8, 0.95: 9, 0.99: 10}

z = np.load(npz_path)
r, v, ts = z["r"], z["valid"], z["ts"]
S, T = r.shape
cut = int(T * 0.8)
prim = pd.read_csv(os.path.join(man_dir, "cesnet_primary_windows.csv"))
sid = prim["series_row"].values.astype(int)
st = prim["start_bin"].values.astype(int)
assert len(prim) == 3000 and st.min() >= cut
Z = np.stack([r[i, s:s + N] for i, s in zip(sid, st)]).astype(np.float32)
assert all(v[i, s:s + N].min() > 0 for i, s in zip(sid, st)), "window with invalid bin"
scale = {i: float(np.mean(np.abs(np.diff(r[i, :cut][v[i, :cut] > 0])))) for i in range(S)}
true = {h: Z[:, SPLIT + h - 1] for h in HS}
ctx_all = Z[:, :SPLIT]


def provisioning(cap_r, act_r):
    cap, act = np.expm1(cap_r), np.expm1(act_r)
    over = float(np.maximum(cap - act, 0).sum() / max(act.sum(), 1e-9) * 100)
    viol = float((act > cap).mean() * 100)
    return over, viol


def boot_ci(cap_r, act_r, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    cap, act = np.expm1(cap_r), np.expm1(act_r)
    n = len(cap)
    ov, vi = [], []
    for _ in range(B):
        k = rng.integers(0, n, n)
        ov.append(np.maximum(cap[k] - act[k], 0).sum() / act[k].sum() * 100)
        vi.append((act[k] > cap[k]).mean() * 100)
    return [float(np.percentile(ov, 2.5)), float(np.percentile(ov, 97.5))], [float(np.percentile(vi, 2.5)), float(np.percentile(vi, 97.5))]


res = {"model": name, "n_windows": int(len(Z))}
t0 = time.time()
if name == "bolt":
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(mdir, device_map="cpu", torch_dtype=torch.bfloat16)
    Q = np.concatenate([pipe.predict_quantiles(torch.tensor(ctx_all[i:i + 256]), prediction_length=HMAX, quantile_levels=QLEV)[0].float().numpy()
                        for i in range(0, len(ctx_all), 256)], 0)
    res["note"] = "Chronos-Bolt library clamps levels above 0.9 to the 0.9 value, so tau 0.95 and 0.99 equal tau 0.9 here"
elif name == "chronos2":
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(mdir, device_map="cpu")
    outs = []
    for i in range(0, len(ctx_all), 128):
        q, _ = pipe.predict_quantiles([{"target": c} for c in ctx_all[i:i + 128]], prediction_length=HMAX, quantile_levels=QLEV)
        outs.append(np.stack([x[0].numpy() for x in q]) if q[0].ndim == 3 else np.stack([x.numpy() for x in q]))
        print(f"  {min(i + 128, len(ctx_all))}/{len(ctx_all)}", flush=True)
    Q = np.concatenate(outs, 0)
    res["note"] = "univariate, runtimes-free: context is the series only, no covariates"
if name in ("bolt", "chronos2"):
    assert Q.shape == (3000, HMAX, len(QLEV)), Q.shape
    res["seconds_total_cpu"] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(out, f"Q_{name}.npz"), Q=Q.astype(np.float32), qlev=np.array(QLEV))
    res["forecast"] = {}
    for h in HS:
        qh = Q[:, h - 1, :]
        mase, _ = MET.mase_per_series(qh[:, 4], true[h], sid, scale)
        res["forecast"][f"h{h}"] = {"MASE": mase, "pinball99": MET.pinball(true[h], qh[:, 10], 0.99),
                                    "CRPS9": MET.crps_from_quantiles(true[h], qh[:, :9], QS9)}
    res["provisioning"] = {}
    for tau in TAUS:
        ov, vi = provisioning(Q[:, 0, CAPCOL[tau]], true[1])
        cio, civ = boot_ci(Q[:, 0, CAPCOL[tau]], true[1])
        res["provisioning"][str(tau)] = {"over_pct": ov, "viol_pct": vi, "over_ci95": cio, "viol_ci95": civ}
    res["calibration_h1"] = {str(q): float((true[1] <= Q[:, 0, j]).mean()) for j, q in enumerate(QLEV)}
else:
    last = Z[:, SPLIT - 1]
    wk = Z[:, SPLIT - 168]
    ew = Z[:, 0].copy()
    for k in range(1, SPLIT):
        ew = 0.3 * Z[:, k] + 0.7 * ew
    res["forecast"] = {}
    for h in HS:
        sn = Z[:, SPLIT + h - 1 - 168]
        res["forecast"][f"h{h}"] = {nm: {"MASE": MET.mase_per_series(p, true[h], sid, scale)[0]}
                                    for nm, p in [("persistence", last), ("seasonal-naive", sn), ("EWMA", ew)]}
    res["provisioning"] = {}
    for nm, cap in [("rule:lastweek x1.0", wk), ("rule:lastweek x1.2", np.log1p(1.2 * np.expm1(wk))),
                    ("rule:EWMA x1.2", np.log1p(1.2 * np.expm1(ew))), ("rule:last value x1.0", last),
                    ("rule:last value x1.2", np.log1p(1.2 * np.expm1(last)))]:
        ov, vi = provisioning(cap, true[1])
        cio, civ = boot_ci(cap, true[1])
        res["provisioning"][nm] = {"over_pct": ov, "viol_pct": vi, "over_ci95": cio, "viol_ci95": civ}
with open(os.path.join(out, f"task_a_{name}.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

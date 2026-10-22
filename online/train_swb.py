"""Train the SafeWall-style baseline for one trace. Usage: python3 -I train_swb.py CENTILE_REPO TRACE   (TRACE: theta, theta2018, theta2019, theta2020)"""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, xgboost as xgb
from rt_common import get_trace
from swb_common import *

trace = sys.argv[2]
Z, usr, ed = get_trace(trace)
ts, dur, lim, nn = Z["ts"], Z["dur"], Z["lim"], Z["nn"]
t0 = int(ts[Z["win_rows"]].min())
cand = np.where(ed <= t0)[0]                                  # history: jobs that ended before the window starts
by = {}
idx = np.argsort(usr, kind="stable"); us = usr[idx]
cuts = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
for a, b in zip(cuts[:-1], cuts[1:]):
    r = idx[a:b]; o = np.argsort(ed[r], kind="stable"); by[us[a]] = (r[o], ed[r][o], ts[r][o])
rows, ctxs = [], []
t1 = time.time()
for i in cand[np.argsort(ts[cand], kind="stable")]:
    rr, ee, tt = by[usr[i]]
    k = int(np.searchsorted(ee, ts[i], side="right")); m = tt[:k] < ts[i]
    c = rr[:k][m][-C:]
    if len(c) >= 3: rows.append(i); ctxs.append(c)
rows = np.array(rows)
X = feats(ctxs, rows, dur, lim, nn); y = np.minimum(dur[rows], lim[rows])       # label capped at the declared walltime
ntr = int(0.8 * len(rows)); tr, te = slice(0, ntr), slice(ntr, None)
print(trace, "history jobs", len(cand), "with >=3 context", len(rows), "train", ntr, "test", len(rows) - ntr, f"features {time.time() - t1:.0f}s", flush=True)
mdl = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=2, objective="reg:squarederror")
mdl.fit(X[tr], y[tr])
yh_tr = np.maximum(mdl.predict(X[tr]), 0.0)
bg, per = buffers(usr[rows[tr]], y[tr] - yh_tr)
yh = np.maximum(mdl.predict(X[te]), 0.0)
bu = np.array([user_buffer(u, bg, per) for u in usr[rows[te]]])
rec = yh + np.maximum(bu, 0.0); yt = y[te]
acc = lambda p: float(np.mean(np.minimum(p, yt) / np.maximum(np.maximum(p, yt), 1e-9)))
met = {"trace": trace, "history_jobs": int(len(cand)), "jobs_with_3_context": int(len(rows)), "n_train": ntr, "n_test": int(len(rows) - ntr), "users_in_train_buffers": len(per),
       "global_buffer_s": bg, "ur_unbuffered_pct": float(np.mean(yt > yh) * 100), "ur_buffered_pct": float(np.mean(yt > rec) * 100),
       "accuracy_unbuffered": acc(yh), "accuracy_buffered": acc(rec), "rmse_unbuffered_s": float(np.sqrt(np.mean((yh - yt) ** 2))), "rmse_buffered_s": float(np.sqrt(np.mean((rec - yt) ** 2))),
       "mae_buffered_s": float(np.mean(np.abs(rec - yt))), "median_runtime_test_s": float(np.median(yt)),
       "reproduced": bool(35 <= np.mean(yt > yh) * 100 <= 60 and 1 <= np.mean(yt > rec) * 100 <= 15)}
mdl.save_model(f"{OUTD}/{trace}_model.json"); pickle.dump({"bg": bg, "per": per}, open(f"{OUTD}/{trace}_buf.pkl", "wb"))
json.dump(met, open(f"{OUTD}/{trace}_sanity.json", "w"), indent=1); print(json.dumps(met, indent=1))

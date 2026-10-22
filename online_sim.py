"""Exact online EASY replay: each job's walltime estimate is computed at its arrival from the jobs that had finished IN THIS RUN'S OWN schedule.
Usage: python3 -I online_sim.py CENTILE_REPO TRACE NODES ARM OUT_DIR
ARM: validate (user-limit estimates, contexts built but not forecast; checks them against the replay-timeline contexts from build_rt.py),
     tsafrir, q90 (completion-causal classical estimates), bolt (Chronos-Bolt-small, q 0.5).
Context = user's last 96 finished jobs by finish time: a job finishes at (its start in this run) + min(runtime, kill limit) if it is a replay-window job
the machine can fit, and at its recorded end time otherwise (history before the window, window jobs wider than the machine). Submitted strictly earlier."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json, time, heapq, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, hashlib, pickle
from rt_common import get_trace, OUT

repo, trace, NODES, ARM, out = sys.argv[1:6]; NODES = int(NODES); C = int(os.environ.get("CTX", 96)); TAG = os.environ.get("TAG", "")
os.environ["NODES"] = str(NODES); sys.path.insert(0, os.path.join(repo, "pretrain"))
os.makedirs(out, exist_ok=True)
Z, usr_all, ed_all = get_trace(trace)
rows = Z["win_rows"]; ts_all, dur_all = Z["ts"], Z["dur"]
lim_all = np.maximum(Z["lim"], dur_all)
sub = ts_all[rows].astype(float); nn = Z["nn"][rows]; dur = dur_all[rows]; lim = lim_all[rows]
fit = nn <= NODES
fidx = np.where(fit)[0]                                   # positions (in window order) of jobs the machine can run
n = len(fidx); fsub, fnn, fdur, flim = sub[fidx], nn[fit], dur[fit], lim[fit]
frow = rows[fidx]                                         # table row of each simulated job
code, uniq = pd.factorize(usr_all); code = np.asarray(code)
in_sim = np.zeros(len(usr_all), bool); in_sim[frow] = True
# 'pre' set per user: every row that is not simulated (history, dropped wide jobs); finish = recorded end
pre = {}
r_all = np.where(~in_sim)[0]
o = np.lexsort((r_all, ed_all[r_all], code[r_all]))     # by user, finish time, row
r_s = r_all[o]; c_s = code[r_s]
cuts = np.flatnonzero(np.r_[True, c_s[1:] != c_s[:-1], True])
for a, b in zip(cuts[:-1], cuts[1:]):
    rr = r_s[a:b]; pre[c_s[a]] = (ed_all[rr].astype(float), ts_all[rr].astype(float), np.log1p(dur_all[rr]), rr)
logd = np.log1p(dur_all)
fin = {}                                                  # user code -> lists (finish time, log runtime, row) of simulated jobs finished so far


def context(j, t):
    """Return (rows, log-runtime values) of the last C finished jobs for simulated job j at time t, ordered by finish time."""
    c = code[frow[j]]
    fs, rs, vs = [], [], []
    if c in pre:
        pf, pts, pv, pr = pre[c]
        k = int(np.searchsorted(pf, t, side="right"))
        lo = max(0, k - 3 * C)
        m = pts[lo:k] < t
        fs.append(pf[lo:k][m][-C:]); vs.append(pv[lo:k][m][-C:]); rs.append(pr[lo:k][m][-C:])
    if c in fin:
        a, b, r_ = fin[c]
        fs.append(np.array(a[-3*C:])); vs.append(np.array(b[-3*C:])); rs.append(np.array(r_[-3*C:]))
    if not fs: return np.zeros(0, np.int64), np.zeros(0)
    f_, v_, r_ = np.concatenate(fs), np.concatenate(vs), np.concatenate(rs)
    o = np.lexsort((r_, f_))
    return r_[o][-C:], v_[o][-C:]


QIDX = int(os.environ.get("QIDX", 4))     # index into LEV (0.1 ... 0.9); 4 is the median, 7 is 0.8
MODELS = ("bolt", "chronos2", "chronos2cov", "moirai2", "ttm90", "timesfm")
MODEL = ARM.split("_")[0] if ARM.split("_")[0] in MODELS else None
SWARM = ARM.split("_")[0] if ARM.split("_")[0] in ("swb", "xgbraw") else None      # SafeWall-style buffered XGBoost, and the unbuffered diagnostic
if SWARM:
    from swb_common import load_swb, feats, user_buffer
    SWM, SW_BG, SW_PER = load_swb(trace)
    SW_DUR, SW_LIM, SW_NN = Z["dur"], Z["lim"], Z["nn"]
LEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
UP = _IN + "/ZeroShotSchedule"; ULP = _IN + "/ZeroShotScheduleLocal"
fcast = None
if MODEL:
    import torch
    torch.set_num_threads(2)
    def padL(vs, L):
        x = np.zeros((len(vs), L), np.float32); m = np.zeros((len(vs), L), bool)
        for i, v in enumerate(vs):
            v = v[-L:]; x[i, L - len(v):] = v; m[i, L - len(v):] = True
        return x, m
    if MODEL == "bolt":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(f"{UP}/chronos-bolt-small", device_map="cpu", torch_dtype=torch.bfloat16)
        def fcast(vs): return pipe.predict_quantiles([torch.tensor(v.astype(np.float32)) for v in vs], prediction_length=1, quantile_levels=LEV)[0][:, 0, QIDX].float().numpy()
    elif MODEL in ("chronos2", "chronos2cov"):
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(f"{UP}/chronos-2", device_map="cpu")
        NNL, LIML, GAPL = (np.log1p(Z[k]).astype(np.float32) for k in ("nn", "lim", "gap"))   # same covariates as run_forecast_b.py
        def fcast(vs):
            items = []
            for v, j in zip(vs, CUR_J):
                d = {"target": v.astype(np.float32)}
                if MODEL == "chronos2cov":
                    r_, row = EX[j]
                    d["past_covariates"] = {"nodes": NNL[r_], "limit": LIML[r_], "gap": GAPL[r_]}
                    d["future_covariates"] = {"nodes": NNL[row:row + 1], "limit": LIML[row:row + 1], "gap": GAPL[row:row + 1]}
                items.append(d)
            q, _ = pipe.predict_quantiles(items, prediction_length=1, quantile_levels=LEV)
            return np.array([t[0, 0, 4].item() for t in q])
    elif MODEL == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained(f"{UP}/moirai-2.0")
        mm = Moirai2Forecast(module=mod, prediction_length=1, context_length=C, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        def fcast(vs):
            x, m = padL(vs, C)
            with torch.no_grad():
                o = mm(past_target=torch.tensor(x)[:, :, None], past_observed_target=torch.tensor(m)[:, :, None], past_is_pad=torch.tensor(~m))
            return o[:, :, 0].numpy()[:, 4]
    elif MODEL == "ttm90":
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction as T
        mm = T.from_pretrained(f"{ULP}/ttm-90-30").eval()
        Fd = _OUT + "/taskb" if trace == "fdata" else _OUT + "/pm100"
        Ft = np.load(f"{Fd}/F_ttm90.npz")
        res_ = np.sort((np.log1p(Z["dur"][Z["cal_rows"]]) - Ft["cal_P"])[Ft["cal_forecast_mask"]])
        RES = float(res_[min(int(np.ceil((len(res_) + 1) * 0.5)) - 1, len(res_) - 1)])   # split-conformal median offset, calibration set only
        def fcast(vs):
            x, mask = padL(vs, 90)
            mu = (x * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
            sd = np.sqrt((((x - mu) ** 2) * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)) + 1e-6
            xs = torch.tensor(((x - mu) / sd) * mask, dtype=torch.float32)[:, :, None]
            with torch.no_grad():
                o = mm(past_values=xs, past_observed_mask=torch.tensor(mask)[:, :, None], freq_token=torch.full((len(vs),), 7, dtype=torch.long), return_loss=False)
            return o.prediction_outputs[:, 0, 0].numpy() * sd[:, 0] + mu[:, 0] + RES
    elif MODEL == "timesfm":
        import timesfm
        mm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(f"{UP}/timesfm-2.5", torch_compile=False)
        mm.compile(timesfm.ForecastConfig(max_context=96, max_horizon=32, normalize_inputs=True, use_continuous_quantile_head=True,
                                          force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        def fcast(vs):
            _, q = mm.forecast(horizon=1, inputs=[v.astype(np.float64) for v in vs])
            return np.asarray(q)[:, 0, 5]            # column 0 is the mean, column 5 is level 0.5

est = np.maximum(flim, 60.0).copy()                       # user-limit estimates until overwritten
CPATH = os.path.join(out, "cache", f"{ARM}{TAG}_{trace}_{NODES}.pkl"); os.makedirs(os.path.dirname(CPATH), exist_ok=True)
CACHE = pickle.load(open(CPATH, "rb")) if os.path.exists(CPATH) else {}     # forecasts are a pure function of the context, so a restart replays the finished part from here
ctx_rows = {}; nbatch = 0; nfore = 0; tbolt = 0.0


EX = {}; CUR_J = []


def on_arrivals(jobs, t):
    """Fill est[j] for jobs arriving at time t."""
    global nbatch, nfore, tbolt, CUR_J
    if ARM == "bolt_frozen":
        ctxs = []
        for j in jobs:
            w = WCF[fidx[j]]; w = w[w >= 0]; ctxs.append((w, logd[w]))
    else:
        ctxs = [context(j, t) for j in jobs]
    if ARM == "validate":
        for j, (r_, v_) in zip(jobs, ctxs): ctx_rows[j] = r_
        return
    need = []
    for j, (r_, v_) in zip(jobs, ctxs):
        if len(v_) < 3: est[j] = max(flim[j], 60.0); continue
        if ARM == "q90":
            est[j] = max(min(np.quantile(np.expm1(v_[-50:]), 0.9) * 1.1, flim[j]), 60.0)
        elif ARM == "tsafrir":
            est[j] = float(np.clip(np.mean(np.expm1(v_[-2:])), 60.0, max(flim[j], 60.0)))
        else:
            need.append((j, v_)); EX[j] = (r_, frow[j])
    if need and SWARM:
        rr = [EX[j][0] for j, _ in need]; ow = [EX[j][1] for j, _ in need]
        yh = np.maximum(SWM.predict(feats(rr, ow, SW_DUR, SW_LIM, SW_NN)), 0.0)
        for (j, _), p, o in zip(need, yh, ow):
            rec = p if SWARM == "xgbraw" else p + max(0.0, user_buffer(usr_all[o], SW_BG, SW_PER))
            est[j] = min(max(float(rec), 60.0), max(flim[j], 60.0))
        nfore += len(need); nbatch += 1
        need = []
    if need:
        t1 = time.time()
        if MODEL == "chronos2cov":      # the forecast depends on the covariates too, so they enter the cache key
            keys = [hashlib.md5(np.ascontiguousarray(v, dtype=np.float64).tobytes() + np.ascontiguousarray(EX[j][0]).tobytes() + int(EX[j][1]).to_bytes(8, "little")).digest() for j, v in need]
        else:
            keys = [hashlib.md5(np.ascontiguousarray(v, dtype=np.float64).tobytes()).digest() for _, v in need]
        todo = [i for i, k in enumerate(keys) if k not in CACHE]
        if todo:
            CUR_J = [need[i][0] for i in todo]
            qq_ = np.asarray(fcast([need[i][1] for i in todo]), float)
            for i, x in zip(todo, qq_): CACHE[keys[i]] = float(x)
            nbatch += 1; nfore += len(todo)
            if nbatch % 300 == 0:
                pickle.dump(CACHE, open(CPATH + ".tmp", "wb")); os.replace(CPATH + ".tmp", CPATH)
        q = np.array([CACHE[k] for k in keys])
        tbolt += time.time() - t1
        for (j, _), qq in zip(need, q): est[j] = min(max(float(np.expm1(qq)), 60.0), max(flim[j], 60.0))


ENF = os.environ.get("ENFORCE", "0") == "1"       # kill-and-requeue mode (pre-registered): a job whose runtime exceeds its estimate dies at start + estimate
NKILL = 0; WASTED = 0.0


def easy_online(sub, nn, dur, lim, cap):
    """CENTILE's easy_sim (run_backfill_replay.py), with estimates computed at arrival and finished jobs recorded per user."""
    n = len(sub); order = np.argsort(sub, kind="stable")
    global NKILL, WASTED, FSTART
    start = np.full(n, -1.0); killed = np.zeros(n, bool); nbf = 0
    FSTART = np.full(n, -1.0); attempt = np.zeros(n, int)
    run = []; runest = {}; free = cap; queue = []; i = 0; t = sub[order[0]]
    def dies(j): return ENF and attempt[j] == 0 and dur[j] > est[j]
    def finish_actual(j, st): return st + est[j] if dies(j) else st + min(dur[j], lim[j])
    while i < n or queue or run:
        while run and run[0][0] <= t + 1e-9:
            ft, j = heapq.heappop(run); free += nn[j]; runest.pop(j, None)
            if dies(j):                                   # killed at its estimate: not a completed job; rejoins the back of the queue with its kill limit
                attempt[j] = 1; NKILL += 1; WASTED += nn[j] * (ft - start[j]); est[j] = max(lim[j], 60.0); queue.append(j); continue
            c = code[frow[j]]; L = fin.setdefault(c, ([], [], [])); L[0].append(ft); L[1].append(logd[frow[j]]); L[2].append(frow[j])
        batch = []
        while i < n and sub[order[i]] <= t + 1e-9: batch.append(order[i]); i += 1
        if batch:
            on_arrivals(batch, t)                         # contexts use only jobs finished by t in this run
            queue.extend(batch)
        progressed = True
        while progressed and queue:
            progressed = False
            h = queue[0]
            if nn[h] <= free:
                st = t; start[h] = st if start[h] < 0 else start[h]; FSTART[h] = st; killed[h] = dur[h] > lim[h]
                heapq.heappush(run, (finish_actual(h, st), h)); runest[h] = st + est[h]
                free -= nn[h]; queue.pop(0); progressed = True; continue
            ends = sorted((max(runest[j], t + 600.0), nn[j]) for j in runest)
            acc = free; shadow = None; spare = 0
            for fe, nd in ends:
                acc += nd
                if acc >= nn[h]: shadow = fe; spare = acc - nn[h]; break
            if shadow is None: break
            for k in list(queue[1:]):
                if nn[k] <= free and (t + est[k] <= shadow + 1e-9 or nn[k] <= spare):
                    st = t; start[k] = st if start[k] < 0 else start[k]; FSTART[k] = st; killed[k] = dur[k] > lim[k]
                    heapq.heappush(run, (finish_actual(k, st), k)); runest[k] = st + est[k]
                    free -= nn[k]; queue.remove(k); nbf += 1; progressed = True
                    if nn[k] <= spare: spare -= nn[k]
            if not progressed: break
        nt = []
        if run: nt.append(run[0][0])
        if i < n: nt.append(sub[order[i]])
        if not nt: break
        t = max(t, min(nt))
    return start, killed, nbf


t0 = time.time()
start, killed, nbf = easy_online(fsub, fnn, fdur, flim, NODES)
ok = start >= 0; wait = start[ok] - fsub[ok]; run_ = np.minimum(fdur[ok], flim[ok])
if ENF: wait = FSTART[ok] - fsub[ok]                      # successful attempt, measured from the original submit time
bs = np.maximum((wait + run_) / np.maximum(run_, 60.0), 1.0); mk = ((FSTART[ok] if ENF else start[ok]) + run_).max() - fsub.min()
res = {"ctx_cap": C, "trace": trace, "nodes": NODES, "arm": ARM, "mean_bs": float(bs.mean()), "p95_bs": float(np.quantile(bs, 0.95)), "util_pct": float((fnn[ok] * run_).sum() / (NODES * mk) * 100),
       "started_pct": float(ok.mean() * 100), "backfilled": int(nbf), "seconds": round(time.time() - t0, 1), "bolt_forecasts": nfore, "bolt_batches": nbatch, "bolt_seconds": round(tbolt, 1), "enforce": ENF, "killed_attempts": int(NKILL), "killed_jobs_pct": float(NKILL / len(fsub) * 100),
       "wasted_node_hours_pct_of_used": float(WASTED / ((fnn[ok] * run_).sum() + WASTED) * 100)}
if ARM == "validate":
    WC = np.load(f"{OUT}/rt_{trace}_{NODES}/contexts_c96.npz")["win_ctx"]
    bad = 0; checked = 0
    for j in range(n):
        old = WC[fidx[j]]; old = old[old >= 0]
        checked += 1; bad += not np.array_equal(old, ctx_rows[j])
    res.update({"contexts_checked": checked, "contexts_not_equal_to_build_rt": int(bad)})
full = np.full(n, np.nan, np.float32); full[ok] = bs
np.savez_compressed(os.path.join(out, f"online_{ARM}{TAG}_{NODES}.npz"), usr=usr_all[frow], fit_index=fidx, bs=full)
json.dump(res, open(os.path.join(out, f"online_{ARM}{TAG}_{NODES}.json"), "w"), indent=1); print(json.dumps(res, indent=1))

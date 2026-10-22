"""Task B forecasts: next-runtime quantiles for every replayed job and for the calibration set.

Usage: python -I run_forecast_b.py MODEL CTX_NPZ OUT_DIR       MODEL: bolt chronos2 chronos2cov moirai2 ttm90 timesfm
Series values are log1p(runtime seconds). Context = the user's last C finished jobs (completion order, left padded).
Output F_<model>.npz: win_Q / cal_Q (n, 9) at levels 0.1..0.9 (native models), or win_P / cal_P (n,) point forecasts (TTM).
Rows with fewer than 3 context jobs are not forecast (the replay gives them the declared limit).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json, time
import numpy as np, torch

model, ctx_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(2)
U = _IN + "/ZeroShotSchedule"
UL = _IN + "/ZeroShotScheduleLocal"
Z = np.load(ctx_path)
C = Z["win_ctx"].shape[1]
dur = np.log1p(Z["dur"]).astype(np.float32)
lim, nn, gap, ts = np.log1p(Z["lim"]).astype(np.float32), np.log1p(Z["nn"]).astype(np.float32), np.log1p(Z["gap"]).astype(np.float32), Z["ts"]
LEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMOKE = int(os.environ.get("SMOKE", 0))


def predictor():
    if model == "bolt":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(f"{U}/chronos-bolt-small", device_map="cpu", torch_dtype=torch.bfloat16)
        def f(job_rows, ctx):
            seqs = [torch.tensor(dur[c[c >= 0]]) for c in ctx]
            return pipe.predict_quantiles(seqs, prediction_length=1, quantile_levels=LEV)[0][:, 0, :].float().numpy()
        return f, 256
    if model in ("chronos2", "chronos2cov"):
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(f"{U}/chronos-2", device_map="cpu")
        def f(job_rows, ctx):
            items = []
            for rw, c in zip(job_rows, ctx):
                c = c[c >= 0]
                d = {"target": dur[c]}
                if model == "chronos2cov":
                    d["past_covariates"] = {"nodes": nn[c], "limit": lim[c], "gap": gap[c]}
                    d["future_covariates"] = {"nodes": nn[rw:rw + 1], "limit": lim[rw:rw + 1], "gap": gap[rw:rw + 1]}
                items.append(d)
            q, _ = pipe.predict_quantiles(items, prediction_length=1, quantile_levels=LEV)
            return np.stack([t[0, 0, :].numpy() for t in q])
        return f, 128
    if model == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained(f"{U}/moirai-2.0")
        m = Moirai2Forecast(module=mod, prediction_length=1, context_length=C, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        def f(job_rows, ctx):
            x = np.where(ctx >= 0, dur[np.maximum(ctx, 0)], 0.0).astype(np.float32)
            obs = torch.tensor(ctx >= 0)[:, :, None]
            with torch.no_grad():
                o = m(past_target=torch.tensor(x)[:, :, None], past_observed_target=obs, past_is_pad=torch.tensor(ctx < 0))
            return o[:, :, 0].numpy()
        return f, 128
    if model == "ttm90":
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction as T
        m = T.from_pretrained(f"{UL}/ttm-90-30").eval()
        L = 90
        def f(job_rows, ctx):
            c = ctx[:, -L:]
            mask = c >= 0
            x = np.where(mask, dur[np.maximum(c, 0)], 0.0).astype(np.float32)
            mu = (x * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
            sd = np.sqrt((((x - mu) ** 2) * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)) + 1e-6
            xs = torch.tensor(((x - mu) / sd) * mask, dtype=torch.float32)[:, :, None]
            with torch.no_grad():
                o = m(past_values=xs, past_observed_mask=torch.tensor(mask)[:, :, None], freq_token=torch.full((len(c),), 7, dtype=torch.long), return_loss=False)
            return o.prediction_outputs[:, 0, 0].numpy() * sd[:, 0] + mu[:, 0]
        return f, 256
    if model == "timesfm":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(f"{U}/timesfm-2.5", torch_compile=False)
        m.compile(timesfm.ForecastConfig(max_context=96, max_horizon=32, normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        def f(job_rows, ctx):
            _, q = m.forecast(horizon=1, inputs=[dur[c[c >= 0]].astype(np.float64) for c in ctx])
            return np.asarray(q)[:, 0, 1:10]          # column 0 is the mean, columns 1 to 9 are levels 0.1 to 0.9
        return f, 64
    raise SystemExit("unknown model")


fn, bs = predictor()
res, lat = {}, {"model": model, "threads": 2}
for tag in os.environ.get("TAGS", "cal,win").split(","):
    rows, ctx, ln = Z[f"{tag}_rows"], Z[f"{tag}_ctx"], Z[f"{tag}_len"]
    keep = np.where(ln >= 3)[0]
    if SMOKE:
        keep = keep[:SMOKE]
    order = keep[np.argsort(ln[keep], kind="stable")]            # batch similar lengths together
    first = None
    t0 = time.time()
    outs = {}
    ck = os.path.join(out, f"ckpt_{model}_{tag}.npz")
    start_i = 0
    if os.path.exists(ck) and not SMOKE:                       # resume after a container restart
        c = np.load(ck)
        start_i = int(c["next_i"])
        for ixs, vals in zip(np.split(c["ix"], np.cumsum(c["lens"])[:-1]), np.split(c["vals"], np.cumsum(c["lens"])[:-1])):
            outs[tuple(ixs)] = vals
        print(f"resumed {tag} at {start_i}", flush=True)
    for i in range(start_i, len(order), bs):
        ix = order[i:i + bs]
        outs[tuple(ix)] = fn(rows[ix], ctx[ix])
        if (i // bs) % 50 == 49 and not SMOKE:
            ks = list(outs)
            np.savez(ck, next_i=i + bs, ix=np.concatenate([np.array(k) for k in ks]), lens=np.array([len(k) for k in ks]), vals=np.concatenate([outs[k] for k in ks]))
        if (i // bs) % 20 == 0:
            print(f"{model} {tag} {min(i + bs, len(order))}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    A = np.full((len(rows),) + (outs[next(iter(outs))].shape[1:]), np.nan, np.float32)
    for ix, o in outs.items():
        A[list(ix)] = o
    lat[f"seconds_{tag}"] = round(time.time() - t0, 1)
    lat[f"ms_per_forecast_{tag}"] = round(1000 * (time.time() - t0) / max(len(order), 1), 2)
    assert np.isfinite(A[keep]).all()
    res[f"{tag}_{'Q' if A.ndim == 2 else 'P'}"] = A
    res[f"{tag}_forecast_mask"] = np.isin(np.arange(len(rows)), keep)
suffix = "_smoke" if SMOKE else ""
np.savez_compressed(os.path.join(out, f"F_{model}{suffix}.npz"), **res)
with open(os.path.join(out, f"lat_{model}{suffix}.json"), "w") as fh:
    json.dump(lat, fh, indent=2)
print(json.dumps(lat, indent=2))

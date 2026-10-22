"""Run one model on the primary, recent and calibration windows; save forecasts and CPU latency.

Usage: python -I run_models.py MODEL OUT_DIR      MODEL in bolt chronos2 timesfm moirai2 ttm90 ttm180 ttm512
Output: OUT/F_<model>_<set>.npz with Q (n, 24, K) quantile forecasts or P (n, 24) point forecasts, in log1p bytes,
        plus OUT/lat_<model>.json. Inputs are scaled and unscaled inside this file where a model needs it.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, os, json, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_a import load_set, extended_context, SPLIT, HMAX

model, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(2)
U = _IN + "/ZeroShotSchedule"
UL = _IN + "/ZeroShotScheduleLocal"
QLEV = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def make_predictor():
    """Returns (predict_fn(ctx float32 (b,168) or dict) -> array, kind, qlevels)."""
    if model == "bolt":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(f"{U}/chronos-bolt-small", device_map="cpu", torch_dtype=torch.bfloat16)
        f = lambda c: pipe.predict_quantiles(torch.tensor(c), prediction_length=HMAX, quantile_levels=QLEV)[0].float().numpy()
        return f, "Q", QLEV, 256
    if model == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(f"{U}/chronos-2", device_map="cpu")
        def f(c):
            q, _ = pipe.predict_quantiles([{"target": x} for x in c], prediction_length=HMAX, quantile_levels=QLEV)
            return np.stack([t[0].numpy() for t in q])
        return f, "Q", QLEV, 128
    if model == "timesfm":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(f"{U}/timesfm-2.5", torch_compile=False)
        m.compile(timesfm.ForecastConfig(max_context=192, max_horizon=HMAX, normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        def f(c):
            p, q = m.forecast(horizon=HMAX, inputs=[x for x in c])
            f.last_point = np.asarray(p)
            return np.asarray(q)[:, :, 1:10]   # column 0 is the mean, columns 1 to 9 are levels 0.1 to 0.9
        return f, "Q", QLEV[:9], 64
    if model == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained(f"{U}/moirai-2.0")
        m = Moirai2Forecast(module=mod, prediction_length=HMAX, context_length=SPLIT, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        def f(c):
            x = torch.tensor(c)[:, :, None]
            with torch.no_grad():
                o = m(past_target=x, past_observed_target=torch.ones_like(x, dtype=torch.bool),
                      past_is_pad=torch.zeros(len(c), SPLIT, dtype=torch.bool))
            return o.permute(0, 2, 1).numpy()           # (b, 24, 9)
        return f, "Q", QLEV[:9], 128
    if model.startswith("ttm"):
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction as T
        L = {"ttm90": 90, "ttm180": 180, "ttm512": 512}[model]
        d = {"ttm90": "ttm-90-30", "ttm180": "ttm-180-60", "ttm512": "ttm-512-96"}[model]
        m = T.from_pretrained(f"{UL}/{d}").eval()
        assert m.config.context_length == L
        def f(c, mask):
            mu = (c * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
            sd = np.sqrt((((c - mu) ** 2) * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)) + 1e-6
            x = torch.tensor(((c - mu) / sd) * mask, dtype=torch.float32)[:, :, None]
            with torch.no_grad():
                o = m(past_values=x, past_observed_mask=torch.tensor(mask)[:, :, None], freq_token=torch.full((len(c),), 7, dtype=torch.long),
                      return_loss=False)
            return o.prediction_outputs[:, :HMAX, 0].numpy() * sd + mu
        return f, "P", None, 128, L
    raise SystemExit("unknown model")


pred = make_predictor()
fn, kind, qlev, bs = pred[:4]
L = pred[4] if len(pred) > 4 else None
lat = {"model": model, "threads": 2, "device": "cpu"}
for sname in ("primary", "recent", "calibration"):
    d = load_set(sname)
    if os.environ.get("SMOKE"):
        k = int(os.environ["SMOKE"])
        d = {**d, "sid": d["sid"][:k], "st": d["st"][:k], "Z": d["Z"][:k], "n": k}
    if L:
        ctx, mask = extended_context(d, L)
        lat["masked_bins_share_" + sname] = float(1 - mask.mean())
    else:
        ctx = d["Z"][:, :SPLIT]
    t0 = time.time()
    outs = []
    for i in range(0, d["n"], bs):
        outs.append(fn(ctx[i:i + bs], mask[i:i + bs]) if L else fn(ctx[i:i + bs]))
        if (i // bs) % 5 == 0:
            print(f"{model} {sname} {min(i + bs, d['n'])}/{d['n']}", flush=True)
    A = np.concatenate(outs, 0).astype(np.float32)
    lat[f"batch_seconds_{sname}"] = round(time.time() - t0, 1)
    lat[f"batch_ms_per_forecast_{sname}"] = round(1000 * (time.time() - t0) / d["n"], 2)
    assert A.shape[:2] == (d["n"], HMAX) and np.isfinite(A).all(), A.shape
    if kind == "Q":
        np.savez_compressed(f"{out}/F_{model}_{sname}.npz", Q=A, qlev=np.array(qlev))
    else:
        np.savez_compressed(f"{out}/F_{model}_{sname}.npz", P=A)
    if sname == "primary":
        ts = []
        for k in range(30):
            t1 = time.time()
            fn(ctx[k:k + 1], mask[k:k + 1]) if L else fn(ctx[k:k + 1])
            ts.append(time.time() - t1)
        lat["single_window_ms_median"] = round(1000 * float(np.median(ts)), 2)
        if model == "timesfm":
            lat["timesfm_mean_vs_q50_gap"] = "see log"
with open(f"{out}/lat_{model}.json", "w") as fh:
    json.dump(lat, fh, indent=2)
print(json.dumps(lat, indent=2))

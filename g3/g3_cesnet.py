"""G3 on CESNET: shifted-future test through CENTILE's window code and Chronos-Bolt.

Usage: python3 -I g3_cesnet.py CENTILE_REPO NPZ MANIFEST_DIR BOLT_DIR OUT_JSON [N_WINDOWS]
"""
import sys, os, json, importlib
import numpy as np, pandas as pd, torch

repo, npz_path, man_dir, bolt_dir, out_json = sys.argv[1:6]
NW = int(sys.argv[6]) if len(sys.argv) > 6 else 512
sys.path.insert(0, os.path.join(repo, "pretrain"))
sys.path.insert(0, os.path.join(repo, "common"))
cc = importlib.import_module("run_cesnet_chronos")
from chronos import BaseChronosPipeline

z = np.load(npz_path)
r, v, ts = z["r"], z["valid"], z["ts"]
S, T = r.shape
cut = int(T * 0.8)
SPLIT, N = cc.SPLIT, cc.N
res = {}

prim = pd.read_csv(os.path.join(man_dir, "cesnet_primary_windows.csv"))
cal = pd.read_csv(os.path.join(man_dir, "cesnet_calibration_windows.csv"))
rec = pd.read_csv(os.path.join(man_dir, "cesnet_recent_windows.csv"))

# 1. label and timing alignment
dec_bin = prim["start_bin"].values + SPLIT
res["alignment"] = {
    "context_last_bin_ends_at_or_before_decision": bool(np.all(prim["start_bin"].values + SPLIT - 1 < dec_bin)),
    "decision_time_equals_label_bin_start": bool(np.all((pd.to_datetime(prim["decision_time_utc"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64").values == ts[dec_bin])),
    "calibration_last_bin_before_cut": int((cal["start_bin"].max() + N - 1)),
    "cut_bin": int(cut),
    "calibration_ends_before_first_eval_bin": bool(cal["start_bin"].max() + N - 1 < cut <= prim["start_bin"].min()),
    "recent_first_bin_at_or_after_cut": bool(rec["start_bin"].min() >= cut),
}
assert res["alignment"]["calibration_ends_before_first_eval_bin"]

# 2. MASE scale must use training bins only: perturb every bin from the cut on and recompute
scale = lambda rr: {i: float(np.mean(np.abs(np.diff(rr[i, :cut][v[i, :cut] > 0])))) for i in range(S)}
rng = np.random.default_rng(0)
r_p = r.copy()
r_p[:, cut:] = rng.normal(10, 3, size=r[:, cut:].shape).astype(np.float32)
res["mase_scale_unchanged_when_test_region_perturbed"] = bool(scale(r) == scale(r_p))

# 3. Bolt on the real window path, one window at a time, with the window's own future perturbed
pipe = BaseChronosPipeline.from_pretrained(bolt_dir, device_map="cpu", torch_dtype=torch.bfloat16)
QLEV = cc.QLEV
sub = prim.sample(n=NW, random_state=0).reset_index(drop=True)
maxdiff = 0.0
n_changed = 0
for _, row in sub.iterrows():
    i, s = int(row["series_row"]), int(row["start_bin"])
    outs = []
    for variant in ("orig", "shifted"):
        rr = r.copy() if variant == "shifted" else r
        if variant == "shifted":
            rr[i, s + SPLIT:] = rng.normal(10, 3, size=T - (s + SPLIT)).astype(np.float32)   # everything at or after the decision
        Ze, SIDe = cc.windows(rr[i:i + 1], v[i:i + 1], ts, s, s + N, 1)                       # CENTILE's own window builder
        assert len(Ze) == 1
        ctx = torch.tensor(Ze[:, :SPLIT])
        q, _ = pipe.predict_quantiles(ctx, prediction_length=cc.HMAX, quantile_levels=QLEV)
        outs.append(q.float().numpy())
    d = float(np.abs(outs[0] - outs[1]).max())
    maxdiff = max(maxdiff, d)
    n_changed += d > 0
res["bolt_shifted_future_test"] = {"windows": int(len(sub)), "windows_with_changed_forecast": int(n_changed),
                                   "max_abs_difference": maxdiff,
                                   "perturbation": "all bins of the window's series at or after the decision bin replaced by noise"}
with open(out_json, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

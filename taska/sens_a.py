"""Post hoc sensitivity metrics for Task A. Builds the same arms as analyze_tuned.py (code before '# operator dots'), then scores:
 M1 mean per-window relative over-provisioning, mean(max(cap-act,0)/act), at matched window violation (20, 30 percent)
 M2 byte over-provisioning (percent of demand) at matched byte shortfall, shortfall = sum(max(act-cap,0))/sum(act), at 1, 2 and 5 percent
Point estimates on the primary windows only. These metrics were chosen after seeing that a constant cap wins the original metric."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import sys, json
src = open(_ROOT + "/taska/analyze_tuned.py").read().split("# operator dots")[0]
sys.argv = ["x", _OUT + "/taska_full", _OUT + "/taska_tuned", "10"]
exec(compile(src, "analyze_tuned_head", "exec"))
y = D["primary"]["Z"][:, SPLIT]
act = np.expm1(y)
def interp(x, v, T):
    o = np.argsort(x); xs, vs = x[o], v[o]
    return round(float(np.interp(T, xs, vs)), 1) if xs[0] <= T <= xs[-1] else None
out = {}
for nm, a in arms.items():
    caps = a["caps"]["primary"]
    cap = np.expm1(caps)
    viol = (act[:, None] > cap).mean(0) * 100
    rel = (np.maximum(cap - act[:, None], 0) / act[:, None]).mean(0) * 100
    over = np.maximum(cap - act[:, None], 0).sum(0) / act.sum() * 100
    short = np.maximum(act[:, None] - cap, 0).sum(0) / act.sum() * 100
    out[nm] = {"M1_rel_over_at_viol20": interp(viol, rel, 20), "M1_rel_over_at_viol30": interp(viol, rel, 30),
               "M2_over_at_short1": interp(short, over, 1), "M2_over_at_short2": interp(short, over, 2), "M2_over_at_short5": interp(short, over, 5)}
json.dump(out, open(_OUT + "/taska_tuned/sensitivity_metrics.json", "w"), indent=1)
for nm, r in out.items(): print(f"{nm:28s}", r)

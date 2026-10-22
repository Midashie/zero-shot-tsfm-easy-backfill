import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, json, os, sys
O=_OUT + "/online"
S=[("PM100 512",512),("F-DATA 20K",20000),("F-DATA 23K",23000)]
arms=[a for a in ("tsafrir","q90","bolt","moirai2","ttm90","chronos2","chronos2cov","timesfm","bolt_c32","bolt_c192") if all(os.path.exists(f"{O}/online_{a}_{n}.npz") for _,n in S)]
def load(a,n):
    z=np.load(f"{O}/online_{a}_{n}.npz",allow_pickle=True); return z["usr"],z["bs"]
res=[]; B=10000
for nm,n in S:
    u0,b0=load("validate",n)
    rng=np.random.default_rng(1000+n); IDX={}
    for a in arms:
        u1,b1=load(a,n); assert (u0==u1).all()
        ok=~np.isnan(b0)&~np.isnan(b1)
        # job-level started in both; per-user sums
        us,inv=np.unique(u0[ok],return_inverse=True)
        s0=np.bincount(inv,b0[ok]);s1=np.bincount(inv,b1[ok]);c=np.bincount(inv).astype(float)
        m0,m1=s0.sum()/c.sum(),s1.sum()/c.sum()
        idx=IDX.setdefault(len(us),rng.integers(0,len(us),(B,len(us))))
        d=(s1[idx].sum(1)-s0[idx].sum(1))/c[idx].sum(1)
        p=2*min((d<=0).mean(),(d>=0).mean()); p=max(p,1/B)
        res.append(dict(setting=nm,arm=a,user_limit=m0,arm_mean=m1,diff=m1-m0,rel_pct=(m1/m0-1)*100,ci=[float(np.quantile(d,.025)),float(np.quantile(d,.975))],p=p,n_users=len(us),n_jobs=int(ok.sum())))
def holm(items):
    ps=np.array([r["p"] for r in items]); o=np.argsort(ps); m=len(ps); run=0
    for rk,i in enumerate(o):
        run=max(run,min(1,(m-rk)*ps[i])); items[i]["p_holm"]=run
SENS=("bolt_c32","bolt_c192")
for r in res: r["family"]="sensitivity (post hoc)" if r["arm"] in SENS else "planned arms"
holm([r for r in res if r["family"]=="planned arms"]); holm([r for r in res if r["family"]!="planned arms"])
json.dump(res,open(f"{O}/online_results.json","w"),indent=1)
for r in res: print(f'{r["setting"]:11s} {r["arm"]:8s} UL {r["user_limit"]:7.2f} arm {r["arm_mean"]:7.2f} {r["rel_pct"]:+6.1f}% CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f} holm {r["p_holm"]:.4f} [{r["family"][:4]}]')

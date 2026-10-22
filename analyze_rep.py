"""Pre-registered replication: Bolt-192 vs user limits (primary) and Bolt-192 vs Bolt-96 (secondary), 4 settings, Holm over 4 within each test."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, json, os
O=_OUT + "/online"; B=10000
S=[("PM100 420",420),("PM100 640",640),("F-DATA 16.4K",16400),("F-DATA 25K",25000)]
def load(name,n):
    z=np.load(f"{O}/online_{name}_{n}.npz",allow_pickle=True); return z["usr"],z["bs"]
def paired(u,b0,b1,rng):
    ok=~np.isnan(b0)&~np.isnan(b1)
    us,inv=np.unique(u[ok],return_inverse=True)
    s0=np.bincount(inv,b0[ok]);s1=np.bincount(inv,b1[ok]);c=np.bincount(inv).astype(float)
    idx=rng.integers(0,len(us),(B,len(us)))
    d=(s1[idx].sum(1)-s0[idx].sum(1))/c[idx].sum(1)
    p=max(2*min((d<=0).mean(),(d>=0).mean()),1/B)
    return dict(m0=float(s0.sum()/c.sum()),m1=float(s1.sum()/c.sum()),ci=[float(np.quantile(d,.025)),float(np.quantile(d,.975))],p=float(p),n_users=int(len(us)))
def holm(L):
    ps=np.array([r["p"] for r in L]);o=np.argsort(ps);m=len(ps);run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*ps[i])); L[i]["p_holm"]=float(run)
res={"primary":[],"secondary":[]}
for nm,n in S:
    if not os.path.exists(f"{O}/online_bolt_rep192_{n}.npz") or not os.path.exists(f"{O}/online_bolt_rep96_{n}.npz"): continue
    rng=np.random.default_rng(2000+n)
    u,ul=load("validate",n); _,b96=load("bolt_rep96",n); _,b192=load("bolt_rep192",n)
    r=paired(u,ul,b192,rng); r.update(setting=nm,test="Bolt-192 vs user limit"); res["primary"].append(r)
    r=paired(u,b96,b192,rng); r.update(setting=nm,test="Bolt-192 vs Bolt-96"); res["secondary"].append(r)
    r=paired(u,ul,b96,rng); print(f'  (context) {nm}: user limit {r["m0"]:.2f}, Bolt-96 {r["m1"]:.2f}, diff CI [{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f}')
for k in res: holm(res[k])
for k in res:
    for r in res[k]: print(f'{k:9s} {r["setting"]:13s} {r["test"]:22s} {r["m0"]:8.2f} -> {r["m1"]:8.2f} ({(r["m1"]/r["m0"]-1)*100:+6.1f}%) CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f} holm {r["p_holm"]:.4f}')
P=res["primary"]
if len(P)==4:
    lower=all(r["m1"]<r["m0"] for r in P); sig_better=sum(1 for r in P if r["p_holm"]<0.05 and r["m1"]<r["m0"]); sig_worse=sum(1 for r in P if r["p_holm"]<0.05 and r["m1"]>r["m0"])
    verdict="REPLICATES" if (lower and sig_better>=2) else ("FAILS" if (sig_better<2 or sig_worse>0) else "INCONCLUSIVE")
    print("lower in all 4:",lower,"| sig better:",sig_better,"| sig worse:",sig_worse,"| verdict:",verdict)
    res["verdict"]=verdict
json.dump(res,open(f"{O}/replication_results.json","w"),indent=1)

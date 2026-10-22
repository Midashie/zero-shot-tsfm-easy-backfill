"""Pre-registered: Bolt q0.8 vs user limits, 7 settings, paired by-user bootstrap (10,000), Holm over 7. Context: q0.8 vs q0.5 (not in the rule)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_OUT = _os.environ.get("OUT_DIR", _os.path.join(_ROOT, "outputs"))
_IN = _os.environ.get("DATA_DIR", _os.path.join(_ROOT, "data"))
_CENTILE = _os.environ.get("CENTILE_DIR", _os.path.join(_ROOT, "centile"))
import numpy as np, json, os
O=_OUT + "/online"; B=10000
S=[("PM100 420",420,"validate"),("PM100 512",512,"validate"),("PM100 640",640,"validate"),("F-DATA 16.4K",16400,"validate"),("F-DATA 20K",20000,"validate"),("F-DATA 23K",23000,"validate"),("F-DATA 25K",25000,"validate")]
MED={512:"bolt",20000:"bolt",23000:"bolt",420:"bolt_rep96",640:"bolt_rep96",16400:"bolt_rep96",25000:"bolt_rep96"}   # existing median runs, 96 context jobs
def load(name,n):
    z=np.load(f"{O}/online_{name}_{n}.npz",allow_pickle=True); return z["usr"],z["bs"]
def paired(u,b0,b1,rng):
    ok=~np.isnan(b0)&~np.isnan(b1)
    us,inv=np.unique(u[ok],return_inverse=True)
    s0=np.bincount(inv,b0[ok]);s1=np.bincount(inv,b1[ok]);c=np.bincount(inv).astype(float)
    idx=rng.integers(0,len(us),(B,len(us)))
    d=(s1[idx].sum(1)-s0[idx].sum(1))/c[idx].sum(1)
    return dict(m0=float(s0.sum()/c.sum()),m1=float(s1.sum()/c.sum()),ci=[float(np.quantile(d,.025)),float(np.quantile(d,.975))],p=float(max(2*min((d<=0).mean(),(d>=0).mean()),1/B)))
def holm(L):
    ps=np.array([r["p"] for r in L]);o=np.argsort(ps);m=len(ps);run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*ps[i])); L[i]["p_holm"]=float(run)
P,Cx=[],[]
for nm,n,base in S:
    if not os.path.exists(f"{O}/online_bolt_q80_{n}.npz"): continue
    rng=np.random.default_rng(3000+n)
    u,ul=load(base,n); _,q8=load("bolt_q80",n); _,q5=load(MED[n],n)
    r=paired(u,ul,q8,rng); r["setting"]=nm; P.append(r)
    r=paired(u,q5,q8,rng); r["setting"]=nm; Cx.append(r)
holm(P); holm(Cx)
for r in P: print(f'PRIMARY {r["setting"]:13s} UL {r["m0"]:8.2f} -> q0.8 {r["m1"]:8.2f} ({(r["m1"]/r["m0"]-1)*100:+7.1f}%) CI[{r["ci"][0]:+.2f},{r["ci"][1]:+.2f}] p {r["p"]:.4f} holm {r["p_holm"]:.4f}')
for r in Cx: print(f'context {r["setting"]:13s} q0.5 {r["m0"]:8.2f} -> q0.8 {r["m1"]:8.2f} ({(r["m1"]/r["m0"]-1)*100:+7.1f}%) p {r["p"]:.4f} holm {r["p_holm"]:.4f}')
out={"primary":P,"context":Cx}
if len(P)==7:
    win=sum(r["p_holm"]<0.05 and r["m1"]<r["m0"] for r in P); loss=sum(r["p_holm"]<0.05 and r["m1"]>r["m0"] for r in P)
    v="WORKS" if (win>=2 and loss==0) else ("FAILS" if (win==0 or loss>=2) else "INCONCLUSIVE")
    print("significant wins:",win,"| significant losses:",loss,"| verdict:",v); out["verdict"]=v; out["wins"]=win; out["losses"]=loss
json.dump(out,open(f"{O}/q80_results.json","w"),indent=1)

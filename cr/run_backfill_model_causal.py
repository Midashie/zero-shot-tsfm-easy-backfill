"""Completion-causal copy of CENTILE's run_backfill_model.py (pretrain/).

One change in substance: the "previous runtime" input of every job is the runtime of the user's most
recent job that had FINISHED by that job's submit time (recorded end time <= submit time), instead of the
previous job in submission order. Everything else follows the original: model, loss, windows, epochs.

Other differences, all needed for the same reason (no label or input unknown at decision time):
  - the limit covariate is the declared limit; CENTILE's max(limit, runtime) patch is dropped
    (it changes no row in 21_04, checked at load)
  - training targets and the standardisation statistics use only pre-window jobs that had finished by t0
  - inference builds contexts from precomputed completion-causal lags instead of growing a submit-order history
Env: same as the original (LMAX D H L EP MIX BS STRIDE QS SEED OUT SMOKE NFILE DAYS SKIPD MAXJ), plus
REPO (CENTILE checkout) and SHIFT=1 (run the shifted-future test after training).
"""
import os, sys, glob, json, numpy as np, pandas as pd, torch
REPO = os.environ["REPO"]
sys.path.insert(0, os.path.join(REPO, "common"))
from nfm_v2 import HSTUv2, MixtureHead, dev
import torch.nn as nn

NFILE = int(os.environ.get('NFILE', 0)); DAYS = float(os.environ.get('DAYS', 14)); SKIPD = float(os.environ.get('SKIPD', 7))
MAXJ = int(os.environ.get('MAXJ', 300000)); LMAX = int(os.environ.get('LMAX', 96))
D = int(os.environ.get('D', 256)); H = int(os.environ.get('H', 4)); L = int(os.environ.get('L', 4))
EP = int(os.environ.get('EP', 25)); MIX = int(os.environ.get('MIX', 4)); BS = int(os.environ.get('BS', 256))
STRIDE = int(os.environ.get('STRIDE', LMAX // 2))
QS = [float(x) for x in os.environ.get('QS', '0.5,0.7,0.8,0.9,0.95').split(',')]
OUT = os.path.expanduser(os.environ.get('OUT', '~/nfm/data/backfill_est_causal')); os.makedirs(OUT, exist_ok=True)
SMOKE = int(os.environ.get('SMOKE', 0)); SEED = int(os.environ.get('SEED', 0)); SHIFT = int(os.environ.get('SHIFT', 0))
if SMOKE:
    EP = 3; D = 64; L = 2; MAXJ = 20000

lg = lambda a: np.log1p(np.asarray(a, float))
unix = lambda s: pd.to_datetime(s, utc=True, errors='coerce').astype('datetime64[ns, UTC]').astype(np.int64) // 10 ** 9


def load():
    f = sorted(glob.glob(os.path.expanduser("~/nfm/data/fdata/*.parquet")))[NFILE]
    d = pd.read_parquet(f, columns=['usr', 'adt', 'sdt', 'edt', 'nnumr', 'elpl', 'duration'])
    d['adt'] = pd.to_datetime(d['adt'], utc=True, errors='coerce')
    d = d.dropna(subset=['adt']).sort_values('adt')                         # same call as the original
    d['t'] = d['adt'].astype('datetime64[ns, UTC]').astype(np.int64) // 10 ** 9
    d['sd'] = unix(d['sdt']); d['ed'] = unix(d['edt'])
    d['dur'] = pd.to_numeric(d['duration'], errors='coerce').clip(lower=1)
    d['lim'] = pd.to_numeric(d['elpl'], errors='coerce')
    d['nn'] = pd.to_numeric(d['nnumr'], errors='coerce').fillna(1).clip(lower=1).astype(int)
    d = d.dropna(subset=['dur', 'lim']).reset_index(drop=True)
    print(f"rows where runtime exceeds declared limit (CENTILE would patch these): {int((d['dur'] > d['lim']).sum())}", flush=True)
    t0 = d['t'].iloc[0] + SKIPD * 86400
    a = int(np.searchsorted(d['t'].values, t0, 'left'))
    n = int(((d['t'].values >= t0) & (d['t'].values < t0 + DAYS * 86400)).sum()); n = min(n, MAXJ)
    return d, a, n, t0, os.path.basename(f)


def causal_lag(d):
    """log1p runtime of the user's most recent job finished before each row's submit time; 0 when none."""
    lag = np.zeros(len(d), np.float32)
    T, ED, DU = d['t'].values, d['ed'].values, d['dur'].values
    for u, idx in d.groupby('usr').indices.items():
        t, ed, du = T[idx], ED[idx], DU[idx]
        o = np.argsort(ed, kind='stable'); eds, ts_, dus = ed[o], t[o], du[o]
        k = np.searchsorted(eds, t, side='right')
        for j in range(len(idx)):
            kk = int(k[j])
            while kk > 0 and ts_[kk - 1] >= t[j]: kk -= 1                   # only jobs submitted strictly earlier
            if kk > 0: lag[idx[j]] = np.log1p(dus[kk - 1])
    return lag


class TriInput(nn.Module):
    def __init__(self, D): super().__init__(); self.lin = nn.Linear(3, D)
    def forward(self, inp): return self.lin(inp.float())


def make(N):
    return HSTUv2(TriInput(D), MixtureHead(D, MIX), D=D, H=H, L=L, N=N)


def build_train(pre, lag, t0):
    X = []; Y = []; V = []
    for u, idx in pre.groupby('usr').indices.items():
        g = pre.iloc[idx]
        dur = lg(g['dur'].values); nnf = lg(g['nn'].values); limf = lg(g['lim'].values); lagv = lag[idx]
        ok = (g['ed'].values <= t0).astype(np.float32)                      # label known at the replay start
        m = len(g)
        for s in range(0, max(1, m - 1), STRIDE):
            e = min(s + LMAX, m); T = e - s
            xin = np.zeros((LMAX, 3), np.float32); xin[:T, 0] = lagv[s:e]; xin[:T, 1] = nnf[s:e]; xin[:T, 2] = limf[s:e]
            y = np.zeros(LMAX, np.float32); y[:T] = dur[s:e]; v = np.zeros(LMAX, np.float32); v[:T] = ok[s:e]
            X.append(xin); Y.append(y); V.append(v)
            if e >= m: break
    return np.stack(X), np.stack(Y), np.stack(V)


def train(X, Y, V, mu, sd, N):
    m = make(N).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01); nb = len(X)
    Yz = (Y - mu) / sd; ts = torch.zeros(BS, N, dtype=torch.long, device=dev)
    Xt = torch.tensor(X); Yt = torch.tensor(Yz); Vt = torch.tensor(V)
    vm = Vt.reshape(-1).bool(); flat = Xt.reshape(-1, 3)[vm]
    cmu = flat.mean(0); csd = flat.std(0) + 1e-6
    for ep in range(EP):
        m.train(); idx = np.random.permutation(nb)
        for i in range(0, nb, BS):
            j = idx[i:i + BS]; x = ((Xt[j] - cmu) / csd).to(dev); y = Yt[j].to(dev); v = Vt[j].to(dev); t = ts[:len(j)]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
                hid = m(x, t)
            loss = m.head.nll(hid.float(), y, v)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); return m, cmu.to(dev), csd.to(dev)


@torch.no_grad()
def infer(m, cmu, csd, d, lag, rows, mu, sd, N, seed):
    """Score the window rows `rows` (positions in d). Context = the user's previous N-1 submitted jobs plus the current one."""
    torch.manual_seed(seed)                                                  # quantiles are sampled from the mixture
    users = d.groupby('usr').indices
    ua = d['usr'].values; nnf = lg(d['nn'].values); limf = lg(d['lim'].values); limraw = d['lim'].values
    est = np.zeros((len(rows), len(QS)), np.float32)
    for b0 in range(0, len(rows), BS):
        chunk = rows[b0:b0 + BS]; X = np.zeros((len(chunk), N, 3), np.float32); ridx = []
        for bi, gi in enumerate(chunk):
            arr = users[ua[gi]]; p = int(np.searchsorted(arr, gi))
            seq = np.concatenate([arr[max(0, p - (N - 1)):p], [gi]]); k = len(seq) - 1
            X[bi, :k + 1, 0] = lag[seq]; X[bi, :k + 1, 1] = nnf[seq]; X[bi, :k + 1, 2] = limf[seq]; ridx.append(k)
        x = ((torch.tensor(X) - cmu.cpu()) / csd.cpu()).to(dev); t = torch.zeros(len(chunk), N, dtype=torch.long, device=dev)
        hid = m(x, t).float()
        hr = hid[torch.arange(len(chunk)), torch.tensor(ridx, device=dev)]
        qz = m.head.quantiles(hr.unsqueeze(1), tuple(QS), S=2048)[:, 0, :].cpu().numpy()
        sec = np.expm1(qz * sd + mu)
        for bi, gi in enumerate(chunk): est[b0 + bi] = np.clip(sec[bi], 60.0, max(60.0, limraw[gi]))
    return est


def shifted_future_test(m, cmu, csd, d, lag, a, n, t0, mu, sd, N):
    tau = t0 + 7 * 86400
    rows_all = np.arange(a, a + n); rows = rows_all[d['t'].values[rows_all] <= tau]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(rows, min(6000, len(rows)), replace=False))
    d2 = d.copy(); fut = d2['ed'].values > tau
    new_ed = np.maximum(d2['ed'].values[fut], tau + 1) + rng.integers(1, 7 * 86400, size=int(fut.sum()))
    ed2 = d2['ed'].values.copy(); ed2[fut] = new_ed; du2 = d2['dur'].values.copy(); du2[fut] = np.maximum(new_ed - d2['sd'].values[fut], 1)
    d2['ed'] = ed2; d2['dur'] = du2
    lag2 = causal_lag(d2)
    past = d['t'].values <= tau
    e1 = infer(m, cmu, csd, d, lag, rows, mu, sd, N, SEED + 7)
    e2 = infer(m, cmu, csd, d2, lag2, rows, mu, sd, N, SEED + 7)
    return {"tau_utc_unix": int(tau), "rows_scored": int(len(rows)), "rows_perturbed_in_file": int(fut.sum()),
            "max_abs_lag_change_rows_submitted_at_or_before_tau": float(np.abs(lag[past] - lag2[past]).max()),
            "max_abs_lag_change_rows_submitted_after_tau": float(np.abs(lag[~past] - lag2[~past]).max()),
            "estimates_changed": int((np.abs(e1 - e2) > 0).any(axis=1).sum()), "max_abs_estimate_change": float(np.abs(e1 - e2).max())}


if __name__ == '__main__':
    np.random.seed(SEED); torch.manual_seed(SEED)
    d, a, n, t0, fn = load()
    lag = causal_lag(d)
    rows = np.arange(a, a + n)
    pre = d.iloc[:a]
    done = pre['ed'].values <= t0
    print(f"eval file={fn} pre(train)={a:,} of which finished by t0={int(done.sum()):,} window(eval)={n:,} LMAX={LMAX} D={D} L={L} EP={EP} SEED={SEED} QS={QS}", flush=True)
    a_ = lg(pre['dur'].values[done]); mu, sd = float(a_.mean()), float(a_.std() + 1e-6)
    print(f"log1p(dur) train stats mu={mu:.3f} sd={sd:.3f}", flush=True)
    X, Y, V = build_train(pre, lag[:a], t0)
    print(f"train windows={len(X):,}", flush=True)
    m, cmu, csd = train(X, Y, V, mu, sd, LMAX)
    est = infer(m, cmu, csd, d, lag, rows, mu, sd, LMAX, SEED + 1)
    for qi, q in enumerate(QS):
        p = os.path.join(OUT, f"hstu_q{int(round(q * 100))}.npz"); np.savez(p, est=est[:, qi])
        e = est[:, qi]; print(f"  q{int(round(q * 100))}: est sec median={np.median(e):.0f} p90={np.quantile(e, 0.9):.0f} -> {p}", flush=True)
    tru = d['dur'].values[rows]
    for qi, q in enumerate(QS):
        e = est[:, qi]; print(f"  q{int(round(q * 100))} coverage(P[est>=dur])={(e >= tru).mean():.3f} median est/dur={np.median(e / np.maximum(tru, 1)):.2f}", flush=True)
    if SHIFT:
        r = shifted_future_test(m, cmu, csd, d, lag, a, n, t0, mu, sd, LMAX)
        json.dump(r, open(os.path.join(OUT, "shifted_future_test.json"), "w"), indent=2); print("SHIFT_TEST", json.dumps(r), flush=True)
    print("BACKFILL_MODEL_DONE", flush=True)

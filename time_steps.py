import os, sys, time
os.environ.update(dict(LMAX='96', D='256', H='8', L='4', EP='1', MIX='6', STRIDE='24', ARM='HSTU', OUT='/tmp/timing_out'))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import run_backfill_model as M
print('device', M.dev, 'threads', torch.get_num_threads())
d, w, pre, t0, fn = M.load()
t = time.time(); X, Y, V = M.build_train(pre, None); print('train sequences', X.shape, f'build {time.time()-t:.1f}s')
mu, sd = M.standardize(pre)
m = M.make('HSTU', M.LMAX).to(M.dev); print('params', sum(p.numel() for p in m.parameters()))
opt = torch.optim.AdamW(m.parameters(), 1e-3)
Xt = torch.tensor(X[:M.BS]); Yt = torch.tensor((Y[:M.BS]-mu)/sd); Vt = torch.tensor(V[:M.BS]); ts = torch.zeros(len(Xt), M.LMAX, dtype=torch.long)
flat = Xt.reshape(-1,3)[Vt.reshape(-1).bool()]; cmu, csd = flat.mean(0), flat.std(0)+1e-6
times = []
for k in range(4):
    t = time.time(); hid = m((Xt-cmu)/csd, ts); loss = m.head.nll(hid.float(), Yt, Vt); opt.zero_grad(); loss.backward(); opt.step(); times.append(time.time()-t)
print('step seconds', [round(x,2) for x in times], 'batch', len(Xt), 'steps/epoch', -(-len(X)//M.BS))

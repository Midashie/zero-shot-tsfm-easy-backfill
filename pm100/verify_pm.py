"""Shifted-future test for the PM100 contexts. Usage: python3 -I verify_pm.py CONTEXT_NPZ OUT_JSON
For 4,000 sampled replay jobs submitted at s, every same-user job unfinished at s gets a later end time and a new runtime;
the context rebuilt with the perturbed arrays must not change in rows or runtime values, and must equal the saved context."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pm_common import load_pm

ctx_npz, out = sys.argv[1:3]
C = 96
d = load_pm(); ts, sd, ed, dur, usr = d["ts"], d["sd"], d["ed"], d["dur"], d["usr"]
Z = np.load(ctx_npz); rows = Z["win_rows"]
rng = np.random.default_rng(0)
sample = rng.choice(len(rows), 4000, replace=False)
groups = {}
idx = np.argsort(usr, kind="stable"); us = usr[idx]
cuts = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
for a, b in zip(cuts[:-1], cuts[1:]): groups[us[a]] = idx[a:b]


def ctx(ids, ed_, t_i):
    o = ids[np.argsort(ed_[ids], kind="stable")]
    k = int(np.searchsorted(ed_[o], t_i, side="right"))
    o = o[:k]; o = o[ts[o] < t_i]
    return o[-C:]


cr = cv = tot = 0
for i in sample:
    rw = rows[i]; t_i = ts[rw]; ids = groups[usr[rw]]
    base = ctx(ids, ed, t_i)
    ed2, dur2 = ed.copy(), dur.copy()
    unf = ids[ed[ids] > t_i]; tot += len(unf)
    if len(unf):
        ne = np.maximum(ed[unf], t_i + 1) + rng.integers(1, 7 * 86400, size=len(unf))
        ed2[unf] = ne; dur2[unf] = np.maximum(ne - sd[unf], 1)
    pert = ctx(ids, ed2, t_i)
    cr += not np.array_equal(base, pert); cv += not np.array_equal(dur[base], dur2[pert])
    saved = Z["win_ctx"][i]; saved = saved[saved >= 0]
    assert np.array_equal(saved, base), "saved context differs from rebuilt context"
res = {"sampled_jobs": 4000, "unfinished_same_user_jobs_perturbed": int(tot), "contexts_with_changed_rows": int(cr),
       "contexts_with_changed_runtime_values": int(cv), "saved_contexts_match_rebuild": True}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))

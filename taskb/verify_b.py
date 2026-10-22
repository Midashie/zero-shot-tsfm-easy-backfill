"""Shifted-future test for the Task B contexts (G3 check on the new forecast inputs).
For a random sample of replayed jobs submitted at time s, pick tau = s. Every job of any user that was NOT yet finished at tau
(recorded end time after tau) gets a new, later end time and a new runtime. The context of each sampled job (jobs finished by
its own submit time, rebuilt with the perturbed arrays) must not change, in rows or in runtime values.
Usage: python3 -I verify_b.py PARQUET MANIFEST_CSV CONTEXT_NPZ OUT_JSON
"""
import sys, json
import numpy as np, pandas as pd

pq, man_csv, ctx_npz, out = sys.argv[1:5]
C = 96
raw = pd.read_parquet(pq, columns=["usr", "adt", "sdt", "edt", "duration"])
u = lambda s: pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]").astype(np.int64) // 10 ** 9
ts, sd, ed = u(raw["adt"]).values, u(raw["sdt"]).values, u(raw["edt"]).values
dur = pd.to_numeric(raw["duration"], errors="coerce").clip(lower=1).values.astype(np.float64)
usr = raw["usr"].values
Z = np.load(ctx_npz)
rows = Z["win_rows"]
rng = np.random.default_rng(0)
sample = rng.choice(len(rows), 4000, replace=False)
by_user = {}
for k, name in enumerate(pd.unique(usr)):
    pass
groups = pd.Series(np.arange(len(raw))).groupby(usr).apply(lambda s: s.values).to_dict()


def ctx(urow_ids, ed_, dur_, t_i):
    o = urow_ids[np.argsort(ed_[urow_ids], kind="stable")]
    k = int(np.searchsorted(ed_[o], t_i, side="right"))
    o = o[:k]
    o = o[ts[o] < t_i]
    return o[-C:]


changed_rows = changed_vals = 0
n_perturbed_total = 0
for i in sample:
    rw = rows[i]
    t_i = ts[rw]
    ids = groups[usr[rw]]
    base = ctx(ids, ed, dur, t_i)
    ed2, dur2 = ed.copy(), dur.copy()
    unfinished = ids[ed[ids] > t_i]
    n_perturbed_total += len(unfinished)
    if len(unfinished):
        new_ed = np.maximum(ed[unfinished], t_i + 1) + rng.integers(1, 7 * 86400, size=len(unfinished))
        ed2[unfinished] = new_ed
        dur2[unfinished] = np.maximum(new_ed - sd[unfinished], 1)
    pert = ctx(ids, ed2, dur2, t_i)
    changed_rows += not np.array_equal(base, pert)
    changed_vals += not np.array_equal(dur[base], dur2[pert])
    # and the saved context matches the rebuilt one
    saved = Z["win_ctx"][i]
    saved = saved[saved >= 0]
    assert np.array_equal(saved, base), "saved context differs from rebuilt context"
res = {"sampled_jobs": int(len(sample)), "jobs_with_unfinished_same_user_jobs_perturbed": int(n_perturbed_total),
       "contexts_with_changed_rows": int(changed_rows), "contexts_with_changed_runtime_values": int(changed_vals),
       "saved_contexts_match_rebuild": True,
       "perturbation": "all same-user jobs unfinished at the submit time get a later end time and a new runtime"}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))

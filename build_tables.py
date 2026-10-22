import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
O = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results", "online"))
theta = {y: json.load(open(f"{O}/theta{'' if y == 2017 else y}_results.json")) for y in (2017, 2018, 2019, 2020)}
swb = json.load(open(f"{O}/swb_results.json"))
enf = json.load(open(f"{O}/enf_results.json"))
SIZES = {2017: [3600, 4392, 5500], 2018: [4200, 5092, 6400], 2019: [4500, 5492, 6900], 2020: [5300, 6492, 8100]}


def pct(r):
    return (r["m1"] / r["m0"] - 1) * 100


def cell(r):
    s = f"{pct(r):+.1f}%"
    return s + ("*" if r["p_holm"] < 0.05 else "")


# Table: Theta hint-only Bolt
rows = ["| Window | Nodes | User limit | Bolt-96 | Bolt-192 |", "| --- | --- | --- | --- | --- |"]
for y in (2017, 2018, 2019, 2020):
    P = theta[y]["primary"]
    for n in SIZES[y]:
        a = [r for r in P if r["size"] == n and r["ctx"] == 96][0]
        b = [r for r in P if r["size"] == n and r["ctx"] == 192][0]
        rows.append(f"| {y} | {n:,} | {a['m0']:.2f} | {a['m1']:.2f} ({cell(a)}) | {b['m1']:.2f} ({cell(b)}) |")
T_THETA = "\n".join(rows)

# Table: SafeWall-style
rows = ["| Window | Nodes | User limit | SafeWall-style | Unbuffered XGBoost |", "| --- | --- | --- | --- | --- |"]
for y in (2017, 2018, 2019, 2020):
    P = swb[str(y)]["primary"]; U = swb[str(y)]["xgb_unbuffered"]
    for n in SIZES[y]:
        a = [r for r in P if r["size"] == n][0]; b = [r for r in U if r["size"] == n][0]
        rows.append(f"| {y} | {n:,} | {a['m0']:.2f} | {a['m1']:.2f} ({cell(a)}) | {b['m1']:.2f} ({cell(b)}) |")
T_SWB = "\n".join(rows)

# Table: kill-and-requeue
rows = ["| Window | Nodes | User limit | Bolt-96 median | Bolt-96 0.8 quantile | SafeWall-style |", "| --- | --- | --- | --- | --- | --- |"]
for y in (2017, 2018, 2019, 2020):
    P = enf[str(y)]["primary"]
    for n in SIZES[y]:
        d = {r["arm"]: r for r in P if r["size"] == n}
        m0 = d["bolt_median"]["m0"]
        rows.append(f"| {y} | {n:,} | {m0:.2f} | {d['bolt_median']['m1']:.2f} ({cell(d['bolt_median'])}) | {d['bolt_q80']['m1']:.2f} ({cell(d['bolt_q80'])}) | {d['swb']['m1']:.2f} ({cell(d['swb'])}) |")
T_ENF = "\n".join(rows)

# counts used in the text (computed, not typed)
def count(P):
    g = sum(1 for r in P if r["p_holm"] < .05 and r["m1"] < r["m0"])
    l = sum(1 for r in P if r["p_holm"] < .05 and r["m1"] > r["m0"])
    return g, l, len(P) - g - l
bg = bl = bn = 0
for y in theta:
    g, l, n = count(theta[y]["primary"]); bg += g; bl += l; bn += n
sg = sl = sn = 0
for y in swb:
    g, l, n = count(swb[y]["primary"]); sg += g; sl += l; sn += n
eg = el = en = 0
for y in enf:
    g, l, n = count(enf[y]["primary"]); eg += g; el += l; en += n
assert (bg, bl, bn) == (4, 11, 9), (bg, bl, bn)
assert (sg, sl, sn) == (0, 9, 3), (sg, sl, sn)
assert (eg, el, en) == (0, 32, 4), (eg, el, en)
e17 = {(r["arm"], r["size"]): r for r in enf["2017"]["primary"]}
assert round(pct(e17[("bolt_median", 4392)])) == 97 and round(pct(e17[("bolt_median", 5500)])) == 25

tables = {"Table 5 (Theta, hint-only Bolt)": T_THETA, "Table 6 (SafeWall-style)": T_SWB, "Table 7 (kill-and-requeue)": T_ENF}
out = os.path.join(HERE, "tables.md")
open(out, "w").write("\n\n".join(f"### {k}\n\n{v}" for k, v in tables.items()) + "\n")
print("ok, wrote", out)

"""Per-condition accuracy with bootstrap 95% CI, cut share, loop share (authors' mathqa.loop_signals), mean tokens,
on the problems every finished condition shares; then exact McNemar tests for paired condition pairs."""
import glob, gzip, json, os, random, sys
from math import comb
sys.path[:0] = [os.path.expanduser("~/wsp/logs/exp8_math/code"), os.path.expanduser("~/wsp/scripts/shim")]
import mathqa as Q
R = os.path.expanduser("~/wsp/results/math500")
data = {}
for d in sorted(glob.glob(R + "/*/")):
    rs = {}
    for f in sorted(glob.glob(d + "chunk_*.jsonl.gz")):
        for l in gzip.open(f, "rt"):
            r = json.loads(l); s = Q.loop_signals(r["text"])
            rs[r["id"]] = (r["correct"], r["finish"] == "length", int(s["verbatim_loop_end"] or s["repetitive_end"]), r["ntok"])
    if rs: data[os.path.basename(d.rstrip("/"))] = rs
ids = sorted(set.intersection(*[set(v) for v in data.values()]))
print("problems shared by all conditions:", len(ids))
random.seed(0)
def ci(xs):
    bs = sorted(sum(random.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
    return bs[50], bs[1949]
print(f"{'condition':<12}{'acc':>7}{'95% CI':>17}{'cut':>7}{'loop':>7}{'tokens':>8}")
for k, v in data.items():
    xs = [v[i][0] for i in ids]; lo, hi = ci(xs); n = len(ids)
    print(f"{k:<12}{sum(xs)/n:>7.3f}   [{lo:.3f}, {hi:.3f}]{sum(v[i][1] for i in ids)/n:>7.3f}{sum(v[i][2] for i in ids)/n:>7.3f}{sum(v[i][3] for i in ids)/n:>8.0f}")
def mcnemar(a, b):
    gain = sum(1 for i in ids if not a[i][0] and b[i][0]); loss = sum(1 for i in ids if a[i][0] and not b[i][0])
    n, k = gain + loss, min(gain, loss)
    return gain, loss, (min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0)
for x, y in [("base", "p50c"), ("base", "base_rp110"), ("base_rp110", "p50c")] + [("p50c", c) for c in data if c.startswith("p50c_")]:
    if x in data and y in data:
        g, l, p = mcnemar(data[x], data[y]); print(f"{x} -> {y}: {g} problems gained, {l} lost, exact McNemar p = {p:.2g}")

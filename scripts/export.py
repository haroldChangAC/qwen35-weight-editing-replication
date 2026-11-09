"""Per-problem table (export/per_problem.csv) + compact printouts for Fig 4 (outcome shares) and Fig 5 (token CDF)."""
import csv, glob, gzip, json, os, sys
sys.path[:0] = [os.path.expanduser("~/wsp/logs/exp8_math/code"), os.path.expanduser("~/wsp/scripts/shim")]
import mathqa as Q
R = os.path.expanduser("~/wsp/results/math500"); rows = []
for d in sorted(glob.glob(R + "/*/")):
    c = os.path.basename(d.rstrip("/"))
    for f in sorted(glob.glob(d + "chunk_*.jsonl.gz")):
        for l in gzip.open(f, "rt"):
            r = json.loads(l); s = Q.loop_signals(r["text"]); cut = r["finish"] == "length"; loop = int(s["verbatim_loop_end"] or s["repetitive_end"])
            o = "correct" if r["correct"] else ("cut_loop" if cut and loop else "cut_noloop" if cut else "no_answer" if r["pred"] is None else "wrong")
            rows.append({"cond": c, "id": r["id"], "correct": r["correct"], "finish": r["finish"], "ntok": r["ntok"], "loop": loop, "outcome": o})
with open("export/per_problem.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
conds = sorted({r["cond"] for r in rows}); outs = ["correct", "wrong", "no_answer", "cut_noloop", "cut_loop"]
print("FIG4", ",".join(outs))
for c in conds:
    print(c, ",".join(str(sum(1 for r in rows if r["cond"] == c and r["outcome"] == o)) for o in outs))
print("FIG5 percentiles 0,5,...,100 of ntok")
for c in ["base", "base_rp110", "p50c", "p50c_rp110"]:
    t = sorted(r["ntok"] for r in rows if r["cond"] == c)
    print(c, ",".join(str(t[min(len(t) - 1, round(p / 100 * (len(t) - 1)))]) for p in range(0, 101, 5)))

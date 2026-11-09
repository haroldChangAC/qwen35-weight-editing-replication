"""Table of every finished condition: problems, accuracy, share cut at the token limit, mean tokens."""
import glob, gzip, json, os
R = os.path.expanduser("~/wsp/results/math500")
print(f"{'condition':<14}{'n':>5}{'acc':>8}{'cut':>8}{'no_ans':>8}{'mean_tok':>10}")
for d in sorted(glob.glob(f"{R}/*/")):
    rs = [json.loads(l) for f in sorted(glob.glob(f"{d}chunk_*.jsonl.gz")) for l in gzip.open(f, "rt")]
    if not rs: continue
    n = len(rs)
    acc = sum(r["correct"] for r in rs) / n
    cut = sum(r["finish"] == "length" for r in rs) / n
    noans = sum(r["pred"] is None and r["finish"] != "length" for r in rs) / n
    print(f"{os.path.basename(d.rstrip('/')):<14}{n:>5}{acc:>8.3f}{cut:>8.3f}{noans:>8.3f}{sum(r['ntok'] for r in rs)/n:>10.0f}")

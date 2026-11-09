"""Compact matrices for the richer figures: level/subject accuracy, solved-set overlap, outcome transitions, exact token quantiles."""
import csv, collections, json, os
from datasets import load_dataset
meta = {r["unique_id"]: (int(r["level"]), r["subject"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
rows = list(csv.DictReader(open("export/per_problem.csv")))
C = ["base","base_rp105","base_rp110","base_rp120","p10c","p25c","p50c","p50c_a050","p50c_a150","p50c_rp110"]
by = collections.defaultdict(dict)
for r in rows: by[r["cond"]][r["id"]] = r
ids = sorted(by["base"]); out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
levels = [1,2,3,4,5]; subj = sorted({v[1] for v in meta.values()})
P("NLEVEL", *[sum(1 for i in ids if meta[i][0]==L) for L in levels])
for c in C: P("LEVEL", c, *[sum(int(by[c][i]["correct"]) for i in ids if meta[i][0]==L) for L in levels])
P("SUBJ", *[s.replace(" ","_") for s in subj]); P("NSUBJ", *[sum(1 for i in ids if meta[i][1]==s) for s in subj])
for c in C: P("SUBJACC", c, *[sum(int(by[c][i]["correct"]) for i in ids if meta[i][1]==s) for s in subj])
S = {c: {i for i in ids if by[c][i]["correct"]=="1"} for c in C}
for a in C: P("OVL", a, *[len(S[a] & S[b]) for b in C])
P("ANY", len(set.union(*S.values())), "NONE", len(ids)-len(set.union(*S.values())), "ALL", len(set.intersection(*S.values())))
dec = S["base"]|S["base_rp105"]|S["base_rp110"]|S["base_rp120"]
P("P50C_ONLY_VS_DECODING", len(S["p50c"]-dec), "DECODING_ONLY_VS_P50C", len(dec-S["p50c"]))
O = ["correct","wrong","no_answer","cut_noloop","cut_loop"]
for a, b in [("base","p50c"),("base","base_rp110"),("p50c","p50c_rp110"),("base_rp110","p50c")]:
    m = collections.Counter((by[a][i]["outcome"], by[b][i]["outcome"]) for i in ids)
    for oa in O: P("TRANS", a, b, oa, *[m[(oa, ob)] for ob in O])
for c in ["base","base_rp110","p50c","p50c_rp110"]:
    t = sorted(int(by[c][i]["ntok"]) for i in ids); P("Q", c, *[t[min(499, round(p/100*499))] for p in range(0,101,2)])
open("export/fig_data.txt","w").write("\n".join(out)+"\n"); print("LINES", len(out))

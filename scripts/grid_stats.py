import csv, collections, random
from math import comb
from datasets import load_dataset
meta = {r['unique_id']: int(r['level']) for r in load_dataset('HuggingFaceH4/MATH-500', split='test')}
by = collections.defaultdict(dict)
for r in csv.DictReader(open('export/per_problem.csv')): by[r['cond']][r['id']] = r
ids = sorted(by['base']); n = len(ids)
V = {c: [int(by[c][i]['correct']) for i in ids] for c in by}
def ci(xs):
    random.seed(0); bs = sorted(sum(random.choice(xs) for _ in xs)/len(xs) for _ in range(2000)); return bs[50], bs[1949]
print('ALL cond acc lo hi cut loop noans tok L1..L5')
for c in sorted(by):
    d = by[c]; lo, hi = ci(V[c])
    cut = sum(d[i]['finish']=='length' for i in ids)/n; lp = sum(int(d[i]['loop']) for i in ids)/n
    na = sum(d[i]['outcome']=='no_answer' for i in ids)/n; tk = sum(int(d[i]['ntok']) for i in ids)/n
    lv = [sum(V[c][k] for k,i in enumerate(ids) if meta[i]==L) for L in range(1,6)]
    print('ALL', c, '%.3f %.3f %.3f %.3f %.3f %.3f %.0f' % (sum(V[c])/n, lo, hi, cut, lp, na, tk), *lv)
G = [['base','base_rp105','base_rp110'],['p50c_a050','p50c_a050_rp105','p50c_a050_rp110'],['p50c','p50c_rp105','p50c_rp110'],['p50c_a150','p50c_a150_rp105','p50c_a150_rp110']]
def mc(a, b):
    g = sum(1 for k in range(n) if not V[a][k] and V[b][k]); l = sum(1 for k in range(n) if V[a][k] and not V[b][k]); m = g + l
    return g, l, (min(1.0, 2*sum(comb(m, j) for j in range(min(g, l)+1))/2**m) if m else 1.0)
for row in G:
    for c in row[1:]: print('MC', row[0], c, *mc(row[0], c))
for j in (1, 2):
    for r in G[1:]: print('MC', G[0][j], r[j], *mc(G[0][j], r[j]))
random.seed(0); B = [[random.randrange(n) for _ in range(n)] for _ in range(2000)]
def boot(f):
    est = f(range(n)); bs = sorted(f(s) for s in B); return est, bs[50], bs[1949]
acc = lambda c, s: sum(V[c][k] for k in s)/len(s)
for a in range(4):
    for j in (1, 2):
        print('DRP', a, j, '%.3f %.3f %.3f' % boot(lambda s: acc(G[a][j], s) - acc(G[a][0], s)))
        if a: print('INT', a, j, '%.3f %.3f %.3f' % boot(lambda s: acc(G[a][j], s) - acc(G[a][0], s) - acc(G[0][j], s) + acc(G[0][0], s)))
U = [any(V[c][k] for row in G for c in row) for k in range(n)]; print('UNION', sum(U))

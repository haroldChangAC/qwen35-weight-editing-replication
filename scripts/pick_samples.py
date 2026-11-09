import csv, gzip, json, glob, random, collections
P = collections.defaultdict(dict)
for r in csv.DictReader(open('export/per_problem.csv')): P[r['cond']][r['id']] = r
Q = {json.loads(l)['id']: json.loads(l) for l in open('bench/math500.jsonl')}
def load(c):
    d = {}
    for f in sorted(glob.glob('results/math500/%s/chunk_*.jsonl.gz' % c)):
        for l in gzip.open(f, 'rt'):
            r = json.loads(l); d[r['id']] = r
    return d
o = lambda c, i: P[c][i]['outcome']
ids = sorted(P['base'])
CATS = [('A base ends in loop', 'base', lambda i: o('base', i) == 'cut_loop', 3),
 ('B fixed by edit', 'p50c', lambda i: o('p50c', i) == 'correct' and o('base', i) == 'cut_loop', 3),
 ('C edit runs to limit without loop', 'p50c', lambda i: o('p50c', i) == 'cut_noloop', 3),
 ('D edit finishes without answer', 'p50c', lambda i: o('p50c', i) == 'no_answer', 3),
 ('E edit wrong answer', 'p50c', lambda i: o('p50c', i) == 'wrong', 2),
 ('F penalty pushes solved problem to limit', 'p50c_rp110', lambda i: o('p50c_rp110', i) == 'cut_noloop' and o('p50c', i) == 'correct', 3),
 ('G alpha 1.5 no answer', 'p50c_a150', lambda i: o('p50c_a150', i) == 'no_answer', 2),
 ('H original RP 1.20 no answer', 'base_rp120', lambda i: o('base_rp120', i) == 'no_answer', 1)]
random.seed(0); out = open('export/samples_full.jsonl', 'w'); view = open('export/samples_view.txt', 'w'); k = 0
for lab, c, f, n in CATS:
    cand = [i for i in ids if f(i)]; D = load(c)
    for i in random.sample(cand, n):
        k += 1; r = D[i]; t = r['text']
        out.write(json.dumps({'k': k, 'cat': lab, 'cond': c, 'id': i, 'ncand': len(cand), 'problem': Q[i]['problem'], **r}) + '\n')
        view.write('#%d [%s] %s %s (pool %d) gold=%s pred=%s ntok=%s chars=%d\n' % (k, lab, c, i, len(cand), r['gold'], r['pred'], r['ntok'], len(t)))
        view.write('Q: ' + Q[i]['problem'][:160].replace('\n', ' ') + '\n')
        view.write('END: ' + t[-420:].replace('\n', ' / ') + '\n\n')
print('picked', k)

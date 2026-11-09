import csv, gzip, json, glob, re, sys, os, collections
sys.path[:0] = [os.path.expanduser('~/wsp/logs/exp8_math/code'), os.path.expanduser('~/wsp/scripts/shim')]
import mathqa as Q
P = collections.defaultdict(dict)
for r in csv.DictReader(open('export/per_problem.csv')): P[r['cond']][r['id']] = r
RX = re.compile(r'(?:final answer|answer)\s*(?:is|=)?\s*:?\s*\**\s*\$*\s*([^\n$]+?)\s*\$*\s*\.?\s*$', re.I | re.M)
print('cond noans rescued spaces50 rescued_ids')
for c in sorted(P):
    na = res = ws = 0; ex = []
    for f in sorted(glob.glob('results/math500/%s/chunk_*.jsonl.gz' % c)):
        for l in gzip.open(f, 'rt'):
            r = json.loads(l); t = r['text']
            if t and t.count(' ') > 0.5 * len(t): ws += 1
            if P[c][r['id']]['outcome'] != 'no_answer': continue
            na += 1; m = RX.findall(t[-600:])
            if m and Q.is_correct(m[-1].strip(), r['gold']): res += 1; ex.append(r['id'].split('/')[-2][:4] + '/' + r['id'].split('/')[-1][:-5])
    print(c, na, res, ws, ' '.join(ex[:4]))

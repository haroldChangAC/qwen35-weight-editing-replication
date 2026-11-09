import csv, collections
from datasets import load_dataset
meta = {r['unique_id']: (int(r['level']), r['subject']) for r in load_dataset('HuggingFaceH4/MATH-500', split='test')}
rows = list(csv.DictReader(open('export/per_problem.csv')))
by = collections.defaultdict(dict)
for r in rows: by[r['cond']][r['id']] = r
ids = sorted(by['base']); subj = sorted({v[1] for v in meta.values()})
code = {'correct':'C','wrong':'W','no_answer':'N','cut_noloop':'c','cut_loop':'L'}
out = open('export/grid_dump.txt','w')
out.write('NIDS %d\n' % len(ids))
out.write('LV ' + ''.join(str(meta[i][0]) for i in ids) + '\n')
out.write('SJ ' + ''.join(str(subj.index(meta[i][1])) for i in ids) + '\n')
for c in sorted(by):
    d = by[c]; assert set(d) == set(ids), c
    t = [int(d[i]['ntok']) for i in ids]
    out.write('S %s %.1f %s\n' % (c, sum(t)/len(t), ''.join(code[d[i]['outcome']] for i in ids)))
out.close(); print('ok')

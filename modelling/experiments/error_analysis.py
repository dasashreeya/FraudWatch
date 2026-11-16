"""Where does the saved best model (artifacts/, see metadata.json) miss fraud on VALIDATION? Guides feature work.

Run from modelling/:  python experiments/error_analysis.py
"""
import json

import numpy as np
import pandas as pd

import lab
from data import LABEL
from models import load_model

t = lab.base_table()
val = t[t.part == 'val'].copy()
best = json.loads((lab.ROOT / 'artifacts' / 'metadata.json').read_text())['best_model']
model = load_model(lab.ROOT / 'artifacts' / best)
print(f'model: {best}')
val['p'] = model.predict_proba(val)
thr = model.threshold
val['caught'] = val.p >= thr

# position of each fraud transaction within its card's fraud burst (over the whole timeline)
fr = t.loc[t[LABEL] == 1, ['cc_num', 'ts']].copy()
fr['pos'] = fr.groupby('cc_num').cumcount() + 1
fr['burst_len'] = fr.groupby('cc_num').ts.transform('size')
fr['h_since_start'] = (fr.ts - fr.groupby('cc_num').ts.transform('min')) / 3600
val = val.join(fr[['pos', 'burst_len', 'h_since_start']])
f = val[val[LABEL] == 1]
print(f'threshold {thr:.3f} | val fraud {len(f)} | caught {f.caught.sum()} ({f.caught.mean():.1%}) | '
      f'false positives {(val.caught & (val[LABEL] == 0)).sum()}')
print(f'val PR-AUC {lab.average_precision_score(val[LABEL], val.p):.4f}\n')


def show(by, name):
    g = f.groupby(by, observed=True).agg(n=('caught', 'size'), recall=('caught', 'mean'), median_p=('p', 'median'))
    print(f'--- recall by {name}\n{g.round(3).to_string()}\n')


show(pd.cut(f.pos, [0, 1, 2, 3, 5, 100]), 'position in burst (1 = first fraud transaction of the card)')
show(pd.cut(f.h_since_start, [-1, 0.001, 1, 6, 24, 48, 100]), 'hours since the burst started')
show('category', 'category')
show(f.hour.isin([22, 23, 0, 1, 2, 3]).map({True: 'night 22-03', False: 'day'}), 'time of day')
show(pd.cut(f.amt, [0, 20, 50, 100, 300, 600, 1000, 5000]), 'amount')
show(f.u_amt_cnt_all.eq(0).map({True: 'first transaction ever', False: 'card has history'}), 'card history')

print('--- missed fraud: what they look like (median values)')
cols = ['amt', 'hour', 'amt_vs_cat', 'u_amt_cnt_all', 'u_amt_cnt_3', 'since_last_s', 'pos', 'p']
print(pd.concat({'missed': f.loc[~f.caught, cols].median(), 'caught': f.loc[f.caught, cols].median()}, axis=1).round(2))

fp = val[val.caught & (val[LABEL] == 0)]
print(f'\n--- false positives ({len(fp)}): categories / night share / median amount')
print(fp.category.value_counts().head(8).to_string())
print(f'night share {fp.hour.isin([22, 23, 0, 1, 2, 3]).mean():.2f} | median amt {fp.amt.median():.0f} | '
      f'median amt_vs_cat {fp.amt_vs_cat.median():.2f}')

# Are the transactions right before a fraud transaction (same card, within 3 days) also fraud?
t_sorted = t[['cc_num', 'ts', LABEL, 'hour', 'amt_vs_cat']]
prev_fraud = t_sorted.groupby('cc_num')[LABEL].shift()
prev_gap = (t_sorted.ts - t_sorted.groupby('cc_num').ts.shift()) / 3600
v = val.index
print('\n--- previous transaction of the same card (val rows)')
tab = pd.crosstab(val[LABEL], prev_fraud.loc[v].fillna(-1).astype(int), rownames=['is_fraud'],
                  colnames=['previous is_fraud (-1 = none)'])
print(tab)
print(f'median hours since previous transaction: fraud {prev_gap.loc[f.index].median():.1f}, '
      f'non-fraud {prev_gap.loc[val.index[val[LABEL] == 0]].median():.1f}')

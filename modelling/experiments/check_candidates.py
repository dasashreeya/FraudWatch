"""Leakage test for candidate feature groups (same method as check_leakage.py).

Each group is computed on the full data and on the data cut off at 2019-09-01 with shuffled labels. A feature
that uses only earlier transactions and no labels has identical values before the cut-off.

Run from modelling/:  python experiments/check_candidates.py [group ...]
"""
import sys

import numpy as np
import pandas as pd

import lab  # first: puts the project folder on the import path
import candidates
import config
from data import prepare, read_transactions
from features import add_history_features

CUTOFF = '2019-09-01'
names = sys.argv[1:] or list(candidates.GROUPS)

raw = read_transactions(config.DATA_DIR / config.TRAIN_FILE, 'train')
cut = raw[pd.to_datetime(raw.trans_date_trans_time) < CUTOFF].copy()
cut['is_fraud'] = np.random.RandomState(0).permutation(cut.is_fraud.to_numpy())
part = add_history_features(prepare(cut))
full = lab.base_table().set_index('trans_num')

ok = True
for name in names:
    a = lab.feature_group(name).set_index(lab.base_table().trans_num).loc[part.trans_num]
    b = candidates.GROUPS[name](part)
    av, bv = a.to_numpy(float), b[a.columns].to_numpy(float)
    tol = 1e-4 * np.nanstd(av, axis=0) + 1e-6
    changed = (~np.isclose(av, bv, rtol=0, atol=tol, equal_nan=True)).sum(0)
    bad = [c for c, n in zip(a.columns, changed) if n]
    ok &= not bad
    print(f'{name:16s} {len(a.columns):2d} features | changed: {bad or "none"}')
print('RESULT:', 'PASS' if ok else 'FAIL')

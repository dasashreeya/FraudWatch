"""Score propagation ("stacking over time"): features built from the fraud scores of the card's EARLIER
transactions.

Stage 1: a model on the production features scores every transaction.
  - fit rows get OUT-OF-FOLD scores (5 folds over cards), so a card's own labels never shape its own scores;
  - every later row (validation, test) is scored by a stage-1 model trained on all fit rows.
Stage 2: per card, using only earlier transactions: the scores of the previous 1-3 transactions, the max and
mean score over the last 24 h / 72 h, how many earlier transactions in the last 72 h scored above 0.5, and the
hours since the last one that did. In production this is natural: earlier transactions were already scored
when they happened.

Validation / test rows only see stage-1 models trained on fit rows, so their evaluation stays honest. For fit
rows the features depend on other cards' labels (like the job buying-power model's look-ahead).
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import lab
from data import LABEL
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, _rolling, win_stats
from models import MatrixEncoder

STAGE1_ROUNDS = 500     # fixed (no early stopping), so stage 1 never looks at the validation rows


def stage1_scores(split='A', groups=(), n_folds=5, seed=42, rounds=STAGE1_ROUNDS):
    """Stage-1 LightGBM scores for every row (OOF by card on fit rows). Cached."""
    import lightgbm as lgb
    key = f'stack1_{split}_{"+".join(groups) or "base"}_{n_folds}_{seed}_{rounds}'
    path = lab.CACHE / f'{key}.npy'
    if path.exists():
        return np.load(path)
    t0 = time.time()
    t, new_cols = lab.table(groups)
    fit_m, _, _ = lab.split_masks(t, split)
    numeric = NUMERIC_FEATURES + new_cols
    enc = MatrixEncoder(numeric, CATEGORICAL_FEATURES).fit(t[fit_m])
    X = enc.transform(t).to_numpy(np.float32)
    y = t[LABEL].to_numpy(int)
    params = dict(lab.LGB_PARAMS, seed=seed)
    scores = np.full(len(t), np.nan)
    fit_idx = np.flatnonzero(fit_m)
    for k, (tr, te) in enumerate(GroupKFold(n_folds).split(fit_idx, groups=t.cc_num.to_numpy()[fit_idx])):
        b = lgb.train(params, lgb.Dataset(X[fit_idx[tr]], y[fit_idx[tr]]), num_boost_round=rounds)
        scores[fit_idx[te]] = b.predict(X[fit_idx[te]])
        lab.log(f'  stage 1 fold {k + 1}/{n_folds} done')
    b = lgb.train(params, lgb.Dataset(X[fit_idx], y[fit_idx]), num_boost_round=rounds)
    rest = np.flatnonzero(~fit_m)
    scores[rest] = b.predict(X[rest])
    np.save(path, scores)
    lab.log(f'stage-1 scores {key} ready ({time.time() - t0:.0f}s)')
    return scores


def propagation_features(b, s):
    """Stage-2 features from stage-1 scores `s` (aligned with base table `b`, sorted by card and time)."""
    out = pd.DataFrame(index=b.index)
    s = pd.Series(s, index=b.index)
    g = s.groupby(b.cc_num)
    for k in (1, 2, 3):
        out[f's_lag{k}'] = g.shift(k).fillna(-1).to_numpy()
    for win, name in (('1D', '24h'), ('3D', '72h')):
        out[f's_max_{name}'] = np.nan_to_num(_rolling(b, s, win, 'max'), nan=-1)
    tmp = b[['cc_num', 'ts']].copy()
    tmp['s'] = s.to_numpy()
    tmp['hi'] = (s.to_numpy() > 0.5).astype(float)
    m = win_stats(tmp, ['cc_num'], 's', [1, 3], 'sm')
    h = win_stats(tmp, ['cc_num'], 'hi', [3], 'hi', empty=0.0)
    out['s_mean_24h'], out['s_mean_72h'] = m['sm_mean_1'], m['sm_mean_3']
    out['s_hi_cnt_72h'] = h['hi_cnt_3'] * h['hi_mean_3']
    # hours since the card's last earlier transaction with score > 0.5
    last_hi_ts = pd.Series(np.where(tmp.hi > 0, b.ts, np.nan), index=b.index).groupby(b.cc_num).shift(1)
    last_hi_ts = last_hi_ts.groupby(b.cc_num).ffill()
    out['hours_since_hi'] = ((b.ts - last_hi_ts) / 3600).fillna(-1).to_numpy()
    return out


def run_stacked(tag, split='A', groups=(), stage1_groups=None, model='lightgbm', seeds=(42,), **kw):
    """Stage 2 = `model` on production features + candidate groups + propagation features."""
    stage1_groups = groups if stage1_groups is None else stage1_groups
    s = stage1_scores(split, stage1_groups)
    feats = propagation_features(lab.base_table(), s)
    name = f'_prop_{split}_{"+".join(stage1_groups) or "base"}'
    lab.EXTRA[name] = feats
    try:
        return lab.run(tag, groups=list(groups) + [name], model=model, seeds=seeds, split=split, **kw)
    finally:
        lab.EXTRA.pop(name, None)

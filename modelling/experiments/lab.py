"""Experiment lab: cached feature tables and fast train / evaluate loops for trying new features and settings.

This is for local experiments only; the production pipeline is train.py / predict.py. It reuses their code
(data.py, features.py, models.MatrixEncoder) so that a baseline run here reproduces train.py exactly.

    import lab
    lab.run('production', model='lightgbm', seeds=[42, 1, 2])                       # the current features
    lab.run('+my_group', groups=['my_group'], model='lightgbm', seeds=[42, 1, 2])    # plus a candidate group
    lab.run('-merchant location', drop=['merchant_location'], seeds=[42, 1, 2])      # without some features

Selection discipline: every run is scored on VALIDATION. Two validation windows are available:
    split 'A'  fit = first 85% of the training file, val = last 15%   (the production split)
    split 'B'  fit = first 70% of the training file, val = the next 15% (an earlier, independent window)
The test set is only scored when eval_test=True, which is reserved for the final comparison.

Files (all under runs/lab/, git-ignored):
    cache/base_<hash>.parquet    the production feature table; the hash changes when features.py / data.py change
    cache/fe_<group>_<hash>.parquet   one candidate feature group; the hash changes when its code changes
    results.csv                  one row per (run, seed)
    preds/<run>__<seed>.npz      validation (and test) probabilities, for ensembles and error analysis
"""
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from data import LABEL, assign_partitions, prepare, read_transactions  # noqa: E402
from features import (CATEGORICAL_FEATURES, MLOC_F, NUMERIC_FEATURES, JobBuyingPower,  # noqa: E402
                      add_history_features)
from models import MatrixEncoder, best_f1_threshold, evaluate  # noqa: E402

LAB = ROOT / 'runs' / 'lab'
CACHE = LAB / 'cache'
PREDS = LAB / 'preds'
RESULTS = LAB / 'results.csv'

# Named groups of existing features that experiments may drop (any single feature name works too)
DROP_SETS = {
    'merchant_location': MLOC_F + ['dist_um_km'],             # distance-based, noise in this simulated data
    'identity': ['city', 'job', 'state', 'city_pop'],          # features that (nearly) identify the card
}

LGB_PARAMS = dict(objective='binary', learning_rate=0.1, max_depth=7, num_leaves=127, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0,
                  metric='None', verbose=-1)
XGB_PARAMS = dict(objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=7, subsample=0.8,
                  colsample_bytree=0.8)
CB_PARAMS = dict(learning_rate=0.1, depth=6, has_time=True, one_hot_max_size=60, simple_ctr=['Borders'],
                 combinations_ctr=['Borders'])                       # same as models.CatBoostModel

EXTRA = {}   # name -> DataFrame of extra columns computed outside candidates.py (e.g. stacking.py)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Feature tables
# ---------------------------------------------------------------------------------------------------------------
_BASE = None


def base_table():
    """The production feature table, built exactly as train.py builds it, cached as parquet."""
    global _BASE
    if _BASE is not None:
        return _BASE
    src = b''.join((ROOT / f).read_bytes() for f in ('features.py', 'data.py'))
    path = CACHE / f'base_{hashlib.md5(src).hexdigest()[:8]}.parquet'
    if path.exists():
        _BASE = pd.read_parquet(path)
        return _BASE
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw = pd.concat([read_transactions(config.DATA_DIR / config.TRAIN_FILE, 'train'),
                     read_transactions(config.DATA_DIR / config.TEST_FILE, 'test')], ignore_index=True)
    df = prepare(raw)
    del raw
    feats = add_history_features(df)
    del df
    assign_partitions(feats, config.VAL_FRAC)
    JobBuyingPower().fit(feats[feats.part == 'fit']).transform(feats)
    feats.to_parquet(path)
    log(f'built base table {feats.shape} in {time.time() - t0:.0f}s')
    _BASE = feats
    return feats


def feature_group(name):
    """Columns of one candidate feature group (see candidates.py), computed once and cached."""
    import candidates
    fn = candidates.GROUPS[name]
    src = inspect.getsource(fn) + ''.join(inspect.getsource(h) for h in getattr(fn, 'helpers', []))
    digest = hashlib.md5(src.encode()).hexdigest()[:8]
    path = CACHE / f'fe_{name}_{digest}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    out = fn(base_table())
    out = out.astype({c: np.float64 for c in out.columns if out[c].dtype != object})
    out.to_parquet(path)
    log(f'computed feature group {name}: {list(out.columns)} ({time.time() - t0:.0f}s)')
    return out


def table(groups=()):
    """Base table plus candidate groups (a new DataFrame; the cached base table is left unchanged)."""
    b = base_table()
    if not groups:
        return b, []
    parts, new_cols = [b], []
    for g in groups:
        f = EXTRA[g] if g in EXTRA else feature_group(g)
        parts.append(f.set_index(b.index))
        new_cols += list(f.columns)
    return pd.concat(parts, axis=1), new_cols


def split_masks(t, split='A'):
    part = t.part.to_numpy()
    if split == 'A':
        return part == 'fit', part == 'val', part == 'test'
    if split == 'B':
        is_train = (t.source == 'train').to_numpy()
        ts = t.ts.to_numpy()
        q70, q85 = np.quantile(ts[is_train], [0.70, 0.85])
        return is_train & (ts <= q70), is_train & (ts > q70) & (ts <= q85), part == 'test'
    raise ValueError(split)


# ---------------------------------------------------------------------------------------------------------------
# Training (same procedure as models.py, with adjustable settings)
# ---------------------------------------------------------------------------------------------------------------
class _Stop:
    def __init__(self, patience):
        self.hist, self.patience = [], patience

    def add(self, v):
        self.hist.append(float(v))
        return len(self.hist) - 1 - int(np.argmax(self.hist)) >= self.patience

    @property
    def best(self):
        return int(np.argmax(self.hist)) + 1


def train_lgb(Xf, yf, Xv, yv, seed, params, max_iters, patience):
    import lightgbm as lgb
    p = dict(LGB_PARAMS, seed=seed, **params)
    dfit = lgb.Dataset(Xf, yf)
    dval = lgb.Dataset(Xv, yv, reference=dfit)
    booster = lgb.train(p, dfit, num_boost_round=max_iters, valid_sets=[dval], valid_names=['val'],
                        feval=lambda pr, d: ('pr_auc', average_precision_score(d.get_label(), pr), True),
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    it = booster.best_iteration
    return (lambda X: booster.predict(X, num_iteration=it)), it, booster


def train_xgb(Xf, yf, Xv, yv, seed, params, max_iters, patience):
    import xgboost as xgb
    from xgboost.callback import TrainingCallback
    dfit, dval, stop = xgb.DMatrix(Xf, label=yf), xgb.DMatrix(Xv), _Stop(patience)

    class EarlyStop(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            return stop.add(average_precision_score(yv, model.predict(dval)))

    p = dict(XGB_PARAMS, seed=seed, **params)
    booster = xgb.train(p, dfit, num_boost_round=max_iters, verbose_eval=False, callbacks=[EarlyStop()])
    booster = booster[:stop.best]
    return (lambda X: booster.predict(xgb.DMatrix(X)).astype(np.float64)), stop.best, booster


class _CBMetric:
    stop = None

    def __init__(self, n_val):
        self.n_val = n_val

    def is_max_optimal(self):
        return True

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def evaluate(self, approxes, target, weight):
        if len(target) != self.n_val:
            return 0.0, 1.0
        ap = average_precision_score(target, 1 / (1 + np.exp(-np.asarray(approxes[0], float))))
        _CBMetric.stop.add(ap)
        return ap, 1.0


def train_cb(fit_df, val_df, numeric, categorical, seed, params, max_iters, patience):
    from catboost import CatBoostClassifier, Pool
    pool = lambda d, lab=True: Pool(d[numeric + categorical], d[LABEL].astype(int) if lab else None,
                                    cat_features=categorical)
    _CBMetric.stop = _Stop(patience)
    p = dict(CB_PARAMS, **params)
    if p.get('has_time'):           # CatBoost then treats the row order as time: sort the training rows by time
        fit_df = fit_df.sort_values('ts', kind='stable')
    m = CatBoostClassifier(iterations=max_iters, random_seed=seed, od_type='Iter', od_wait=patience, metric_period=1,
                           thread_count=-1, eval_metric=_CBMetric(len(val_df)), verbose=0, allow_writing_files=False,
                           **p)
    m.fit(pool(fit_df), eval_set=pool(val_df), use_best_model=True)
    return (lambda d: m.predict_proba(pool(d, lab=False))[:, 1]), m.get_best_iteration() + 1, m


# ---------------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------------
def _metrics(y, p, thr):
    m = evaluate(y, p, thr)
    return {k: m[k] for k in ('pr_auc', 'auc_roc', 'precision', 'recall', 'f1', 'flagged')}


def run(tag, groups=(), drop=(), model='lightgbm', seeds=(42,), split='A', params=None, max_iters=600,
        patience=50, eval_test=False, numeric=None, categorical=None, save_preds=True):
    """Train `model` once per seed on the chosen split and log validation (and optionally test) metrics.

    groups: candidate feature groups to add (candidates.py); drop: feature names or DROP_SETS keys to remove.
    Returns a DataFrame with one row per seed.
    """
    params = params or {}
    t, new_cols = table(groups)
    drop_cols = set()
    for d in drop:
        drop_cols |= set(DROP_SETS.get(d, [d]))
    numeric = numeric or [c for c in NUMERIC_FEATURES + new_cols if c not in drop_cols]
    categorical = categorical if categorical is not None else [c for c in CATEGORICAL_FEATURES if c not in drop_cols]
    fit_m, val_m, test_m = split_masks(t, split)
    fit, val = t[fit_m], t[val_m]
    test = t[test_m] if eval_test else None
    y_fit, y_val = fit[LABEL].to_numpy(int), val[LABEL].to_numpy(int)

    if model in ('lightgbm', 'xgboost'):
        enc = MatrixEncoder(numeric, categorical).fit(fit)
        Xf, Xv = enc.transform(fit), enc.transform(val)
        Xt = enc.transform(test) if eval_test else None
    rows = []
    for seed in seeds:
        t0 = time.time()
        if model == 'lightgbm':
            predict, it, _ = train_lgb(Xf, y_fit, Xv, y_val, seed, params, max_iters, patience)
            p_val, p_test = predict(Xv), (predict(Xt) if eval_test else None)
        elif model == 'xgboost':
            predict, it, _ = train_xgb(Xf, y_fit, Xv, y_val, seed, params, max_iters, patience)
            p_val, p_test = predict(Xv), (predict(Xt) if eval_test else None)
        elif model == 'catboost':
            predict, it, _ = train_cb(fit, val, numeric, categorical, seed, params, max_iters, patience)
            p_val, p_test = predict(val), (predict(test) if eval_test else None)
        else:
            raise ValueError(model)
        secs = time.time() - t0
        thr = best_f1_threshold(y_val, p_val)
        row = dict(tag=tag, model=model, split=split, seed=seed, groups='+'.join(groups), drop='+'.join(drop),
                   params=json.dumps(params, sort_keys=True), max_iters=max_iters, n_numeric=len(numeric),
                   rounds=it, secs=round(secs), threshold=thr,
                   **{f'val_{k}': v for k, v in _metrics(y_val, p_val, thr).items()})
        if eval_test:
            row.update({f'test_{k}': v for k, v in _metrics(test[LABEL].to_numpy(int), p_test, thr).items()})
        rows.append(row)
        LAB.mkdir(parents=True, exist_ok=True)       # saved per seed, so an interrupted batch keeps finished runs
        if RESULTS.exists():
            old = pd.read_csv(RESULTS)
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS, index=False)
        else:
            pd.DataFrame([row]).to_csv(RESULTS, index=False)
        log(f'{tag} | {model} split {split} seed {seed}: val PR-AUC {row["val_pr_auc"]:.4f} '
            f'(P {row["val_precision"]:.3f} R {row["val_recall"]:.3f}) '
            + (f'| test PR-AUC {row["test_pr_auc"]:.4f} (P {row["test_precision"]:.3f} R {row["test_recall"]:.3f}) '
               if eval_test else '') + f'| {it} rounds {secs:.0f}s')
        if save_preds:
            PREDS.mkdir(parents=True, exist_ok=True)
            key = f'{tag}__{model}__{split}__{seed}'.replace(' ', '_').replace('/', '-')
            np.savez_compressed(PREDS / f'{key}.npz', val_idx=np.flatnonzero(val_m), p_val=p_val,
                                **({'test_idx': np.flatnonzero(test_m), 'p_test': p_test} if eval_test else {}))
    return pd.DataFrame(rows)


def summary(tags=None, cols=('val_pr_auc', 'val_recall', 'val_precision', 'rounds')):
    """Mean and std per (tag, model, split) over seeds, from results.csv."""
    r = pd.read_csv(RESULTS)
    if tags is not None:
        r = r[r.tag.isin(tags)]
    g = r.groupby(['tag', 'model', 'split'], sort=False)
    out = g[list(cols)].mean().round(4)
    out['val_pr_auc_std'] = g.val_pr_auc.std().round(4)
    out['seeds'] = g.size()
    return out

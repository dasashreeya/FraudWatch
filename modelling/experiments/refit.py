"""Does retraining on fit + validation (with the early-stopped number of rounds) beat the fit-only model?

Simulated on the earlier window so the test set stays untouched:
  model 1: trained on the first 70% of the training file, early-stopped on the next 15% (window B);
  model 2: retrained on the first 85% (fit + window B) for model 1's rounds x 85/70, no early stopping;
both are scored on the last 15% (window A), with the threshold chosen on window B by model 1. This mirrors
production: the fit-only model meets the test period after a 2.5-month gap; the retrained one does not.
"""
import time

import numpy as np
import pandas as pd

import lab
from data import LABEL
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from models import MatrixEncoder, best_f1_threshold


def _fixed(model, X, y, seed, params, rounds, numeric=None, categorical=None):
    """Train for a fixed number of rounds without a validation set; returns a predict function."""
    if model == 'lightgbm':
        import lightgbm as lgb
        b = lgb.train(dict(lab.LGB_PARAMS, seed=seed, **params), lgb.Dataset(X, y), num_boost_round=rounds)
        return lambda Z: b.predict(Z)
    if model == 'xgboost':
        import xgboost as xgb
        b = xgb.train(dict(lab.XGB_PARAMS, seed=seed, **params), xgb.DMatrix(X, label=y), num_boost_round=rounds)
        return lambda Z: b.predict(xgb.DMatrix(Z)).astype(np.float64)
    from catboost import CatBoostClassifier, Pool
    p = dict(lab.CB_PARAMS, **params)
    if p.get('has_time'):
        X = X.sort_values('ts', kind='stable')
        y = X[LABEL].to_numpy(int)
    m = CatBoostClassifier(iterations=rounds, random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False,
                           **p)
    m.fit(Pool(X[numeric + categorical], y, cat_features=categorical))
    return lambda d: m.predict_proba(Pool(d[numeric + categorical], cat_features=categorical))[:, 1]


def run_refit(tag, groups=(), drop=(), model='lightgbm', seeds=(42,), params=None):
    params = params or {}
    t, new_cols = lab.table(groups)
    drop_cols = set()
    for d in drop:
        drop_cols |= set(lab.DROP_SETS.get(d, [d]))
    numeric = [c for c in NUMERIC_FEATURES + new_cols if c not in drop_cols]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in drop_cols]
    fit_b, val_b, _ = lab.split_masks(t, 'B')
    _, val_a, _ = lab.split_masks(t, 'A')
    both = fit_b | val_b
    y = t[LABEL].to_numpy(int)
    if model != 'catboost':
        enc1 = MatrixEncoder(numeric, categorical).fit(t[fit_b])
        enc2 = MatrixEncoder(numeric, categorical).fit(t[both])
        X1f, X1v, X1a = enc1.transform(t[fit_b]), enc1.transform(t[val_b]), enc1.transform(t[val_a])
        X2f, X2a = enc2.transform(t[both]), enc2.transform(t[val_a])
    rows = []
    for seed in seeds:
        t0 = time.time()
        if model == 'lightgbm':
            pred1, rounds, _ = lab.train_lgb(X1f, y[fit_b], X1v, y[val_b], seed, params, 600, 50)
            p1v, p1a = pred1(X1v), pred1(X1a)
        elif model == 'xgboost':
            pred1, rounds, _ = lab.train_xgb(X1f, y[fit_b], X1v, y[val_b], seed, params, 600, 50)
            p1v, p1a = pred1(X1v), pred1(X1a)
        else:
            pred1, rounds, _ = lab.train_cb(t[fit_b], t[val_b], numeric, categorical, seed, params, 600, 50)
            p1v, p1a = pred1(t[val_b]), pred1(t[val_a])
        thr = best_f1_threshold(y[val_b], p1v)
        r2 = int(round(rounds * both.sum() / fit_b.sum()))
        if model == 'catboost':
            pred2 = _fixed(model, t[both], y[both], seed, params, r2, numeric, categorical)
            p2a = pred2(t[val_a])
        else:
            pred2 = _fixed(model, X2f, y[both], seed, params, r2)
            p2a = pred2(X2a)
        for name, p in (('fit only', p1a), ('refit fit+val', p2a)):
            m = lab._metrics(y[val_a], p, thr)
            rows.append(dict(tag=tag, model=model, seed=seed, variant=name, rounds=rounds if name == 'fit only' else r2,
                             **m))
            lab.log(f'{tag} | {model} seed {seed} {name:14s}: window-A PR-AUC {m["pr_auc"]:.4f} '
                    f'P {m["precision"]:.3f} R {m["recall"]:.3f} F1 {m["f1"]:.3f}')
        lab.log(f'  ({time.time() - t0:.0f}s)')
    out = pd.DataFrame(rows)
    path = lab.LAB / 'refit_results.csv'
    out.to_csv(path, mode='a', header=not path.exists(), index=False)
    return out

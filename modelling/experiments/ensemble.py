"""Average the saved validation predictions of several models (same tag, split and seed) and score them.

    python experiments/ensemble.py <tag> [--split A] [--models catboost xgboost lightgbm]
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import lab
from data import LABEL
from models import best_f1_threshold, evaluate


def load(tag, model, split, seed):
    key = f'{tag}__{model}__{split}__{seed}'.replace(' ', '_').replace('/', '-')
    return np.load(lab.PREDS / f'{key}.npz')


def score(tag, models, split='A', seeds=(42, 1, 2), how='mean', part='val'):
    y_all = lab.base_table()[LABEL].to_numpy(int)
    rows = []
    for seed in seeds:
        preds = [load(tag, m, split, seed) for m in models]
        idx = preds[0][f'{part}_idx']
        y = y_all[idx]
        ps = [p[f'p_{part}'] for p in preds]
        p = np.mean([rankdata(q) / len(q) for q in ps] if how == 'rank' else ps, axis=0)
        thr = best_f1_threshold(y, p) if part == 'val' else None
        m = evaluate(y, p, thr if thr is not None else 0.5)
        rows.append(dict(seed=seed, pr_auc=m['pr_auc'], precision=m['precision'], recall=m['recall']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--split', default='A')
    ap.add_argument('--models', nargs='+', default=['catboost', 'xgboost', 'lightgbm'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1, 2])
    a = ap.parse_args()
    for k in range(1, len(a.models) + 1):
        for combo in itertools.combinations(a.models, k):
            for how in (('mean', 'rank') if k > 1 else ('mean',)):
                r = score(a.tag, combo, a.split, a.seeds, how)
                print(f'{"+".join(combo):28s} {how:5s} val PR-AUC {r.pr_auc.mean():.4f} (sd {r.pr_auc.std():.4f}) '
                      f'P {r.precision.mean():.3f} R {r.recall.mean():.3f}')

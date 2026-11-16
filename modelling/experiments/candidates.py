"""Candidate feature groups for experiments (add your own here).

Like the production history features, a group must use only EARLIER transactions (by timestamp) and never a
fraud label, so each value could be computed when the card is swiped. Input: the base table from
lab.base_table() (the production features, sorted by card and then time). Output: a DataFrame of new columns
with the same index. Register it with @group(); lab.run(..., groups=['my_group']) then caches and uses it, and
`python experiments/check_candidates.py my_group` tests it for leakage.

Example:

    @group()
    def merchant_velocity(b):
        '''How many transactions the merchant had in the last hour (all cards).'''
        w = win_stats(b, ['merchant'], 'amt', [1 / 24], 'mv')
        return pd.DataFrame({'m_cnt_1h': w[f'mv_cnt_{1 / 24}']}, index=b.index)

Every group screened so far was adopted into features.py (see experiments/README.md for the results).
"""
import numpy as np  # noqa: F401  (for new groups)
import pandas as pd  # noqa: F401

from features import win_stats  # noqa: F401

GROUPS = {}


def _frame(b, **cols):
    """A small (card, time) frame with extra columns, for win_stats on derived values."""
    tmp = b[['cc_num', 'ts']].copy()
    for k, v in cols.items():
        tmp[k] = np.asarray(v)
    return tmp


def group(*helpers):
    """Register a feature group; `helpers` are hashed with it so cached values refresh when they change."""
    def deco(fn):
        fn.helpers = (_frame,) + helpers
        GROUPS[fn.__name__] = fn
        return fn
    return deco

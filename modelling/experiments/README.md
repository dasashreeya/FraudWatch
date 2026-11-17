# Experiments: how the v2 features were chosen

This folder holds the tools used to improve the model, and a record of what was tried. The production
pipeline (`train.py`, `predict.py`) does not depend on it.

## Method
- **Screen with LightGBM, 3 seeds (42, 1, 2), on validation only.** LightGBM trains in about 30 s, and the
  same procedure as `train.py` is used (early stopping on validation PR-AUC, F1-optimal threshold).
- **Two validation windows**, because one window holds only about 120 fraud cards:
  - **A:** fit on the first 85% of the training file, validate on the last 15% (the production split);
  - **B:** fit on the first 70%, validate on the next 15% (an earlier, independent window).
- **The test set was scored once**, for the final comparison, after every choice had been made.
- **Every candidate feature passes the leakage test** (`check_candidates.py`, the same method as
  `check_leakage.py`): it uses only earlier transactions and no labels.

## 1. Where v1 failed
`error_analysis.py` looks at the validation fraud that the previous model, v1 (val PR-AUC 0.978), missed:
- **Most missed fraud was mid-burst, not the burst's first transaction.** The median missed transaction was the
  4th of its card's fraud burst, in the daytime, about $48, a normal amount for its category.
- **Those transactions look ordinary on their own**, but the card's previous transactions don't: for 1,015 of
  1,133 validation fraud transactions, the card's previous transaction was fraud too.
- **The model had no short-term view of the card.** Its shortest look-back window was 3 days.

So the new features describe the card's **recent activity**: the last hours, and the last few transactions.

## 2. Feature groups, one at a time (validation PR-AUC, window A)

| Added to v1 | Val PR-AUC | Change |
|---|---|---|
| none (v1) | 0.9775 | |
| previous 3 transactions (amount vs category, hour, time since) | **0.9931** | +0.0156 |
| recent unusual activity (night, large-for-category purchases in 24 h / 72 h) | 0.9874 | +0.0099 |
| velocity (count / amount in the last 1 h, 6 h, 24 h; largest amount 24 h / 72 h) | 0.9871 | +0.0096 |
| night activity (share and counts of night transactions) | 0.9807 | +0.0032 |
| day activity (transactions so far today vs a typical day) | 0.9803 | +0.0028 |
| hour habits (share of the card's past transactions at this hour) | 0.9801 | +0.0026 |
| card × category history | 0.9792 | +0.0017 |
| amount vs the card's largest earlier amounts | 0.9790 | +0.0015 |
| amount vs similar cardholders' purchases in the category | 0.9786 | +0.0011 |
| **all nine groups** | **0.9944** | **+0.0169** |

Seed-to-seed standard deviation was 0.0001–0.0015.

## 3. Combinations, confirmed on both windows (validation PR-AUC)

| Feature set | Window A | Window B |
|---|---|---|
| v1 | 0.9775 | 0.9798 |
| + previous 3 transactions | 0.9931 | 0.9934 |
| + all nine groups | 0.9944 | 0.9951 |
| + all nine, + previous 4th–5th transactions and categories, + last 5 / 10 transactions | 0.9955 | 0.9967 |
| … without raw coordinates and travel features (**v2**) | **0.9956** | **0.9971** |
| … without any location feature | 0.9957 | 0.9965 |

Location features made no measurable difference once the recent-activity features were in. The raw coordinates
and the travel features were removed because merchant positions are random in this simulated data and home
coordinates identify the card. The merchant-location features stayed.

## 4. Tried, not adopted

| Idea | Result (validation PR-AUC unless stated) |
|---|---|
| Remove card-identifying features (city, job, state, city population, merchant; 5 variants) | 0.9942–0.9947 (A), 0.9953–0.9961 (B): no gain. v2 still leaves out city and job, because their encodings would leak later transactions into training rows (main README, section 5.3) |
| Average CatBoost, XGBoost and LightGBM | best pair 0.9962 vs 0.9958 for XGBoost alone: within noise |
| Retrain on fit + validation with the early-stopped rounds (simulated on the earlier window) | 0.9957 vs 0.9957: no gain |
| LightGBM learning rate 0.05 (up to 1,500 rounds) / feature fraction 0.5 | 0.9959 / 0.9950 vs 0.9957 |
| CatBoost with up to 2,000 rounds (instead of 600) | early stopping ends it after 500–730 rounds; same score |
| CatBoost without city and job | 0.9948 vs 0.9943 |

**Adopted for CatBoost:** time-ordered category statistics (`has_time=True`), at 0.9946 vs 0.9943. The gain is
within noise, but the setting removes a look-ahead in how CatBoost encodes categories for training rows.

## 5. Final comparison on the test set (3 seeds, mean ± sd)

| Model | Features | Val PR-AUC | Test PR-AUC | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|---|
| LightGBM | v1 | 0.9775 | 0.9513 ± 0.0024 | 0.931 | 0.876 | 0.903 |
| LightGBM | v2 | 0.9952 | **0.9879 ± 0.0002** | 0.975 | 0.955 | 0.965 |
| XGBoost | v1 | 0.9768 | 0.9507 ± 0.0015 | 0.945 | 0.864 | 0.903 |
| XGBoost | v2 | 0.9961 | **0.9882 ± 0.0008** | 0.976 | 0.953 | 0.964 |
| CatBoost | v1 | 0.9786 | 0.9509 ± 0.0020 | 0.942 | 0.876 | 0.908 |
| CatBoost | v2 | 0.9950 | **0.9870 ± 0.0012** | 0.989 | 0.948 | 0.968 |

The screening runs in sections 2–4 and 6 still had city and job as model inputs, a frequency encoding of
merchant (XGBoost / LightGBM) and CatBoost's default category encoding. v2 leaves these out, so that no encoding
carries information about later transactions (main README, sections 5.3 and 5.5); accuracy did not change
measurably.

## 6. Promising next step: score propagation (`stacking.py`)
A first-stage model scores every transaction. The final model then also gets the scores of the card's
**earlier** transactions: the last 3 scores, the highest and mean score in the last 24 h / 72 h, and the hours
since the last score above 0.5. This works in production because earlier transactions were already scored when
they happened. Training rows get out-of-fold first-stage scores (5 folds over cards), so a card's own labels
never shape its own features.

| On top of v2 | Val A | Val B | Test PR-AUC | Test precision | Test recall |
|---|---|---|---|---|---|
| LightGBM | 0.9954 | 0.9982 | **0.9915 ± 0.0004** | 0.985 | 0.962 |
| XGBoost | 0.9958 | | **0.9912 ± 0.0003** | 0.987 | 0.957 |

The same setup without propagation scored val B 0.9965 and test 0.9878 (LightGBM) / 0.9876 (XGBoost).

It is not in the production pipeline yet, for two reasons:
- **Training and scoring would need a second model.** That's a deployment decision.
- **The look-back must be bounded before production.** "Hours since the last high score" would need a limit
  (e.g. 30 days), so that inference never relies on first-stage scores of training rows.

## Files

| File | Purpose |
|---|---|
| `lab.py` | Cached production feature table, `run()` to train and score a configuration, `summary()` |
| `candidates.py` | Where new candidate feature groups go (see its docstring) |
| `check_candidates.py` | Leakage test for candidate groups |
| `error_analysis.py` | Which validation fraud the saved best model misses, and what it looks like |
| `ensemble.py` | Averages saved validation predictions of several models |
| `refit.py` | Fit-only vs retrained-on-fit+validation, simulated on the earlier window |
| `stacking.py` | Score propagation (section 6) |

Everything the lab writes goes to `runs/lab/` (git-ignored): the feature cache, `results.csv` (one row per run
and seed) and saved predictions. A typical session, from the `modelling/` folder:
```bash
cd experiments
python -c "import lab; lab.run('production', seeds=[42, 1, 2]); print(lab.summary())"
```

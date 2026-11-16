# FraudWatch: credit-card fraud detection

This project trains and serves a model that flags fraudulent credit-card transactions. It uses the
[Sparkov simulated transactions dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection): 1.85 million
transactions from 999 cardholders over 2019–2020, of which 0.52% are fraud.

The pipeline covers data cleaning, 111 features, three gradient-boosting models and a time-based evaluation. Most
features describe the card's **recent activity** (its last few hours and last few transactions), because fraud
arrives in bursts. Every feature can be computed at the moment a card is swiped, because it uses only earlier
transactions and never a fraud label. `check_leakage.py` verifies this, and
[section 5.4](#54-keeping-the-merchant-features-leak-free) shows a tempting shortcut that would break it.

**What you can do with it:**
- download the data (`download_data.py`)
- build the features, then train and compare CatBoost, XGBoost and LightGBM (`train.py`)
- score transactions with the included best model (`predict.py`)
- verify that no feature uses future data or labels (`check_leakage.py`)
- try new features or settings with the experiment tools (`experiments/`)

**Headline result** (test set: 555,719 transactions from the last 6 months of 2020, never used for any choice):
the best model (LightGBM, chosen on validation) reaches **PR-AUC 0.987**, catching **95% of fraud transactions
at 97% precision**.

**Versions.** This README describes **v2**, the current model, which `predict.py` uses. The previous model,
**v1** (CatBoost on the card's long-term history only), reached PR-AUC 0.949 and caught 86% at 95% precision. It
is kept, with the code that built it, under the git tag
[`v1`](https://github.com/drduron/650_Group_Project/tree/v1/modelling) (`git checkout v1`).
[Section 6.2](#62-why-v2-outperforms-v1) explains why v2 is better.

---

## Contents
1. [Quick start](#1-quick-start)
2. [Project layout](#2-project-layout)
3. [Setup in detail](#3-setup-in-detail)
4. [The data](#4-the-data)
5. [How the pipeline works](#5-how-the-pipeline-works)
6. [Results](#6-results)
7. [Inference: scoring transactions with the saved model](#7-inference-scoring-transactions-with-the-saved-model)
8. [Leakage check](#8-leakage-check)
9. [Script and module reference](#9-script-and-module-reference)
10. [Reproducibility](#10-reproducibility)
11. [Known limitations and next steps](#11-known-limitations-and-next-steps)

---

## 1. Quick start
Run these from this folder (macOS/Linux shown; Windows notes in [section 3](#3-setup-in-detail)):

```bash
python3.11 -m venv .venv                 # 1. create an isolated Python 3.11 environment
source .venv/bin/activate
pip install -r requirements.txt          # 2. install the pinned libraries
brew install libomp                      #    macOS only: OpenMP runtime for XGBoost and LightGBM

python download_data.py                  # 3. download the dataset into data/ (~210 MB zip, ~500 MB unzipped)
python predict.py                        # 4. score the test file with the included best model (~0.5-2 min)
python train.py                          # 5. optional: rebuild features, retrain and evaluate all 3 models (~5-10 min)
python check_leakage.py                  # 6. optional: verify no feature looks ahead or uses labels (~2-3 min)
```

The best model (LightGBM) is included in `artifacts/`, so step 4 works straight after the download. It should end
with these lines, confirming that inference reproduces the training-time evaluation exactly:
```
metrics on these 555,719 labelled transactions (2,145 fraud): PR-AUC 0.9873 | ROC-AUC 0.9996 | precision 0.9741 | recall 0.9469 | F1 0.9603
training-time test metrics: PR-AUC 0.9873 | precision 0.9741 | recall 0.9469 | F1 0.9603 -> MATCH
```
After step 5, `predict.py` also compares every probability with the ones `train.py` saved:
```
check vs training-time test predictions: 555,719 common transactions, max |probability difference| = 0.0e+00 -> IDENTICAL
```

---

## 2. Project layout
```
.
├── README.md             this file
├── requirements.txt      pinned library versions
├── .gitignore            keeps data, the virtual environment and large outputs out of git
│
├── download_data.py      SCRIPT  download + verify the dataset into data/
├── train.py              SCRIPT  build features, train and evaluate all models, save the models
├── predict.py            SCRIPT  inference: score transactions with a saved model
├── check_leakage.py      SCRIPT  test that no feature uses future transactions or fraud labels
│
├── config.py             module  paths, split fraction, random seed
├── data.py               module  read / validate / clean the CSVs, time-based split
├── features.py           module  all feature engineering (history features + job buying-power model)
├── models.py             module  CatBoost / XGBoost / LightGBM wrappers, early stopping, metrics, save/load
├── leaky_reference.py    module  deliberately leaky feature versions, a test control for check_leakage.py
│
├── experiments/          tools used to choose the features, and a record of what was tried (own README)
├── data/                 (created) fraudTrain.csv, fraudTest.csv                     - not in git
├── artifacts/            best model (lightgbm/) + job_buying_power.pkl + metadata.json; train.py writes all models here
└── results/              (created) metrics.csv, feature_importance/, logs, predictions
```

---

## 3. Setup in detail

**Requirements**
- **Python 3.11.** Tested with 3.11.14. pandas 3.x needs at least 3.11.
- **About 16 GB of RAM recommended.** Training peaks at about 7.5 GB of resident memory, inference at about 6 GB.
- **About 2 GB of free disk:** data 0.5 GB, virtual environment 0.7 GB, models and results 0.2 GB.
- **Platform:** tested on macOS on Apple silicon (M4, 10 cores). The libraries publish wheels for Linux and
  Windows as well, but those platforms have not been tested here.

**Step by step**

1. **Virtual environment.** Keeps these library versions separate from anything else on your machine.
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
   ```
   Run all later commands with the environment active, or call `.venv/bin/python <script>` directly.

2. **Libraries.**
   ```bash
   pip install -r requirements.txt
   ```
   You get catboost, xgboost, lightgbm, scikit-learn, pandas, numpy, scipy and pyarrow, at the exact versions
   used to produce the results below.

3. **macOS only: OpenMP.** XGBoost and LightGBM need the OpenMP runtime, which macOS does not include. Without
   it, `import xgboost` fails with `Library not loaded: @rpath/libomp.dylib`.
   ```bash
   brew install libomp
   ```

4. **Data.**
   ```bash
   python download_data.py
   ```
   This downloads the public Kaggle dataset (no account needed), extracts `fraudTrain.csv` and `fraudTest.csv`
   into `data/`, and checks their row counts. If Kaggle ever asks you to sign in, download the zip from the
   [dataset page](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and run
   `python download_data.py --zip <path-to-zip>`. If you keep the data elsewhere, pass `--data-dir <folder>` to
   every script.

5. **Check.**
   `python -c "import catboost, xgboost, lightgbm; print('ok')"` should print `ok`.

---

## 4. The data

| | `fraudTrain.csv` | `fraudTest.csv` |
|---|---|---|
| Period | 2019-01-01 → 2020-06-21 | 2020-06-21 → 2020-12-31 |
| Transactions | 1,296,675 | 555,719 |
| Fraud | 7,506 (0.58%) | 2,145 (0.39%) |

- **One row per card transaction.** Columns: time (`trans_date_trans_time`), card (`cc_num`), merchant and
  `category`, amount (`amt`), cardholder details (gender, address, city population, job, date of birth),
  merchant coordinates, a transaction id (`trans_num`) and the target **`is_fraud`** (0/1).
- **The data is simulated** with the [Sparkov generator](https://github.com/namebrandon/Sparkov_Data_Generation):
  999 cards, 693 merchants, 14 categories; licence CC0. Properties worth knowing:
  - Fraud comes in **one burst per card**: 2–19 transactions within at most 3 days, mostly late at night, with
    high amounts in a few categories. 976 of the 999 cards have exactly one such burst; the other 23 have no fraud.
    A burst is uninterrupted: once it starts, the card's transactions are fraud until it ends. So the card's
    transactions just before a given one are the strongest clue ([section 5.3](#53-features)).
  - Merchant coordinates are random points near the cardholder's home. Location-based features are therefore
    noise in this dataset.
  - Fraud labels are per burst, so the test set's 2,145 fraud transactions come from only **218 cards**. That
    limits how precisely differences between models can be measured (see [Results](#6-results)).

---

## 5. How the pipeline works

### 5.1 Problem and metric
Binary classification of each transaction as fraud or not. With 0.5% fraud, accuracy is meaningless: predicting
"never fraud" scores 99.5%. The primary metric is therefore:
- **PR-AUC** (average precision): how well fraud is ranked above legitimate transactions. A random model scores
  about the fraud rate (0.004 on test); a perfect model scores 1.
- **Precision, recall and F1** at the model's decision threshold: of the flagged transactions, how many are
  fraud, and how much of the fraud is flagged.
- **ROC-AUC** is reported too, but it is close to 1 for every reasonable model here, so it barely separates them.

### 5.2 Time-based split
Fraud models are used on future data, so the split follows time:

| Part | Rows | Fraud (cards) | Period | Used for |
|---|---|---|---|---|
| **fit** | 1,102,173 | 6,373 (644) | 2019-01-01 → 2020-04-03 | training |
| **val** | 194,502 | 1,133 (120) | 2020-04-03 → 2020-06-21 | early stopping, decision threshold, choosing the best model |
| **test** | 555,719 | 2,145 (218) | 2020-06-21 → 2020-12-31 | final report only |

`fit` and `val` are the first 85% and last 15% of the training file by time (`VAL_FRAC` in `config.py`). `test` is
the test file.

### 5.3 Features
There are 105 numeric and 6 categorical features (`merchant, category, gender, state, job, city`). Almost all of
them describe a transaction **relative to earlier transactions** of the same card, merchant or category.

**Long-term profile** of the card, merchant and category:

| Group | # | What it captures | Examples |
|---|---|---|---|
| Base | 7 | Amount, time, cardholder age and city population, card-to-merchant distance | `amt`, `hour`, `dow`, `age` |
| User history | 11 | The card's spending over the previous 3 / 7 / 30 days and all time; how this amount compares | `u_amt_mean_3`, `amt_vs_uall`, `acct_age_days` |
| User × merchant | 4 | How often the card used this merchant before | `um_cnt_30`, `um_cnt_all` |
| Merchant profiles | 9 | The merchant's and category's typical amounts (all cards, earlier transactions) | `amt_z_m`, `amt_vs_cat` |
| Job buying power | 3 | Typical spending estimated from the job title, and this amount vs that estimate | `job_bp_model`, `amt_vs_job_bp` |
| Merchant similarity | 4 | Has the card recently shopped at merchants similar to this one? (see [5.4](#54-keeping-the-merchant-features-leak-free)) | `sim_max_7` |
| Merchant location | 4 | Unusual distance or merchant location; how common this merchant + state pair is (see [5.4](#54-keeping-the-merchant-features-leak-free)) | `mdist_z`, `ms_pair_n` |
| Time since last | 3 | Time since the card's previous transaction, and since its previous one at this merchant | `log_since_last` |

**Recent activity** of the card:

| Group | # | What it captures | Examples |
|---|---|---|---|
| Previous transactions | 22 | The card's previous 5 transactions: amount vs category average, hour, hours since; also the amounts and categories of the last 3 | `lag1_amt_vs_cat`, `hours_since_lag2` |
| Velocity | 10 | Number, mean and total amount of the card's transactions in the last 1 h / 6 h / 24 h; its largest amount in 24 h / 72 h | `u_amt_mean_24h`, `u_cnt_6h` |
| Recent unusual activity | 6 | Night purchases above twice the category average in the last 24 h / 72 h; mean and largest amount-vs-category | `u_odd_cnt_24h`, `u_avc_max_72h` |
| Last 5 / 10 transactions | 6 | Share at night; mean and largest amount-vs-category | `last5_avc_max` |
| Night activity | 4 | Night transaction (22:00–03:59)? Night transactions in the last 24 h / 72 h; share among all earlier ones | `is_night`, `u_night_cnt_24h` |
| Card × category | 4 | How much the card used this category before; this amount vs its own average there | `uc_frac_all`, `amt_vs_uc` |
| Card habits | 6 | Share of earlier transactions at this hour; transactions so far today vs a typical day; amount vs the card's largest earlier amounts | `today_vs_typical`, `amt_vs_umax` |
| Cardholder segment | 2 | This amount vs earlier purchases in the category by similar cardholders (gender × age band × urban / rural) | `amt_z_segcat` |

**Why recent activity matters.** The previous model (v1), with long-term features only, missed fraud that looks
ordinary on its own. The median missed transaction was the 4th of its burst: about $48, in the daytime, a normal
amount for its category. The card's previous transactions give such transactions away, for example several large
night purchases an hour earlier. This took validation PR-AUC from 0.978 to 0.995
([section 6.2](#62-why-v2-outperforms-v1); details in `experiments/README.md`).

**Not used as model inputs:**
- **Raw coordinates** (`lat`, `long`, `merch_lat`, `merch_long`). They stay in the feature table for reference.
  Home coordinates identify the card, so a model could memorise cards; merchant positions are random in this
  data.
- **Travel and distance-from-usual-location features.** v1 had them; they were removed for the same reason, with
  no loss in accuracy.

**Every history feature uses only transactions strictly before the current one, and never a fraud label.**
That is what makes them computable at swipe time, and it is enforced by `check_leakage.py`.
Most are computed by `features.win_stats`: for every row, the count and mean of a value over earlier rows of
the same group (card, merchant, …) within a time window. It runs in O(n log n) using prefix sums and binary
search.

The **job buying-power** features come from a small model: TF-IDF of the job title plus state, with ridge
regression on age. It is fitted on the training period and saved (`artifacts/job_buying_power.pkl`), so
inference can apply it to new rows. Cards seen in training get out-of-fold estimates, so a card's own spending
never feeds its own feature.

### 5.4 Keeping the merchant features leak-free
The merchant-similarity and merchant-location features need statistics about the whole population: which
merchants share customers, how far a merchant's customers usually live from it, how often cardholders from a
given state shop there. A tempting shortcut is to compute these once, as lookup tables over **all non-fraud
transactions of the training period**, and join them back onto every row, including the training rows they were
built from. For a training transaction on date *t*, the value then depends on the following (the numbers were
measured on this dataset during development):
1. **its own label:** a legitimate row counts itself, while a fraud row was filtered out. For example,
   `ms_pair_n == 0` then occurs only on fraud rows (17 of 17), purely because they removed themselves;
2. **future transactions:** everything from *t* to the end of the training period. On average half of each
   table comes from after the row; for January 2019 rows, 98%;
3. **future and unconfirmed fraud labels**, through the non-fraud filter.

None of this is known when a card is swiped, so a model trained this way learns from information it will not
have in production. This project builds both groups from earlier transactions only, with no labels:

| Feature(s) | Lookup-table shortcut (leaky) | This project |
|---|---|---|
| `sim_mean_7`, `sim_max_7`, `sim_mean_30`, `sim_max_30` | One merchant-similarity matrix from all non-fraud training rows | Matrix rebuilt at the start of each month from all earlier transactions; each row uses its own month's matrix. The first month (Jan 2019) has no history, so it gets NaN |
| `ms_pair_n` | Non-fraud training purchases at this merchant by cardholders from this state, including the row itself and later rows | Running count of earlier purchases for this merchant + state |
| `mdist_z`, `cdist_z` | Distance z-score vs the merchant's / category's non-fraud training mean and std | z-score vs the running mean and std of earlier transactions (needs more than 5, else 0) |
| `mloc_dev_km` | Distance to the merchant's non-fraud training median location | Distance to the running mean location of the merchant's earlier transactions |

The shortcut versions live in `leaky_reference.py`. They are never used to train or score models, only as a
known-leaky control that `check_leakage.py` must flag. During development (with v1's features),
XGBoost trained with the shortcut versions scored test PR-AUC 0.9521, against 0.9508 with the leak-free versions
(mean over 4 seeds). That is within seed-to-seed noise: **avoiding the leak is about correctness, and it costs
no measurable performance.**

### 5.5 Models
All three use the same training procedure:
- train on `fit`, with at most 2,000 boosting rounds (early stopping ends training after a few hundred);
- after every round, compute validation PR-AUC and stop after 50 rounds without improvement;
- keep the best round;
- choose the decision threshold that maximises F1 on validation, and apply it unchanged to test and to new data.

| Model | Categorical handling | Settings |
|---|---|---|
| **CatBoost** | Native: target statistics computed in time order (`has_time=True`), so a training row is encoded only from earlier rows | learning rate 0.1, depth 6 |
| **XGBoost** | One-hot for ≤ 60 levels (category, gender, state); frequency encoding for merchant, job, city | learning rate 0.1, depth 7, 80% row/column sampling |
| **LightGBM** | Same encoded matrix as XGBoost | learning rate 0.1, depth 7 / 127 leaves, 80% row/column sampling, `min_sum_hessian_in_leaf=1`, `lambda_l2=1` |

LightGBM's two regularisation settings copy XGBoost's defaults (`min_child_weight=1`, `lambda=1`). In development
runs with LightGBM's own defaults (`1e-3` and `0`), leaves holding only a few fraud rows got extreme values on this
imbalanced data, and early stopping ended training after 24 rounds at validation PR-AUC 0.92.

The **best model** is the one with the highest **validation** PR-AUC. The test set is never used for any choice.

---

## 6. Results
From `python train.py` (seed 42); also in `results/metrics.csv` and `artifacts/metadata.json`.

| Model | Val PR-AUC | **Test PR-AUC** | Test ROC-AUC | Test precision | Test recall | Test F1 | Threshold | Rounds | Train time* |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** (best on validation) | **0.9952** | 0.9873 | **0.9996** | 0.974 | 0.947 | 0.960 | 0.233 | 473 | 42 s |
| XGBoost | 0.9951 | **0.9880** | 0.9994 | 0.973 | **0.953** | 0.963 | 0.176 | 412 | 59 s |
| CatBoost | 0.9945 | 0.9862 | 0.9993 | **0.992** | 0.942 | **0.966** | 0.213 | 663 | 165 s |

\* Measured on an Apple M4 (10 cores) with little else running. Times depend heavily on machine load: the
`train_secs` column of `results/metrics.csv` records your latest run. Feature building adds 30–90 s. All other
columns are deterministic and match `results/metrics.csv`.

### 6.1 What the numbers mean
- **The three models are effectively tied.** On test, the best model flags 2,085 of 555,719 transactions; 97% of
  those are fraud, and they cover 95% of all fraud transactions.
- **Differences of about 0.002 PR-AUC are noise.** Over 3 seeds (42, 1, 2), test PR-AUC was 0.9878 ± 0.0005
  (LightGBM), 0.9876 ± 0.0004 (XGBoost) and 0.9864 ± 0.0006 (CatBoost), and the test fraud comes from only 218
  cards. Compare future experiments on validation PR-AUC over several seeds (`experiments/`).
- **Validation scores higher than test** (0.995 vs 0.987) partly because fraud is rarer in test (0.39% vs 0.58%),
  which lowers PR-AUC. The threshold chosen on validation then trades slightly differently on test.

### 6.2 Why v2 outperforms v1
**v1**, the previous model, was CatBoost on 69 features that describe only the card's long-term profile: its
spending over the previous 3 / 7 / 30 days and all time, merchant and category norms, job buying power, merchant
similarity and location, and travel between transactions. It is kept under the git tag
[`v1`](https://github.com/drduron/650_Group_Project/tree/v1/modelling). Test results, mean of 3 seeds:

| Model | v1: test PR-AUC | precision / recall | v2: test PR-AUC | precision / recall |
|---|---|---|---|---|
| LightGBM | 0.9513 ± 0.0024 | 0.931 / 0.876 | **0.9878 ± 0.0005** | 0.974 / 0.951 |
| XGBoost | 0.9507 ± 0.0015 | 0.945 / 0.864 | **0.9876 ± 0.0004** | 0.971 / 0.956 |
| CatBoost | 0.9509 ± 0.0020 | 0.942 / 0.876 | **0.9864 ± 0.0006** | 0.985 / 0.951 |

- **Fewer misses and fewer false alarms.** Comparing the saved best models (v1 CatBoost, v2 LightGBM), v2 misses
  **114** of the 2,145 test fraud transactions instead of **302**, with **54** false alarms instead of **94**.
- **Far beyond noise.** The part of PR-AUC still missing shrank by about three quarters (from 0.049 to 0.013).
  Resampling the test cards 300 times, v2 scored higher every time (95% range of the gain: +0.030 to +0.050).

**Why, most important first:**
1. **v2 sees fraud bursts.** A compromised card produces an uninterrupted burst of fraud lasting up to about 3
   days. v1's shortest look-back was 3 days, so it judged each transaction mostly on its own, and most of its
   misses were ordinary-looking transactions in the middle of a burst. v2's recent-activity features (the
   previous 5 transactions, the last 1–72 hours, recent unusual night purchases) recognise that the card has just
   behaved abnormally. This change produced almost all of the gain.
2. **Less noise to memorise.** Home coordinates identify the card, and merchant positions are random in this
   data. v2 drops them and the travel features built on them.
3. **Cleaner training.** CatBoost builds its category statistics from earlier training rows only
   (`has_time=True`), and early stopping may run up to 2,000 rounds instead of 600.

Points 2 and 3 are small clean-ups; their effect on accuracy is within noise.

Every choice was made on validation, using two windows: the production split (A) and an earlier, independent one
(B: fit on the first 70% of the training file, validate on the next 15%). Validation PR-AUC (LightGBM, mean of 3
seeds):

| Step | Window A | Window B |
|---|---|---|
| v1 features | 0.9775 | 0.9798 |
| + the card's previous 3 transactions | 0.9931 | 0.9934 |
| + all recent-activity groups | 0.9955 | 0.9967 |
| − raw coordinates and travel features (v2) | 0.9956 | 0.9971 |

- **No gain:** removing card-identifying features, averaging the three models, retraining on fit +
  validation, and other LightGBM settings.
- **Promising, not included yet:** score propagation, i.e. features built from the model's scores for the card's
  earlier transactions. It reached test PR-AUC 0.991, but it needs a second model in training and inference.
- **Full record:** `experiments/README.md`.

### 6.3 Most important features
From `results/feature_importance/`. Each library measures importance differently, so compare rankings, not values:
- **LightGBM (gain):** `amt_z_segcat` (36%), `u_amt_mean_24h` (31%), `category_grocery_pos` (7%),
  `u_odd_cnt_72h` (6%), `is_night` (3%).
- **XGBoost (gain):** `category_grocery_pos` (50%), `u_odd_cnt_24h` (27%), `u_amt_mean_24h` (9%), `is_night` (2%).
- **CatBoost:** `is_night` (12%), `category` (7%), `amt` (7%), `m_amt_std_all` (4%), `amt_z_segcat` (3%).
- **Overall:** the 60 recent-activity features carry 85% (LightGBM), 45% (XGBoost) and 52% (CatBoost) of the
  total importance.

---

## 7. Inference: scoring transactions with the saved model
`predict.py` loads a saved model and scores transactions. The best model (LightGBM) is included in `artifacts/`,
so this works right after `download_data.py`. XGBoost and CatBoost are not included: run `python train.py` to
create them, then use `--model xgboost` or `--model catboost`.

```bash
python predict.py                          # the whole test file, best model
python predict.py --limit 1000             # only the first 1,000 test transactions (time order)
python predict.py --model xgboost          # a specific saved model
python predict.py --input new.csv --history data/fraudTrain.csv data/fraudTest.csv --output new_scores.csv
```

**Why it needs history.** Features such as "the card's previous transaction" or "amount vs this card's average
over the last 3 days" depend on earlier transactions. So `predict.py` reads the transactions to score (`--input`, default `data/fraudTest.csv`)
**and** the earlier transactions (`--history`, default `data/fraudTrain.csv`). It then:
1. builds the features for all of them with the same code as training, plus the saved job buying-power model;
2. keeps the input rows;
3. applies the saved model and its saved decision threshold.

For new data, pass every earlier transaction you have as history. With less history the features are still
valid, but they describe a shorter past. `--input` files may omit `is_fraud`, which is how unlabelled live data
is scored.

**Output** (`results/predictions_<model>.csv` by default), one row per input transaction, in input order:

| Column | Meaning |
|---|---|
| `trans_num`, `cc_num`, `trans_date_trans_time`, `merchant`, `category`, `amt` | identify the transaction |
| `is_fraud` | true label (only when the input has it) |
| `fraud_probability` | model score between 0 and 1 |
| `fraud_flag` | 1 if `fraud_probability >= threshold` (the threshold chosen on validation), else 0 |

**Built-in verification.** When the input has labels, `predict.py` prints PR-AUC, precision, recall and F1. If
the input is the full test file, it also compares these with the test metrics recorded at training time
(`MATCH`). Whenever train.py's test predictions are available (`results/test_predictions/`, written by `train.py`
and too large for git), it also compares the probabilities transaction by transaction (`IDENTICAL`). See the
expected output in [Quick start](#1-quick-start).

---

## 8. Leakage check
```bash
python check_leakage.py        # ~1-3 min; exit code 0 = PASS, 1 = FAIL
```
It builds every feature twice:
- **A.** on the full data;
- **B.** on the data cut off at 2019-09-01, with the fraud labels randomly shuffled.

It then compares the values for the 573,959 transactions before the cut-off. A feature that uses only earlier
transactions and no labels is unaffected by both changes, so its values are identical. A feature that peeks at
later transactions or at labels changes. Expected output:

```
feature group                      features unchanged   result
base                                      7         7   ok
user history                             11        11   ok
user x merchant                           4         4   ok
merchant profiles                         9         9   ok
job buying power                          3         0   look-ahead within training period (expected, no labels)
merchant similarity                       4         4   ok
merchant location                         4         4   ok
time since last                           3         3   ok
velocity (1 h - 72 h)                    10        10   ok
night activity                            4         4   ok
card x category                           4         4   ok
previous transactions                    22        22   ok
recent unusual activity                   6         6   ok
last 5 / 10 transactions                  6         6   ok
card habits                               6         6   ok
cardholder segment                        2         2   ok
leaky control (leaky_reference.py)        8         0   flagged as leaky (expected)

history features that use future data or labels: none
positive control (leaky_reference.py) flagged: yes
RESULT: PASS
```
For v2, the same test was also run with cut-offs inside the validation period (2020-05-15) and the test period
(2020-10-01), so that rows from those periods were covered too. All 102 history features were again unchanged.

The **leaky control** rows are the deliberately leaky features from
[section 5.4](#54-keeping-the-merchant-features-leak-free). The check must flag them; if it does not, the check itself
is broken. The job buying-power features are fitted on the whole training period. That is look-ahead for training
rows, but they never use fraud labels, and validation, test and new data only see training-period information.
Run this check after adding or changing any feature.

---

## 9. Script and module reference

### `download_data.py`
Downloads `kartik2112/fraud-detection` from Kaggle's public endpoint, extracts only `fraudTrain.csv` and
`fraudTest.csv`, and verifies their row counts. Options: `--data-dir`, `--zip <file>` (use a zip you already
have), `--force`.

### `train.py`
The end-to-end training pipeline:
1. load and clean both CSVs;
2. compute the history features;
3. split into fit / val / test;
4. fit and apply the job buying-power model;
5. for each model, train with early stopping, save, reload, predict validation and test, choose the threshold on
   validation, and evaluate;
6. record the best model.

Options: `--models` (any of `catboost xgboost lightgbm`), `--seed`, `--data-dir`, `--artifacts-dir`,
`--results-dir`. Training only some models keeps the `metadata.json` entries of the other models saved in
`artifacts/`, so the best model is still chosen across all saved models. With the default folders, `train.py`
replaces the included model with a freshly trained copy. Use different `--artifacts-dir` / `--results-dir`
folders for experiments (e.g. `runs/seed1/...`, which is git-ignored) so the saved artifacts are not
overwritten.

Writes:
- `artifacts/<model>/`: `model.cbm` / `model.json` / `model.txt`, plus `encoder.pkl` (XGBoost/LightGBM) and
  `model_info.json` (feature lists, threshold, rounds)
- `artifacts/job_buying_power.pkl`, `artifacts/metadata.json` (best model, all metrics, split, library versions)
- `results/metrics.csv`, `results/feature_importance/<model>.csv`, `results/test_predictions/<model>.csv`,
  `results/train.log`

### `predict.py`
Inference; see [section 7](#7-inference-scoring-transactions-with-the-saved-model). Options: `--input`,
`--history`, `--model`, `--limit`, `--output`, `--artifacts-dir`, `--results-dir`.

### `check_leakage.py`
See [section 8](#8-leakage-check). Option: `--data-dir`.

### Modules
| Module | Contents |
|---|---|
| `config.py` | Default folders (relative to this project), file names, `VAL_FRAC = 0.15`, `SEED = 42` |
| `data.py` | `read_transactions` (read + check columns), `prepare` (types, timestamp, sort by card and time; fails loudly on unparseable values), `assign_partitions` (fit / val / test) |
| `features.py` | Feature lists, `win_stats` and `_rolling` (past-only window statistics), `add_history_features` (one function per feature group, listed in `HISTORY_STEPS`), `JobBuyingPower` |
| `models.py` | `CatBoostModel`, `XGBoostModel`, `LightGBMModel` (common `fit` / `predict_proba` / `save`), `load_model`, `MatrixEncoder`, `best_f1_threshold`, `evaluate` |
| `leaky_reference.py` | The leaky lookup-table versions of the merchant features ([section 5.4](#54-keeping-the-merchant-features-leak-free)), used only by `check_leakage.py` as a control |

### Adding a feature
1. Try it first in `experiments/`: write it as a candidate group in `experiments/candidates.py`, test it for
   leakage with `experiments/check_candidates.py`, and compare validation PR-AUC over several seeds and both
   validation windows (`experiments/README.md`).
2. If it helps, add a function to `features.py` that uses only earlier rows (`win_stats` and `_rolling` help),
   append it to `HISTORY_STEPS`, and add its column names to a feature list (they flow into
   `NUMERIC_FEATURES`).
3. Run `python check_leakage.py`; it must still PASS.
4. Run `python train.py --artifacts-dir runs/<name>/artifacts --results-dir runs/<name>/results`, and compare
   validation PR-AUC with the results above.

---

## 10. Reproducibility
- **Deterministic.** With the same data, library versions and seed, `train.py` reproduces the numbers above;
  rerunning gives identical metrics. `predict.py` rebuilds identical features and probabilities (checked
  automatically, see [section 7](#7-inference-scoring-transactions-with-the-saved-model)).
- **Library versions** are pinned in `requirements.txt` and recorded in `artifacts/metadata.json`. Other versions,
  or other CPU types, may change results in the last decimals.
- **Runtime** on an Apple M4 (10 cores), depending on what else the machine is doing: `train.py` 5–10 minutes,
  `predict.py` on the test file 0.5–2 minutes, `check_leakage.py` 1–3 minutes, `download_data.py` under a minute
  on a fast connection.

---

## 11. Known limitations and next steps
Deliberate simplifications:
- **Job buying-power look-ahead.** Fitted on the whole training period (no labels involved).
- **Remaining location features** (`dist_um_km` and the merchant-location group) are noise in this simulated
  data. They are kept because they cost nothing measurable; the raw coordinates and travel features were removed.
- **Light tuning only.** Learning rate, feature sampling and the round limit were tried, with no gain. The models
  are trained on `fit` only; retraining on fit + validation made no difference (`experiments/README.md`).

Next steps:
- **Score propagation** (features from the scores of the card's earlier transactions) raised test PR-AUC from
  0.988 to 0.991 in experiments (`experiments/README.md`, section 6). Adopting it means training and serving a
  second model.

Things to keep in mind when reading the scores:
- **Simulation shortcuts inflate them compared with real data.** Fraud arrives in tight, uninterrupted bursts,
  which the recent-activity features exploit, and 91 cards exist only for their fraud burst (7.6% of test fraud
  is on such brand-new cards). Real fraud also comes in bursts, but less cleanly.
- **Metrics are per transaction.** A card is usually blocked at the first alert, so the share of fraud bursts
  caught, and how early, would be a more practical measure.
- **Inference is batch.** It recomputes features over the full history each run. A live system would maintain
  these running statistics incrementally.

---

*Dataset: "Credit Card Transactions Fraud Detection Dataset" by kartik2112 on Kaggle (CC0), generated with
Brandon Harris's Sparkov Data Generation.*

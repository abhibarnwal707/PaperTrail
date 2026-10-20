# PaperTrail — EDA summary (Phase 1)

Source: `notebooks/01-eda.ipynb`, IBM Transactions for AML, **HI-Small** (synthetic data).
All numbers below are exact, taken from the executed notebook, `reports/daily.csv` and `reports/typology_counts.csv`.

## 1. Dataset at a glance

| Item | Value |
|---|---|
| Transactions | **5,078,345** |
| Laundering transactions (`label = 1`) | **5,177** |
| Laundering rate | **0.1019%** (about 1 in 980) |
| Unique accounts (bank + account ID) | **515,088** |
| Time range | 2022-09-01 00:00 → 2022-09-18 16:18 (18 calendar days) |
| Columns | 11 raw + `src`, `dst`, `txn_id` built by us |

Account numbers are only unique within a bank, so `src`/`dst` = `{bank}_{account}`.

## 2. Findings

### F1. Extreme class imbalance
Only 0.10% of transactions are laundering. Accuracy is meaningless (predicting "normal" always scores 99.9%), which justifies the PR-AUC / minority-F1 / recall@1%FPR / precision@k protocol. On the training split (2,530 positives in 3,248,921 rows) neg/pos ≈ 1,283, so expect `scale_pos_weight` around 1,300 for XGBoost.

### F2. The time axis is not uniform, and the tail is distorted (matters for the split)
- Volume is highest at the start: Sep 1 has 1,114,921 rows and Sep 2 has 754,449. Weekend days have about 207K each, and Mon–Thu about 482K each. Sep 9 has 654,467.
- Normal traffic effectively **stops after Sep 10**. Sep 11 has 396 rows, Sep 12 has 281, Sep 13 has 184, Sep 14 has 121, Sep 15–16 have 46 each, Sep 17 has 23 and Sep 18 has 11 (1,108 rows in total).
- Laundering does **not** stop. It runs at 322–539 per day on Sep 1–10, then decays: Sep 11 = 232, Sep 12 = 170, Sep 13 = 106, Sep 14 = 70, Sep 15 = 28, Sep 16 = 26, Sep 17 = 15, Sep 18 = 8 (**655 positives**, 12.7% of all).
- So on Sep 11–18 the laundering rate jumps to **56–73%** against 0.03–0.21% on normal days. These are the tails of laundering chains started earlier: **645 of the 655** tail positives belong to attempts that started before Sep 11.
- Laundering positives per day are roughly constant, regardless of background volume. The daily laundering *rate* therefore mostly tracks volume, not behaviour.

**Decision:** treat Sep 11–18 as an artifact tail. Exclude it from train/val/test evaluation, since it would inflate the base rate in the test set. It removes 655 positives (12.7% of all) and only 1,108 rows. Keep it for pattern parsing and document the count. Use a **day-aligned split** (see playbook Step 2.1).

### F3. Laundering is concentrated in one payment format
| pay_format | rows | laundering | rate |
|---|---|---|---|
| **ACH** | 600,797 | **4,483 (86.6% of all positives)** | **0.746%** (about 7× the base rate) |
| Bitcoin | 146,091 | 56 | 0.038% |
| Cash | 490,891 | 108 | 0.022% |
| Cheque | 1,864,331 | 324 | 0.017% |
| Credit Card | 1,323,324 | 206 | 0.016% |
| Wire | 171,855 | **0** | 0 |
| Reinvestment | 481,056 | **0** | 0 |

ACH is the single strongest categorical signal. Wire and Reinvestment never contain laundering in HI-Small. Be honest in the README that this is a property of the synthetic generator.

### F4. Currency matters little, with one outlier
Most currencies have a rate near the base (Euro 0.117%, US Dollar 0.101%, Yen 0.100%). **Saudi Riyal is about 4× higher** (0.420%, 374 positives from 89,014 rows). Bitcoin is lowest (0.038%). Currency may be confounded with `pay_format`, so check this by ablation rather than assuming it.

### F5. Amounts: laundering sits in a narrow mid-to-high band
On a log10 scale, laundering transactions peak sharply around **10^3.5–10^4.5 (about $3K–$30K)**, with a tight spike near 10^4. Normal transactions are broader and lower, peaking around 10^2–10^3.5. Only normal transactions have a mass of **near-zero amounts**. Both classes share a long right tail up to about 10^12. The distributions overlap heavily, so amount is a useful feature but a "high amount" rule alone will have poor precision.

Training-window laundering amounts (Phase 3 check): 5% = 172, 25% = 2,224, median = 7,033, 75% = 18,666, 90% = 137,550, 95% = 782,357. The 3K–30K band therefore holds roughly **half** of the laundering transactions: it is a peak with a long right tail, not a band that contains most of them.

### F6. Self-transfers and cross-currency transactions are almost never laundering
- Self-transfers (`src == dst`): 591,212 rows (11.6%), laundering rate 0.0019%, so only about 11 positives.
- Cross-currency (`cur_paid != cur_recv`): 72,170 rows (1.4%), **zero** positives.

Both are strong "this is safe" signals. They also mean the playbook's rule "cross-currency and amount high" will not work in this data. Note it as a finding about the synthetic generator.

### F7. Hour of day is nearly flat, and weekday is confounded
- Hour 0 has about 634,726 rows (3.3× any other hour). This is the day-1 midnight burst, and the first rows are `Reinvestment` self-transfers. Its laundering rate is low (0.028%). All other hours run at about 0.07–0.12%, with no strong pattern.
- Weekday rates look striking (Sat 0.204%, Sun 0.311% vs Thu 0.056%, Fri 0.067%). **This is an artifact**, not a behavioural signal. Weekends have low volume while laundering volume stays constant, and Sun/Sat also contain the Sep 11 tail. Thursday and Friday are inflated by the huge Sep 1–2 volume.
- **Implication:** do not use weekday or day-of-month as model features. Hour is safe but weak. Do not interpret weekday rates in the README as a finding about laundering behaviour.

### F8. The account graph is heavy-tailed
| Statistic | Total degree (in + out) |
|---|---|
| Median | 6 |
| 75th percentile | 27 |
| Mean | 19.7 |
| 99th percentile | **119** |
| Max | **169,756** |

The log-degree histogram has two modes, a spike of about 140K accounts with degree about 3 and a second bump around degree 25. A handful of super-hub accounts have degree in the hundreds of thousands. This justifies log-transforming degree features, defining hubs as the top 1% (degree ≥ 119), and capping 3-cycle counting.

### F9. The patterns file explains only 62% of the positives
`HI-Small_Patterns.txt` contains 3,209 laundering transactions in 370 attempts, and **every one of them matches a transaction** (0 unmatched, 0 duplicate keys). That covers 3,209 of the 5,177 positives (**62.0%**). The other **1,968 (38.0%) have no typology**.

| Typology | Positives | Share of all positives | Attempts |
|---|---|---|---|
| GATHER-SCATTER | 716 | 13.8% | 51 |
| SCATTER-GATHER | 626 | 12.1% | 44 |
| STACK | 466 | 9.0% | 43 |
| FAN-OUT | 342 | 6.6% | 48 |
| FAN-IN | 318 | 6.1% | 40 |
| CYCLE | 287 | 5.5% | 54 |
| BIPARTITE | 263 | 5.1% | 49 |
| RANDOM | 191 | 3.7% | 41 |
| **No typology** | **1,968** | **38.0%** | n/a |

- Every typology is essentially 100% ACH (STACK has one non-ACH row). The unmapped group is 64.8% ACH, and it contains **all 694 non-ACH positives** (Bitcoin, Cash, Cheque, Credit Card) plus about 1,275 ACH ones.
- So the typology labels describe only the structured, ACH-based schemes. The reason the rest are unlabelled is not documented in the data, so do not claim one in the README.
- **Implication:** the agent's typology ground truth exists only for the 3,209 mapped transactions. Agent evaluation must use accounts whose positives are mapped, and unmapped alerts must be reported separately.

### F10. Laundering attempts are long
Across the 370 attempts, the span from first to last transaction has a median of 75 hours, a mean of 67 hours (std 46) and a maximum of 202 hours (about 8.4 days). That is why attempts that begin before Sep 11 spill into the tail, and why a split cut can fall in the middle of an attempt. Whole attempts can straddle train/val or val/test, so note it as a limitation of any time-based split.

## 3. Implications for the modeling plan

| Area | Change |
|---|---|
| Split | Day-aligned, trimmed at Sep 10 (train Sep 1–6, val Sep 7–8, test Sep 9–10; see section 3). |
| Rule baseline | Use ACH, a 3K–30K amount band, and high 24h send count. Exclude self-transfers, Wire and Reinvestment. Drop the cross-currency rule. (Phase 3 result: the send-count rule never fired, see section 5.) |
| Features | Add `is_ach`, an amount-band flag, `is_self_transfer` and `is_cross_currency`. Drop weekday and date. |
| Imbalance | `scale_pos_weight` / `pos_weight` ≈ neg/pos on the train window (about 1,280 on the train split). |
| Graph features | Log-transform degrees. Hub = top 1% (≥ 119). Cap cycle counts because of super hubs. |
| Evaluation | Never report accuracy. Also report metrics on ACH-only transactions, since 87% of positives are ACH. |
| Agent evaluation | Typology ground truth exists for 62% of positives only. Pick showcase accounts from mapped attempts, and report unmapped alerts separately. |

## 4. Caveats for the README

- The data is synthetic (IBM AMLworld generator). Findings such as "ACH is risky" or "cross-currency is safe" describe the generator, not real banks.
- The Sep 11–18 laundering tail is a data-generation artifact. Say how it was handled and why.

## 5. Baseline results (Phase 3, `notebooks/02_baseline.ipynb`)

Both models use the fixed split in `reports/split.md`, pick their threshold on validation, and score the test split once (test = 862,792 rows, 956 positives, base rate 0.111%). MLflow runs `rules_baseline` and `xgb_v1_tabular` are on DagsHub. Raw numbers: `reports/baseline_table.csv`, `reports/baseline_metrics.json`, `reports/xgb_v1_importance.csv`.

| Model | PR-AUC | F1 (minority) | Recall@1%FPR | Precision@1000 | ACH-only PR-AUC | ACH-only F1 |
|---|---|---|---|---|---|---|
| Rules | 0.0080 | 0.030 | 0.012 | 0.000 | 0.019 | 0.049 |
| XGB tabular (v1) | **0.0461** | **0.116** | **0.409** | **0.116** | 0.052 | 0.121 |

Validation PR-AUC for XGBoost is 0.0493, close to its test value of 0.0461, so the model is not overfitting.

### How to read the numbers
- **Random guessing** gives PR-AUC equal to the base rate (0.0011) and F1 of about 0.002. XGBoost v1 is about 42× the base rate in PR-AUC and about 50× in F1. The rules are about 7× the base rate in PR-AUC.
- **Recall@1%FPR = 0.409:** flagging the 1% highest-scored normal transactions (about 8,600 rows) catches about 391 of the 956 laundering transactions. This is the most useful headline number for a bank, because it fixes the false-alarm budget.
- **Precision@1000 = 0.116:** about 116 of the 1,000 highest-scored test transactions are laundering.
- **F1 = 0.116 is low in absolute terms.** At a 0.1% base rate, F1 is a harsh metric: it needs precision and recall to be high at the same time. This is a baseline with no account history and no graph structure, so a low F1 is expected and is the gap Phases 4–5 are meant to close. Do not describe it as good performance in the README.

### What the baselines show
1. **ACH dominates XGBoost.** `is_ach` and `fmt_ACH` (the same information, twice) carry about 80% of the total gain (0.457 + 0.346). Other features each add 0.4–2.7%, including `log_amt` (1.2%) and `in_amount_band` (1.4%). The v1 model has mostly learned "ACH transactions are riskier".
2. **Only one rule works.** The ACH-and-3K–30K rule flagged 26,685 validation rows at 1.96% precision (about 18× the base rate) with 50.5% recall.
3. **The amount rule is nearly useless.** Its threshold, the training 99th percentile, is 15,952,735 because amounts are extremely heavy-tailed. It flagged 7,020 validation rows at 0.19% precision with 1.3% recall.
4. **The send-count rule never fired.** Its threshold was the 99th percentile of the 24-hour send count over all training rows, which came out at 19,445. A few hub accounts send tens of thousands of transactions per day and dominate that distribution, so no validation transaction exceeded it (0 rows flagged). The rule contributed nothing, and the "Rules" row above is effectively one rule plus the amount rule. A corrected threshold is planned as a separate run (`rules_v2`) so that the original result stays on record.
5. **The rules' alert threshold is mostly an amount cut.** The validation-chosen threshold is 1.085. Since the score is the number of rules plus `log1p(amount)/100`, this means "ACH band rule fired and amount ≥ about $5,060", or the amount rule fired.
6. **Rules' precision@1000 is 0.** Ties are broken by amount, so the 1,000 highest-scored rows are the very large amounts flagged by the amount rule, which are almost never laundering.

### Caveats
- The rule thresholds were fitted on training data only, but the first run of the rules was seen on test. The corrected version will be reported as a second run, not swapped in silently.
- Test has 956 positives, so differences of a few percent between models are within noise.
- Numbers describe synthetic data.

## 5b. Rules v2 and graph-feature results (Phase 4, `notebooks/03-graph-feats.ipynb`)

Same split, same protocol (threshold on validation, test scored once). MLflow runs `rules_v2`, `xgb_v2_graph` and `xgb_v2_graph_shap` are on DagsHub. Feature building for all 5.08M rows took 46 s on Kaggle CPU.

| Model | PR-AUC | F1 (minority) | Recall@1%FPR | Precision@1000 | ACH-only PR-AUC | ACH-only F1 |
|---|---|---|---|---|---|---|
| Rules (v1) | 0.0080 | 0.030 | 0.012 | 0.000 | 0.019 | 0.049 |
| Rules v2 | 0.0070 | 0.025 | 0.014 | 0.000 | 0.019 | 0.050 |
| XGB tabular (v1) | 0.0461 | 0.116 | 0.409 | 0.116 | 0.052 | 0.121 |
| **XGB + history/graph (v2)** | **0.3512** | **0.392** | **0.629** | **0.357** | **0.401** | **0.433** |

### Rules v2: the fix did not help
- Training-set candidates for the send-count rule (`out_count_24h > N`): lift is 0.8 at N=5, 1.1 at N=10, about 1.85 for N from 50 to 1000, and 1.91 at N=5000. The best (N=5000) flags 59,986 validation rows at 0.13% precision, 7.6% recall. That is only about 1.2–1.9× the base rate, so the rule is alive but weak. Raw send volume is a poor signal by itself.
- Rules v2 test PR-AUC (0.0070) is slightly below v1 (0.0080). Report both and say the burst rule adds almost nothing. The one working rule is still the ACH band (1.96% precision, 50.5% recall).

### XGBoost v2: what changed
> **Correction after the ablation (section 5c):** 0.351 was the best of three seeds. The three-seed mean is 0.312 (std 0.034). Use 0.31 as the headline, and the explanation of *why* it works below was partly wrong.

- 51 features (16 tabular + 35 as-of history/graph). Same hyperparameters as v1; `scale_pos_weight` 1283.2, `best_iteration` 68 (training stopped at 118).
- PR-AUC rose from 0.046 to 0.351 (7.6×), F1 from 0.116 to 0.392, recall@1%FPR from 0.41 to 0.63, precision@1000 from 0.12 to 0.36. Validation PR-AUC is 0.369 (validation F1 0.421, recall@1%FPR 0.710), close to test 0.351, so the gain is not a test-set fluke. Test has 956 positives, so small differences are still noise, but this gap is not.
- Top gain: `fmt_ACH` 0.40, `is_ach` 0.30 (the same information twice), `s_uniq_out` 0.10 (distinct counterparties the sender has paid so far), then `is_cross_currency`, `s_out_cnt_24h`, `s_gap_s` and `fmt_Cheque` at 1–2% each. Gain share is not the same as contribution: ACH still has most of the gain, but v1 had that too and scored 0.046. The history features let the model separate risky ACH transactions from ordinary ones.
- The ACH-only slice also improves (PR-AUC 0.052 to 0.401), so the gain is not just from picking out ACH.

### Why I believe it, and what is still unchecked
- Every feature is as-of (unit tests confirm that appending later rows never changes earlier rows), and train/val/test features are built the same way.
- The ablation (5c) has now been done. It shows the gain is real but noisy, and that it depends on the `totals_timing` family, not on `s_uniq_out`. A check that the effect is not an artefact of the synthetic generator is still open. PageRank was computed but I have not checked its importance or whether it helps (it is null on day 1). From `xgb_v2_importance.csv` its gain is small: `s_pagerank` 0.0038, `d_pagerank` 0.0035, `has_reverse_before` 0.0041, each well under 1%, so the graph features barely matter and the gain comes from the account-history features.
- F1 of 0.39 is much better but still far from solved: at the validation-chosen threshold, precision and recall are both moderate.

## 5c. Ablation (Phase 4b, `notebooks/03b-ablation.ipynb`)

Question: where does the PR-AUC jump come from? 15 arms with identical parameters and protocol (threshold on validation, test scored once per arm), each logged to DagsHub as `abl_<arm>`. Raw numbers: `reports/ablation_table.csv`, `reports/ablation_metrics.json`, `reports/typology_recall.csv`. The 35 history features were grouped into four families: `windowed` (20: 24h/7d counts, sums, means), `uniq` (4: distinct counterparties so far), `totals_timing` (8: all-time counts, `gap_s` time since last event, `age_s` account age) and `graph` (3: `has_reverse_before`, two PageRanks).

### 1. The headline number was the lucky seed
| Full model (51 features) | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| Test PR-AUC | **0.351** | 0.295 | 0.291 | **0.312 ± 0.034** |
| Validation PR-AUC | 0.369 | 0.330 | 0.321 | 0.340 |
| best_iteration | 68 | 5 | 11 | |

The earlier 0.351 is the best of the three. The honest figure is about **0.31, still roughly 7× the tabular 0.046**, but any difference smaller than about 0.05 between arms is noise. The best iteration also varies from 5 to 68: validation `aucpr` peaks within a handful of trees and then falls, so early stopping on a single noisy validation curve is unstable. All results in this report should be read with that in mind, and future headline numbers should be multi-seed means.

### 2. Which features carry the gain
| Arm | Test PR-AUC | Val PR-AUC | Reading |
|---|---|---|---|
| tabular only (16) | 0.046 | 0.049 | baseline |
| tab + `totals_timing` | **0.223** | 0.260 | one family gets about 70% of the full mean |
| tab + `uniq` | 0.137 | 0.168 | |
| tab + `windowed` | 0.100 | 0.180 | |
| tab + `graph` | 0.100 | 0.116 | doubles tabular, still small |
| tab + `uniq_out` only (both sides) | 0.081 | 0.096 | |
| full (3 seeds) | 0.312 mean | 0.340 | |
| full − `graph` | 0.292 | 0.349 | within noise |
| full − `uniq` | 0.318 | 0.340 | within noise (no loss at all) |
| full − `windowed` | 0.278 | 0.293 | small, borderline |
| full − `totals_timing` | **0.128** | 0.221 | collapses |
| full − sender-side features | **0.133** | 0.152 | collapses |
| full − receiver-side features | **0.178** | 0.234 | large drop |

- **`s_uniq_out` is not what carries the model.** It had the highest gain among the new features, but removing the whole `uniq` family costs nothing beyond noise. Other features duplicate its information. Gain share misled me here, and the explanation in 5b ("fan-out") is not supported.
- **The `totals_timing` family is the critical one**: all-time counts, `gap_s`, `age_s`. Removing it drops PR-AUC from 0.31 to 0.13. Alone, it gets 0.22.
- **Both sides matter.** Removing the sender-side or receiver-side features each costs a lot, so the model uses the history of both accounts of the transaction.
- **PageRank and the reciprocal flag add little**, as the importance list suggested.
- Two arms (`tab+windowed`, `full-totals_timing`) ran to about 1,000 trees without early stopping and still scored lower, so training behaviour changes a lot with the feature set. This is another sign that the setup is not stable.

### 3. A concern to resolve before calling this real
`age_s` is "seconds since the account's first transaction **in this 18-day window**", and `gap_s`/`tot_*` also depend on where the window starts. Every account that exists on day 1 looks brand new, and ages can only reach about 6 days in train but about 10 days in test, so the feature's distribution shifts between splits. That may produce a signal that is partly a data-window artefact (accounts that appear only briefly, for example), partly real behaviour ("laundering accounts are newly active"). I have not tested this. Next experiment: ablate `age_s`, `gap_s`, `tot_out` and `tot_in` one at a time, compare their train and test distributions, and try a capped or relative age.

### 4. Recall by typology (test positives, validation-chosen thresholds)
| Typology | test positives | tabular | full | full − `uniq` | tab + `uniq` | full − `graph` | tab + `windowed` |
|---|---|---|---|---|---|---|---|
| UNMAPPED | **371 (38.8%)** | 0.01 | **0.00** | 0.00 | 0.01 | 0.00 | 0.05 |
| GATHER-SCATTER | 127 | 0.30 | 0.49 | 0.58 | 0.39 | 0.43 | 0.23 |
| SCATTER-GATHER | 110 | 0.26 | 0.41 | 0.51 | 0.20 | 0.45 | 0.42 |
| STACK | 84 | 0.36 | 0.49 | 0.52 | 0.24 | 0.52 | 0.32 |
| FAN-OUT | 65 | 0.28 | 0.37 | 0.45 | 0.25 | 0.37 | 0.26 |
| FAN-IN | 57 | 0.33 | 0.53 | 0.58 | 0.32 | 0.51 | 0.25 |
| CYCLE | 55 | 0.29 | **0.29** | 0.33 | 0.18 | 0.35 | 0.36 |
| BIPARTITE | 44 | 0.30 | 0.41 | 0.52 | 0.16 | 0.43 | 0.23 |
| RANDOM | 43 | 0.26 | 0.37 | 0.47 | 0.21 | 0.49 | 0.33 |

- **The model is blind to the unmapped positives.** 371 of the 956 test positives (39%) have no typology in the patterns file, and no arm recalls more than about 5% of them. This is the biggest single gap and it caps what any model with these features can reach. Understanding what these transactions are (amounts, accounts, timing, format) is now the top open question.
- **Mapped typologies** are recalled 0.3–0.6 by the full model, about 1.3–1.8× the tabular model. The weighted mean recall on mapped positives is about 0.43.
- **FAN-OUT is not the strongest** (0.37), contrary to my expectation. FAN-IN and GATHER-SCATTER are best. This is another sign that the `s_uniq_out` fan-out story was wrong.
- **CYCLE got no gain** from the history features (0.29 for full, same as tabular). This is the clearest case for the Phase 5 GNN to target.
- Single arms differ by up to 0.1 on typologies with 40–130 positives, so per-typology numbers carry large noise. Use them for direction, not for ranking arms.

### 5. What this means for the plan
1. Report **0.31 ± 0.03 (3 seeds)** as the XGBoost v2 result, not 0.351.
2. Before building on `totals_timing`, resolve the window-start concern in point 3.
3. Stabilise training (lower learning rate, more seeds, and a few trees' worth of regularisation) so that comparisons with the GNN are fair. Compare the GNN to a multi-seed XGBoost mean.
4. Investigate the 371 unmapped positives, because they cap recall.
5. The GNN case: cycles and the unmapped positives. Do not promise more than that.

## 5d. Account age, training settings and day 1 (Phase 4c, `notebooks/03c_stability.py`, `03d_stationary.py`)

Run locally on the Mac (03c, partly) and on Kaggle (03d). Raw numbers: `reports/age_distribution.csv`; 03d outputs `stationary_table.csv`, `stationary_metrics.json` once downloaded.

### Account age is a calendar artefact with no signal in val/test
`age_s` is "time since the account's first transaction in this 18-day window", not a real account age (the data has no opening dates). It therefore grows with the calendar.

| Split | Median sender age, normal rows | Median sender age, laundering rows |
|---|---|---|
| train (Sep 1-6) | 1.6 days | 3.1 days |
| val (Sep 7-8) | 6.9 days | 6.8 days |
| test (Sep 9-10) | 8.5 days | 8.7 days |

- Day 1 is 1.11M rows (22% of all rows) with 322 positives (0.03% against about 0.2% on days 3-4), and almost every sender looks brand new there. That makes laundering rows look older than normal rows in train, a relationship that does not exist in val or test.
- The share of senders with no earlier history is 13% of train normal rows but 0.1% of val/test normal rows. Where it happens in val/test, the laundering rate is 5-20x the base rate (but it covers under 1% of positives).
- **Removing `age_s` costs nothing:** full model test PR-AUC 0.434 +- 0.008, without age 0.429 +- 0.004 (3 seeds, stable settings). Age is dropped.
- More broadly, features whose meaning depends on how long the dataset has run drift: 7-day windows (never full in train), all-time totals, distinct counterparties so far, age, PageRank, the reverse-edge flag. Only windows of at most 24 h plus the gap since the last event are calendar-stable. `graph_features.recent_features` builds that stable set (30 features); 03d tests it.

### Training settings mattered as much as features
| XGBoost settings | Val PR-AUC | Test PR-AUC |
|---|---|---|
| original (depth 6, full class weight, 50 early stopping) | 0.318 +- 0.015 | 0.301 +- 0.014 |
| stable (depth 4, min_child_weight 10, lambda 10, subsample/colsample 0.7, sqrt class weight) | 0.498 +- 0.001 | 0.434 +- 0.008 |

The original settings stopped after about 30 trees (overfit to the train period at once); the stable settings use about 600 and have a seed spread 2-10x smaller. Even the tabular-only model rises from 0.046 to 0.065. The 0.312 from 5c was therefore partly a training-settings problem. The local Mac run reproduced the Kaggle number for the original settings (0.301 vs 0.312), so local and Kaggle runs are comparable.

### 03d (Kaggle), arms finished so far (stable settings, 3 seeds; threshold on validation)
| Arm | Train rows | Test PR-AUC | Test F1 | Recall@1%FPR | Precision@1000 |
|---|---|---|---|---|---|
| tabular | 3,248,921 | 0.065 +- 0.003 | 0.148 | 0.454 | 0.145 |
| tabular + burn-in | 2,134,000 | 0.063 +- 0.001 | 0.145 | 0.452 | 0.140 |
| v2 history (51 features) | 3,248,921 | **0.442 +- 0.004** | 0.499 | 0.675 | 0.461 |
| v2 history + burn-in | 2,134,000 | 0.435 +- 0.011 | 0.491 | 0.685 | 0.444 |

- **Burn-in (dropping day-1 rows from training) does not help**: no change for the tabular model, slightly worse (within noise) for v2 history. The hypothesis that the day-1 rows teach a harmful lesson is not supported; dropping them only loses data. Keep day 1 in training.
- Best iterations are 573-634 of 1,000 allowed, so early stopping now triggers on its own.
- The `recent` arms (stable feature set, with and without burn-in) were still running when this was written. **Pending:** whether the calendar-stable set matches or beats v2 history. If it is clearly lower, the cumulative features carry real signal that needs separate explanation.

### Current best XGBoost result to beat with the GNN
Test PR-AUC **0.442 +- 0.004** (3 seeds), F1 0.499, recall@1%FPR 0.675, precision@1000 0.461.

## 6. Open items

- [x] Save the full daily table (`reports/daily.csv`).
- [x] Step 1.3: parse the patterns file and map typologies (`reports/typology_counts.csv`).
- [x] Step 1.4: Kaggle Dataset `papertrail-processed` is published and loads in the baseline notebook.
- [ ] Confirm that most hour-0 rows are day-1 `Reinvestment` self-transfers.
- [ ] Check whether Saudi Riyal's lift survives within ACH only (cross-tab `pay_format × cur_paid`).
- [ ] Look at why 1,968 positives have no typology (for example, compare their amounts, accounts and times to the mapped ones).
- [ ] Log a corrected `rules_v2` (working send-count threshold) as a separate MLflow run.
- [x] Phase 4: as-of account features and graph features, then `xgb_v2_graph` (section 5b).
- [x] Phase 4b: ablation (section 5c).
- [x] Distribution check of `age_s` and test of removing it (section 5d). Per-feature ablation of the other timing features not done.
- [ ] Characterise the 371 unmapped test positives (the model recalls about none).
- [x] Stabilise XGBoost training (section 5d). Multi-seed means are now the reporting convention.
- [ ] 03d `recent` arms: calendar-stable features vs v2 history.
- [ ] Phase 5: GNN (`notebooks/04_gnn.py` / `04-gnn.ipynb`), debug run then full GPU run.

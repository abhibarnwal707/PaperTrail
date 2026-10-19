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
- 51 features (16 tabular + 35 as-of history/graph). Same hyperparameters as v1; `scale_pos_weight` 1283.2, training stopped at iteration 118 (the exact `best_iteration` is in `phase4_metrics.json`).
- PR-AUC rose from 0.046 to 0.351 (7.6×), F1 from 0.116 to 0.392, recall@1%FPR from 0.41 to 0.63, precision@1000 from 0.12 to 0.36. The last logged validation `aucpr` is 0.363 (exact validation PR-AUC is in `phase4_metrics.json`), close to test 0.351, so the gain is not a test-set fluke. Test has 956 positives, so small differences are still noise, but this gap is not.
- Top gain: `fmt_ACH` 0.40, `is_ach` 0.30 (the same information twice), `s_uniq_out` 0.10 (distinct counterparties the sender has paid so far), then `is_cross_currency`, `s_out_cnt_24h`, `s_gap_s` and `fmt_Cheque` at 1–2% each. Gain share is not the same as contribution: ACH still has most of the gain, but v1 had that too and scored 0.046. The history features let the model separate risky ACH transactions from ordinary ones.
- The ACH-only slice also improves (PR-AUC 0.052 to 0.401), so the gain is not just from picking out ACH.

### Why I believe it, and what is still unchecked
- Every feature is as-of (unit tests confirm that appending later rows never changes earlier rows), and train/val/test features are built the same way.
- Not yet done: (1) an ablation (drop `s_uniq_out`, then drop all history features except it) to see which feature family carries the gain; (2) recall by typology, since fan-out accounts are likely what `s_uniq_out` finds; (3) a check that the effect is not an artefact of the synthetic generator, which is a caveat for the README anyway. PageRank was computed but I have not checked its importance or whether it helps (it is null on day 1; check `xgb_v2_importance.csv`).
- F1 of 0.39 is much better but still far from solved: at the validation-chosen threshold, precision and recall are both moderate.

## 6. Open items

- [x] Save the full daily table (`reports/daily.csv`).
- [x] Step 1.3: parse the patterns file and map typologies (`reports/typology_counts.csv`).
- [x] Step 1.4: Kaggle Dataset `papertrail-processed` is published and loads in the baseline notebook.
- [ ] Confirm that most hour-0 rows are day-1 `Reinvestment` self-transfers.
- [ ] Check whether Saudi Riyal's lift survives within ACH only (cross-tab `pay_format × cur_paid`).
- [ ] Look at why 1,968 positives have no typology (for example, compare their amounts, accounts and times to the mapped ones).
- [ ] Log a corrected `rules_v2` (working send-count threshold) as a separate MLflow run.
- [ ] Phase 4: as-of account features and graph features, then `xgb_v2_graph`.

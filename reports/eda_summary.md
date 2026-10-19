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
| Rule baseline | Use ACH, a 3K–30K amount band, and high 24h send count. Exclude self-transfers, Wire and Reinvestment. Drop the cross-currency rule. |
| Features | Add `is_ach`, an amount-band flag, `is_self_transfer` and `is_cross_currency`. Drop weekday and date. |
| Imbalance | `scale_pos_weight` / `pos_weight` ≈ neg/pos on the train window (about 1,280 on the train split). |
| Graph features | Log-transform degrees. Hub = top 1% (≥ 119). Cap cycle counts because of super hubs. |
| Evaluation | Never report accuracy. Also report metrics on ACH-only transactions, since 87% of positives are ACH. |
| Agent evaluation | Typology ground truth exists for 62% of positives only. Pick showcase accounts from mapped attempts, and report unmapped alerts separately. |

## 4. Caveats for the README

- The data is synthetic (IBM AMLworld generator). Findings such as "ACH is risky" or "cross-currency is safe" describe the generator, not real banks.
- The Sep 11–18 laundering tail is a data-generation artifact. Say how it was handled and why.

## 5. Open items

- [x] Save the full daily table (`reports/daily.csv`).
- [x] Step 1.3: parse the patterns file and map typologies (`reports/typology_counts.csv`).
- [ ] Step 1.4: confirm the Kaggle Dataset `papertrail-processed` is published (`transactions.parquet` 161 MB, `patterns.parquet` 70 KB).
- [ ] Confirm that most hour-0 rows are day-1 `Reinvestment` self-transfers.
- [ ] Check whether Saudi Riyal's lift survives within ACH only (cross-tab `pay_format × cur_paid`).
- [ ] Look at why 1,968 positives have no typology (for example, compare their amounts, accounts and times to the mapped ones).

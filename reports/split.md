# Temporal split

Implemented in `src/papertrail/split.py` (`temporal_split`, `split_summary`). Boundaries are on day edges, so no day is shared between two splits. Counts come from `reports/daily.csv`, and `split_summary` reproduces them exactly.

| Split | Dates | Rows | Positives | Positive rate |
|---|---|---|---|---|
| Train | Sep 1–6 | 3,248,921 (64.0%) | 2,530 | 0.078% |
| Validation | Sep 7–8 | 965,524 (19.0%) | 1,036 | 0.107% |
| Test | Sep 9–10 | 862,792 (17.0%) | 956 | 0.111% |
| Dropped tail | Sep 11–18 | 1,108 | 655 | 59.1% |

## Checks (Step 2.1 exit criteria)
- Every split has laundering positives (`assert_valid_split`).
- Date ranges don't overlap: the train, validation and test boundaries are 2022-09-07 and 2022-09-09, and the tail starts at 2022-09-11.
- The dropped tail is logged above: 1,108 rows and 655 positives (12.7% of all positives).

## Why the tail is dropped
Normal traffic stops after Sep 10, but laundering attempts that started earlier keep finishing until Sep 18. The tail's laundering rate is 59.1% against about 0.1% on normal days, so including it in test would distort every metric. 645 of the 655 tail positives belong to attempts that started before Sep 11. The tail rows stay in the full dataset (pattern parsing, graph history) but are never scored.

## Limitations
- **Attempts straddle splits.** Laundering attempts last a median of 75 hours and up to about 202 hours (8.4 days). An attempt that begins in train can finish in validation, or an attempt that begins in validation can finish in test. The split is by transaction time, not by attempt, so some attempts are partly seen in training. This is the usual time-based split and matches how a deployed model would meet the data, but it can make results slightly optimistic.
- **Positive rate drifts upward.** Train is 0.078%, validation 0.107% and test 0.111%. PR-AUC depends on the base rate, so compare models only within the same split.
- **Only 956 test positives.** Differences of a few percent between models are within noise. Report them with that caveat.

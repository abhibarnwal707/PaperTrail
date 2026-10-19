"""Temporal train/validation/test split.

Normal traffic stops after Sep 10 while laundering continues, so Sep 11-18 is an
artifact tail (see reports/eda_summary.md). It is dropped from every split.
Splits are aligned to day boundaries so no day is shared between two splits.
"""
from datetime import datetime

import polars as pl

TRAIN_END = datetime(2022, 9, 7)    # train: ts <  Sep 7   (Sep 1-6)
VAL_END = datetime(2022, 9, 9)      # val:   Sep 7 <= ts < Sep 9   (Sep 7-8)
TAIL_START = datetime(2022, 9, 11)  # test:  Sep 9 <= ts < Sep 11  (Sep 9-10); tail: ts >= Sep 11


def _check_bounds(train_end: datetime, val_end: datetime, tail_start: datetime) -> None:
    if not train_end < val_end < tail_start:
        raise ValueError(
            f"need train_end < val_end < tail_start, got {train_end}, {val_end}, {tail_start}"
        )


def temporal_split(
    df: pl.DataFrame,
    train_end: datetime = TRAIN_END,
    val_end: datetime = VAL_END,
    tail_start: datetime = TAIL_START,
    ts_col: str = "ts",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (train, val, test). Rows at or after `tail_start` are dropped."""
    _check_bounds(train_end, val_end, tail_start)
    ts = pl.col(ts_col)
    train = df.filter(ts < train_end)
    val = df.filter((ts >= train_end) & (ts < val_end))
    test = df.filter((ts >= val_end) & (ts < tail_start))
    return train, val, test


def split_summary(
    df: pl.DataFrame,
    train_end: datetime = TRAIN_END,
    val_end: datetime = VAL_END,
    tail_start: datetime = TAIL_START,
    ts_col: str = "ts",
    label_col: str = "label",
) -> pl.DataFrame:
    """One row per split, plus the dropped tail: size, positives, positive rate, date range.

    Used to fill reports/split.md and to check that every split has positives.
    """
    _check_bounds(train_end, val_end, tail_start)
    train, val, test = temporal_split(df, train_end, val_end, tail_start, ts_col)
    tail = df.filter(pl.col(ts_col) >= tail_start)
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test), ("dropped_tail", tail)]:
        n = part.height
        pos = int(part[label_col].sum()) if n else 0
        rows.append({
            "split": name,
            "rows": n,
            "positives": pos,
            "pos_rate": pos / n if n else 0.0,
            "start": part[ts_col].min() if n else None,
            "end": part[ts_col].max() if n else None,
        })
    return pl.DataFrame(rows)


def assert_valid_split(summary: pl.DataFrame) -> None:
    """Exit check for Step 2.1: each real split has positives and the date ranges don't overlap."""
    real = summary.filter(pl.col("split") != "dropped_tail").sort("start")
    empty = real.filter(pl.col("positives") == 0)["split"].to_list()
    if empty:
        raise AssertionError(f"splits with no laundering positives: {empty}")
    starts, ends = real["start"].to_list(), real["end"].to_list()
    for i in range(1, len(starts)):
        if starts[i] <= ends[i - 1]:
            raise AssertionError("split date ranges overlap")

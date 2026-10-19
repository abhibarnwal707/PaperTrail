from datetime import datetime

import polars as pl
import pytest

from papertrail.split import assert_valid_split, split_summary, temporal_split


def _toy() -> pl.DataFrame:
    # (day, label): two rows per day, Sep 1..12; one positive on every day except Sep 8
    rows = []
    for d in range(1, 13):
        rows.append((datetime(2022, 9, d, 1, 0), 1 if d != 8 else 0))
        rows.append((datetime(2022, 9, d, 13, 0), 0))
    return pl.DataFrame(rows, schema=["ts", "label"], orient="row").sort("ts")


def test_split_boundaries_and_tail_dropped():
    tr, va, te = temporal_split(_toy())
    assert tr["ts"].max() < datetime(2022, 9, 7) <= va["ts"].min()
    assert va["ts"].max() < datetime(2022, 9, 9) <= te["ts"].min()
    assert te["ts"].max() < datetime(2022, 9, 11)   # Sep 11 and 12 dropped
    assert (tr.height, va.height, te.height) == (12, 4, 4)


def test_splits_are_disjoint_and_exclude_tail():
    df = _toy()
    tr, va, te = temporal_split(df)
    kept = tr.height + va.height + te.height
    assert kept == df.filter(pl.col("ts") < datetime(2022, 9, 11)).height


def test_summary_counts_and_validity():
    s = split_summary(_toy()).to_dicts()
    by = {r["split"]: r for r in s}
    assert by["train"]["positives"] == 6 and by["val"]["positives"] == 1 and by["test"]["positives"] == 2
    assert by["dropped_tail"]["rows"] == 4 and by["dropped_tail"]["positives"] == 2
    assert by["train"]["pos_rate"] == pytest.approx(0.5)
    assert_valid_split(split_summary(_toy()))


def test_assert_valid_split_fails_without_positives():
    df = _toy().with_columns(label=pl.lit(0))
    with pytest.raises(AssertionError):
        assert_valid_split(split_summary(df))


def test_bad_boundaries_raise():
    with pytest.raises(ValueError):
        temporal_split(_toy(), train_end=datetime(2022, 9, 9), val_end=datetime(2022, 9, 7))

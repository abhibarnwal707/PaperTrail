from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from papertrail.gnn.data import (
    TEST,
    TRAIN,
    VAL,
    EdgeTable,
    Snapshot,
    build_edge_table,
    make_batch,
)

T0 = datetime(2022, 9, 1)


def _table(edges, days, n_nodes) -> EdgeTable:
    """Toy table: edge i has attr[i, 0] = i, so a sampled edge can be traced back to the transaction it came from."""
    e = len(edges)
    return EdgeTable(
        src=np.array([a for a, _ in edges], dtype=np.int32), dst=np.array([b for _, b in edges], dtype=np.int32),
        attr=np.arange(e, dtype=np.float32)[:, None], label=np.zeros(e, dtype=np.int8),
        day=np.array(days, dtype=np.int16), split=np.zeros(e, dtype=np.int8), n_nodes=n_nodes, feature_names=["id"],
    )


# edges (src -> dst), day
EDGES = [(0, 1), (1, 2), (2, 0), (3, 1), (0, 3), (4, 0)]
DAYS = [1, 1, 2, 2, 3, 3]


def test_snapshot_holds_only_earlier_days():
    t = _table(EDGES, DAYS, 5)
    assert Snapshot(t, 1).n == 0                      # nothing before day 1
    assert Snapshot(t, 2).n == 2                      # day 1 only
    assert Snapshot(t, 3).n == 4                      # days 1 and 2
    assert Snapshot(t, 4).n == 6
    snap = Snapshot(t, 3)
    forward = snap.eid[snap.eid < snap.n]
    reverse = snap.eid[snap.eid >= snap.n] - snap.n
    assert set(forward) == set(reverse) == {0, 1, 2, 3}
    assert t.day[forward].max() < 3 and t.day[reverse].max() < 3


def test_snapshot_csr_matches_edges_and_reverses():
    t = _table(EDGES, DAYS, 5)
    snap = Snapshot(t, 3)                             # edges 0:(0,1) 1:(1,2) 2:(2,0) 3:(3,1)
    for node in range(5):
        lo, hi = snap.indptr[node], snap.indptr[node + 1]
        assert (snap.receiver[lo:hi] == node).all()
    # node 1 receives edge 0 (from 0) and edge 3 (from 3) forward, and the reverse of edge 1 (1->2) from node 2
    lo, hi = snap.indptr[1], snap.indptr[2]
    got = sorted(zip(snap.sender[lo:hi].tolist(), snap.eid[lo:hi].tolist()))
    assert got == sorted([(0, 0), (3, 3), (2, 1 + snap.n)])
    # degree features are past-only: node 0 sent 1 and received 1 before day 3
    assert snap.x[0].tolist() == pytest.approx(np.log1p([1, 1]).tolist())


def test_batch_never_contains_future_edges():
    t = _table(EDGES, DAYS, 5)
    snap = Snapshot(t, 3)
    rng = np.random.default_rng(0)
    b = make_batch(t, snap, np.array([4, 5]), (10, 10), rng)         # day-3 targets, context = days 1-2
    used = b["edge_attr"][:, 0].astype(int)                            # which transaction each sampled edge came from
    assert set(used) <= {0, 1, 2, 3}
    assert not ({4, 5} & set(used))                                    # the targets are not their own neighbours
    assert t.day[used].max() < 3


def test_batch_relabelling_and_reverse_flag():
    t = _table(EDGES, DAYS, 5)
    snap = Snapshot(t, 4)
    b = make_batch(t, snap, np.array([5]), (10, 10), np.random.default_rng(1))
    nodes, (s, r) = b["nodes"], b["edge_index"]
    assert nodes.tolist() == sorted(set(nodes.tolist()))
    assert s.max() < len(nodes) and r.max() < len(nodes) and len(b["x"]) == len(nodes)
    # local ids map back to the true endpoints; the last attr column flags reverse edges
    for j in range(s.shape[0]):
        tx = int(b["edge_attr"][j, 0])
        rev = bool(b["edge_attr"][j, -1])
        a, c = EDGES[tx]
        assert (nodes[s[j]], nodes[r[j]]) == ((c, a) if rev else (a, c))
    assert nodes[b["tgt_src"][0]] == EDGES[5][0] and nodes[b["tgt_dst"][0]] == EDGES[5][1]
    assert b["tgt_attr"][0, 0] == 5


def test_fanout_caps_the_neighbourhood_and_is_reproducible():
    n = 200
    edges = [(i, 0) for i in range(1, n)] + [(0, 1)]                  # node 0 receives 199 edges
    t = _table(edges, [1] * (n - 1) + [2], n)
    snap = Snapshot(t, 2)
    b1 = make_batch(t, snap, np.array([n - 1]), (5,), np.random.default_rng(3))
    b2 = make_batch(t, snap, np.array([n - 1]), (5,), np.random.default_rng(3))
    into_zero = (b1["nodes"][b1["edge_index"][1]] == 0).sum()
    assert into_zero <= 5
    assert np.array_equal(b1["edge_global"], b2["edge_global"])


def test_empty_snapshot_gives_an_empty_but_valid_batch():
    t = _table(EDGES, DAYS, 5)
    b = make_batch(t, Snapshot(t, 1), np.array([0, 1]), (10, 5), np.random.default_rng(0))
    assert b["edge_index"].shape == (2, 0) and b["edge_attr"].shape[0] == 0
    assert len(b["x"]) == len(b["nodes"]) and b["tgt_attr"].shape[0] == 2


def test_two_hops_reach_further_than_one():
    chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    t = _table(chain + [(5, 6)], [1] * 5 + [2], 7)
    snap = Snapshot(t, 2)
    one = make_batch(t, snap, np.array([5]), (10,), np.random.default_rng(0))
    two = make_batch(t, snap, np.array([5]), (10, 10), np.random.default_rng(0))
    assert len(two["nodes"]) > len(one["nodes"])


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [(i, T0 + timedelta(hours=h), s, d, a, 1, 2, c, c, f, y) for i, (h, s, d, a, c, f, y) in enumerate(rows)],
        schema=["txn_id", "ts", "src", "dst", "amt_paid", "from_bank", "to_bank", "cur_paid", "cur_recv",
                "pay_format", "label"], orient="row")


def test_build_edge_table_from_frames():
    rows = [
        (1, "a", "b", 100.0, "Euro", "ACH", 0), (5, "b", "c", 3000.0, "Euro", "Wire", 0),
        (30, "a", "c", 50.0, "Yen", "ACH", 1), (31, "c", "a", 7.0, "Euro", "ACH", 0),
        (55, "b", "a", 9.0, "Euro", "Wire", 0), (80, "a", "b", 11.0, "Euro", "ACH", 1),
    ]
    df = _frame(rows)
    train, val, test = df.head(2), df.slice(2, 2), df.tail(2)
    t = build_edge_table(train, val, test)
    assert t.n_edges == 6 and t.n_nodes == 3
    assert t.split.tolist() == [TRAIN, TRAIN, VAL, VAL, TEST, TEST]
    assert t.day.tolist() == [1, 1, 2, 2, 3, 4]
    assert t.label.tolist() == [0, 0, 1, 0, 0, 1]
    assert t.attr.shape == (6, len(t.feature_names)) and np.isfinite(t.attr).all()
    assert "log_amt_z" in t.feature_names and "hour_sin" in t.feature_names
    assert t.ids(VAL).tolist() == [2, 3] and t.ids(TEST, min_day=4).tolist() == [5]
    # "Yen" is not in train: the unseen currency maps to all-zero one-hots, not an error
    cur_cols = [i for i, n in enumerate(t.feature_names) if n.startswith("cur_")]
    assert t.attr[2, cur_cols].sum() == 0
    assert t.attr[:, t.feature_names.index("log_amt_z")][:2].mean() == pytest.approx(0.0, abs=1e-5)  # standardised on train


def test_build_edge_table_rejects_unsorted_splits():
    df = _frame([(1, "a", "b", 1.0, "Euro", "ACH", 0), (30, "a", "b", 1.0, "Euro", "ACH", 0)])
    with pytest.raises(ValueError):
        build_edge_table(df.tail(1), df.head(1), df.head(0))

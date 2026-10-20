"""Graph data for the edge classifier (Phase 5). Pure numpy/polars: no torch, no torch-geometric.

Accounts are nodes, transactions are edges. Every transaction on day d is scored from a SNAPSHOT of the graph that holds
only transactions of days before d (the same idea as the daily PageRank in `graph_features`), so no neighbour is ever
from the future. Messages flow both ways: each transaction also exists as a reverse edge flagged `is_reverse = 1`.
The target edge itself is not part of its own snapshot; its features go straight into the classifier head.

The neighbour sampler is numpy too, so it runs (and is unit-tested) without pyg-lib or torch-sparse.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from papertrail.features import build_features, fit_categories

TRAIN, VAL, TEST = 0, 1, 2


@dataclass
class EdgeTable:
    """All usable transactions in time order (train + val + test; the dropped tail is excluded)."""
    src: np.ndarray        # int32 node id of the sender
    dst: np.ndarray        # int32 node id of the receiver
    attr: np.ndarray       # float32 [E, D] edge features (no history, no calendar position)
    label: np.ndarray      # int8
    day: np.ndarray        # int16, dense calendar-day index starting at 1
    split: np.ndarray      # int8: 0 train, 1 val, 2 test
    n_nodes: int
    feature_names: list[str]

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def ids(self, split: int, min_day: int = 1) -> np.ndarray:
        """Global indices of the edges of one split on day >= min_day."""
        return np.flatnonzero((self.split == split) & (self.day >= min_day))


def edge_feature_matrix(df: pl.DataFrame, cats: dict, mu: float, sd: float) -> tuple[np.ndarray, list[str]]:
    """Per-transaction edge features. `mu`/`sd` standardise log amount (fit on train). No calendar position."""
    base = build_features(df, cats)
    hour = base["hour"].to_numpy().astype(np.float32)
    cols = {
        "log_amt_z": ((base["log_amt"].to_numpy() - mu) / sd).astype(np.float32),
        "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
        "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
    }
    for c in ("is_cross_currency", "is_same_bank", "is_self_transfer", "is_round_amount", "in_amount_band"):
        cols[c] = base[c].to_numpy().astype(np.float32)
    for c in base.columns:
        if c.startswith("fmt_"):
            cols[c] = base[c].to_numpy().astype(np.float32)
    cur = df["cur_paid"].to_numpy()
    for name in cats["cur_paid"]:
        cols["cur_" + name.replace(" ", "_")] = (cur == name).astype(np.float32)
    names = list(cols)
    return np.stack([cols[n] for n in names], axis=1), names


def build_edge_table(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> EdgeTable:
    """Concatenate the three splits (each sorted by ts, in order) and build ids, features and day indices.

    Category lists and the amount scaler are fitted on `train` only.
    """
    frames = [train, val, test]
    df = pl.concat(frames)
    if not df["ts"].is_sorted():
        raise ValueError("train, val and test must be time-ordered and non-overlapping (use temporal_split)")
    cats = fit_categories(train)
    log_amt = train["amt_paid"].log1p()
    mu, sd = float(log_amt.mean()), float(log_amt.std()) or 1.0

    accounts = pl.concat([df["src"], df["dst"]]).unique().sort().to_frame("acct").with_row_index("id")
    ids = accounts.with_columns(pl.col("id").cast(pl.Int32))
    src = df.select("src").join(ids.rename({"acct": "src", "id": "i"}), on="src", how="left")["i"].to_numpy()
    dst = df.select("dst").join(ids.rename({"acct": "dst", "id": "i"}), on="dst", how="left")["i"].to_numpy()

    attr, names = edge_feature_matrix(df, cats, mu, sd)
    day = df.select(pl.col("ts").dt.truncate("1d").rank("dense").cast(pl.Int16).alias("d"))["d"].to_numpy()
    split = np.concatenate([np.full(f.height, code, dtype=np.int8) for code, f in enumerate(frames)])
    return EdgeTable(src=src.astype(np.int32), dst=dst.astype(np.int32), attr=attr,
                     label=df["label"].to_numpy().astype(np.int8), day=day, split=split,
                     n_nodes=ids.height, feature_names=names)


class Snapshot:
    """The graph of all transactions strictly before `day`, as a receiver-indexed CSR of directed edges.

    Directed edge ids: 0..n-1 are the transactions (sender -> receiver), n..2n-1 are their reverses.
    """

    def __init__(self, table: EdgeTable, day: int):
        self.table = table
        self.day = day
        n = int(np.searchsorted(table.day, day, side="left"))      # table is time-ordered, so "before day" is a prefix
        self.n = n
        sender = np.concatenate([table.src[:n], table.dst[:n]])
        receiver = np.concatenate([table.dst[:n], table.src[:n]])
        order = np.argsort(receiver, kind="stable")
        self.sender = sender[order]
        self.receiver = receiver[order]
        self.eid = order.astype(np.int64)
        counts = np.bincount(receiver, minlength=table.n_nodes)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out_deg = np.bincount(table.src[:n], minlength=table.n_nodes)
        in_deg = np.bincount(table.dst[:n], minlength=table.n_nodes)
        self.x = np.log1p(np.stack([out_deg, in_deg], axis=1)).astype(np.float32)   # node features: past-only degrees

    def sample_hop(self, nodes: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """Positions (into the CSR arrays) of up to `k` incoming edges per node: all of them if the node has <= k."""
        starts = self.indptr[nodes]
        deg = self.indptr[nodes + 1] - starts
        take_all = deg <= k
        cnt = np.where(take_all, deg, k)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        owner = np.repeat(np.arange(len(nodes)), cnt)
        offset = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sampled = (rng.random(total) * deg[owner]).astype(np.int64)         # uniform, with replacement
        pos = np.where(take_all[owner], offset, sampled)
        return starts[owner] + pos


def make_batch(table: EdgeTable, snap: Snapshot, target_ids: np.ndarray, fanouts: tuple[int, ...],
               rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sampled subgraph around the endpoints of the target edges, all from `snap` (so all strictly in the past)."""
    t_src, t_dst = table.src[target_ids], table.dst[target_ids]
    seeds = np.unique(np.concatenate([t_src, t_dst]))
    frontier, picked = seeds, []
    for k in fanouts:
        pos = snap.sample_hop(frontier, k, rng)
        picked.append(pos)
        frontier = np.unique(snap.sender[pos]) if len(pos) else frontier[:0]
    pos = np.unique(np.concatenate(picked)) if picked else np.empty(0, dtype=np.int64)

    senders, receivers, eid = snap.sender[pos], snap.receiver[pos], snap.eid[pos]
    nodes = np.unique(np.concatenate([seeds, senders, receivers]))
    n = max(snap.n, 1)
    attr = np.concatenate([table.attr[eid % n], (eid >= snap.n).astype(np.float32)[:, None]], axis=1)
    return {
        "x": snap.x[nodes],
        "edge_index": np.stack([np.searchsorted(nodes, senders), np.searchsorted(nodes, receivers)]).astype(np.int64),
        "edge_attr": attr.astype(np.float32),
        "tgt_src": np.searchsorted(nodes, t_src).astype(np.int64),
        "tgt_dst": np.searchsorted(nodes, t_dst).astype(np.int64),
        "tgt_attr": table.attr[target_ids],
        "y": table.label[target_ids].astype(np.float32),
        "nodes": nodes,                  # global node ids of the local nodes (for tests and debugging)
        "edge_global": eid,              # directed edge ids of the sampled edges (for tests and debugging)
    }

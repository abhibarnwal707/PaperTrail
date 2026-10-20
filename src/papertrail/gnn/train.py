"""Training and scoring for the GINE edge classifier (daily-snapshot protocol, see `gnn.data`)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from torch import nn

from papertrail.gnn.data import TRAIN, EdgeTable, Snapshot, make_batch
from papertrail.gnn.model import GINEEdgeClassifier
from papertrail.metrics import pr_auc


@dataclass
class GNNConfig:
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    fanouts: tuple = (10, 5)
    batch_size: int = 1024         # random target edges per training step
    pos_frac: float = 0.05         # extra laundering edges added to every step (resampled), as a fraction of batch_size
    lr: float = 1e-3
    weight_decay: float = 1e-5
    steps_per_epoch: int = 500
    max_epochs: int = 30
    patience: int = 5              # epochs without a validation PR-AUC gain before stopping
    eval_batch: int = 4096
    min_train_day: int = 2         # day-1 targets have an empty snapshot, so they are not trained on
    seed: int = 0

    def as_dict(self):
        return {k: str(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _to_torch(b, device):
    inputs = {k: torch.from_numpy(b[k]).to(device)
              for k in ("x", "edge_index", "edge_attr", "tgt_src", "tgt_dst", "tgt_attr")}
    return inputs, torch.from_numpy(b["y"]).to(device)


def build_model(table: EdgeTable, cfg: GNNConfig) -> GINEEdgeClassifier:
    d = table.attr.shape[1]
    return GINEEdgeClassifier(node_dim=2, edge_dim=d + 1, target_dim=d, hidden=cfg.hidden, layers=cfg.layers,
                              dropout=cfg.dropout)


class SnapshotCache:
    """Snapshots by day (each is a few hundred MB at most); built on first use."""

    def __init__(self, table: EdgeTable):
        self.table, self._snaps = table, {}

    def __getitem__(self, day: int) -> Snapshot:
        if day not in self._snaps:
            self._snaps[day] = Snapshot(self.table, day)
        return self._snaps[day]


@torch.no_grad()
def predict(model, table: EdgeTable, ids: np.ndarray, cfg: GNNConfig, device, snaps: SnapshotCache,
            embeddings: bool = False, seed: int = 0):
    """Probability (or head embedding) for the given edge ids. Each edge is scored from its own day's snapshot."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.zeros((len(ids), cfg.hidden), dtype=np.float32) if embeddings else np.zeros(len(ids), dtype=np.float32)
    days = table.day[ids]
    for d in np.unique(days):
        where = np.flatnonzero(days == d)
        snap = snaps[int(d)]
        for lo in range(0, len(where), cfg.eval_batch):
            sel = where[lo:lo + cfg.eval_batch]
            inputs, _ = _to_torch(make_batch(table, snap, ids[sel], cfg.fanouts, rng), device)
            res = model.embed(**inputs) if embeddings else torch.sigmoid(model(**inputs))
            out[sel] = res.cpu().numpy()
    return out


def train_gnn(table: EdgeTable, cfg: GNNConfig, device, val_ids: np.ndarray, log=print, snaps: SnapshotCache | None = None):
    """Train on the train split; early-stop on validation PR-AUC. Returns (best model, history, snapshot cache)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    snaps = snaps or SnapshotCache(table)
    model = build_model(table, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    ids = table.ids(TRAIN, min_day=cfg.min_train_day)
    by_day = {int(d): ids[table.day[ids] == d] for d in np.unique(table.day[ids])}
    pos_by_day = {d: v[table.label[v] == 1] for d, v in by_day.items()}
    days = np.array(sorted(by_day))
    weights = np.array([len(by_day[d]) for d in days], dtype=float)
    weights /= weights.sum()
    n_pos_extra = round(cfg.pos_frac * cfg.batch_size)
    log(f"train targets: {len(ids)} on days {days.tolist()}, positives: {int(table.label[ids].sum())}")

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(cfg.steps_per_epoch):
            d = int(rng.choice(days, p=weights))
            batch_ids = rng.choice(by_day[d], size=min(cfg.batch_size, len(by_day[d])), replace=False)
            if n_pos_extra and len(pos_by_day[d]):
                batch_ids = np.concatenate([batch_ids, rng.choice(pos_by_day[d], size=n_pos_extra, replace=True)])
            inputs, y = _to_torch(make_batch(table, snaps[d], batch_ids, cfg.fanouts, rng), device)
            opt.zero_grad()
            loss = loss_fn(model(**inputs), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        val_p = predict(model, table, val_ids, cfg, device, snaps)
        val_pr = pr_auc(table.label[val_ids], val_p)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": float(val_pr)})
        log(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val PR-AUC {val_pr:.4f}  ({time.time() - t0:.0f}s)")
        if val_pr > best:
            best, bad = val_pr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop: no gain for {cfg.patience} epochs; best val PR-AUC {best:.4f}")
                break
    model.load_state_dict(best_state)
    return model, history, snaps


def overfit_check(table: EdgeTable, cfg: GNNConfig, device, n: int = 256, steps: int = 300, log=print) -> float:
    """Debug: train on one fixed batch of targets; the loss should fall close to 0. Proves the pipeline can learn."""
    cfg = replace(cfg, dropout=0.0)          # dropout noise would keep the loss from reaching 0
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ids = table.ids(TRAIN, min_day=cfg.min_train_day)
    pos = ids[table.label[ids] == 1]
    pick = np.concatenate([rng.choice(pos, size=min(n // 4, len(pos)), replace=False),
                           rng.choice(ids, size=n - min(n // 4, len(pos)), replace=False)])
    snaps = SnapshotCache(table)
    day = int(table.day[pick].max())
    fixed = make_batch(table, snaps[day], pick, cfg.fanouts, rng)
    inputs, y = _to_torch(fixed, device)
    model = build_model(table, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for s in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(**inputs), y)
        loss.backward()
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            log(f"step {s:3d}  loss {float(loss.detach()):.4f}")
    return float(loss.detach())

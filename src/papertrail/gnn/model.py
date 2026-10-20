"""GINE-style edge classifier in plain PyTorch (no torch-geometric needed).

GINE layer (Hu et al., "Strategies for pre-training GNNs"):  h_i' = MLP((1 + eps) * h_i + sum_j ReLU(h_j + W e_ij)).
Edge classifier: logit = MLP([h_src, h_dst, W_t e_target]) where e_target is the transaction being scored (it is not part
of its own snapshot, so its features enter here). `embed` returns the hidden layer before the logit, for the hybrid model.
"""
import torch
from torch import nn


class GINELayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h, edge_index, edge_attr):
        send, recv = edge_index
        msg = torch.relu(h[send] + self.edge_lin(edge_attr))
        agg = torch.zeros_like(h).index_add_(0, recv, msg)
        return self.mlp((1 + self.eps) * h + agg)


class GINEEdgeClassifier(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, target_dim: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2):
        """`edge_dim` = message edge features (target features + the is_reverse flag), `target_dim` = target features."""
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GINELayer(hidden, edge_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.target_in = nn.Linear(target_dim, hidden)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)

    def embed(self, x, edge_index, edge_attr, tgt_src, tgt_dst, tgt_attr):
        h = torch.relu(self.node_in(x))
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(torch.relu(norm(conv(h, edge_index, edge_attr)))) + h      # residual keeps the node's own signal
        z = torch.cat([h[tgt_src], h[tgt_dst], torch.relu(self.target_in(tgt_attr))], dim=1)
        return self.head(z)

    def forward(self, x, edge_index, edge_attr, tgt_src, tgt_dst, tgt_attr):
        return self.out(self.embed(x, edge_index, edge_attr, tgt_src, tgt_dst, tgt_attr)).squeeze(-1)

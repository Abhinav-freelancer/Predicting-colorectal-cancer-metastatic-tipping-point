"""
Phase 4 - Step 2: Graph Neural Network (GNN)
=============================================
Processes the gene co-expression graph per patient and extracts
a graph-level embedding capturing the topology of EMT gene interactions.

Architecture:
    Input: gene expression graph (N nodes × 6 features)
    3 × GraphSAGE convolution layers (mean aggregation)
        → captures multi-hop neighbourhood gene interactions
    Global mean + max pooling
        → graph-level embedding (invariant to node ordering)
    Output: (batch_size, gnn_out_dim) embedding vector

Why GraphSAGE over GCN:
    - Inductive: works on unseen graphs without retraining
    - Mean aggregation is more robust to varying node degrees
    - Better suited to small graphs (our EMT graphs are 5–23 nodes)

Usage:
    python src/models/gnn.py   # self-test
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


# ── GraphSAGE convolution (pure PyTorch, no PyG required) ────────────────────

class SAGEConv(nn.Module):
    """
    GraphSAGE convolution layer implemented in pure PyTorch.
    Avoids requiring torch-geometric (which can be tricky to install).

    h_v^(l+1) = σ( W · CONCAT(h_v^(l), MEAN_{u∈N(v)} h_u^(l)) )

    Uses batched adjacency representation (padded node features + edge_index).
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        # W operates on [self_feat | neighbour_mean] → out_dim
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self,
                x:          torch.Tensor,   # (B, N, F)
                edge_index: torch.Tensor,   # (2, total_edges)  global indices
                n_nodes:    torch.Tensor,   # (B,) nodes per graph
                batch_size: int) -> torch.Tensor:
        """
        x          : padded node features (batch_size, max_nodes, in_dim)
        edge_index : COO edge list with global node indices
        n_nodes    : number of real (non-padded) nodes per graph
        Returns    : (batch_size, max_nodes, out_dim)
        """
        B, N, F = x.shape
        out     = torch.zeros(B, N, self.linear.out_features,
                              device=x.device, dtype=x.dtype)

        # Build cumulative node offsets
        offsets = torch.zeros(B + 1, dtype=torch.long, device=x.device)
        for b in range(B):
            offsets[b + 1] = offsets[b] + n_nodes[b]

        # For each graph in batch, aggregate neighbour features
        for b in range(B):
            start  = int(offsets[b])
            end    = int(offsets[b + 1])
            n_real = end - start

            # Extract edges belonging to this graph
            mask     = (edge_index[0] >= start) & (edge_index[0] < end)
            local_ei = edge_index[:, mask] - start   # local indices

            # Node features for this graph (real nodes only)
            x_b = x[b, :n_real]   # (n_real, F)

            # Aggregate: mean of neighbours
            if local_ei.shape[1] > 0:
                src, dst = local_ei[0], local_ei[1]
                # Sum neighbour features
                agg = torch.zeros_like(x_b)
                agg.index_add_(0, dst, x_b[src])
                # Count neighbours per node
                deg = torch.zeros(n_real, device=x.device, dtype=x.dtype)
                deg.index_add_(0, dst,
                               torch.ones(src.shape[0],
                                          device=x.device, dtype=x.dtype))
                deg = deg.clamp(min=1.0).unsqueeze(1)
                neigh_mean = agg / deg
            else:
                neigh_mean = x_b   # no edges: use self

            # Concatenate self + neighbour mean → linear → activation
            h = torch.cat([x_b, neigh_mean], dim=-1)   # (n_real, 2F)
            h = self.linear(h)                          # (n_real, out_dim)
            out[b, :n_real] = h

        return out


# ── GNN model ─────────────────────────────────────────────────────────────────

class EMTGraphNet(nn.Module):
    """
    3-layer GraphSAGE network for EMT gene interaction graphs.

    Architecture:
        SAGEConv(6 → 64)  + LayerNorm + ReLU + Dropout
        SAGEConv(64 → 128) + LayerNorm + ReLU + Dropout
        SAGEConv(128 → 128) + LayerNorm + ReLU
        Global mean pooling + Global max pooling → concat
        Linear(256 → gnn_out_dim)
    """

    def __init__(self,
                 node_in_dim:  int   = 6,
                 hidden_dim:   int   = 64,
                 out_dim:      int   = 128,
                 dropout:      float = 0.3):
        super().__init__()
        self.dropout = dropout

        # GraphSAGE layers
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim,  hidden_dim * 2)
        self.conv3 = SAGEConv(hidden_dim * 2, hidden_dim * 2)

        # Layer norms (over feature dim, not batch/node dims)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.norm3 = nn.LayerNorm(hidden_dim * 2)

        # Readout: mean + max pooling → project to out_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
        )

        self.out_dim = out_dim

    def forward(self,
                node_features: torch.Tensor,   # (B, max_N, 6)
                edge_index:    torch.Tensor,   # (2, total_edges)
                n_nodes:       torch.Tensor    # (B,) real nodes per graph
                ) -> torch.Tensor:             # (B, out_dim)

        B, max_N, _ = node_features.shape
        x           = node_features

        # Layer 1
        x = self.conv1(x, edge_index, n_nodes, B)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, n_nodes, B)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index, n_nodes, B)
        x = self.norm3(x)
        x = F.relu(x)

        # Global pooling: mask padded nodes, then mean + max
        mask = torch.zeros(B, max_N, 1, device=x.device)
        for b in range(B):
            mask[b, :n_nodes[b]] = 1.0

        x_masked  = x * mask
        sum_feats = x_masked.sum(dim=1)                          # (B, dim)
        n_real    = n_nodes.float().unsqueeze(1).clamp(min=1)    # (B, 1)
        mean_pool = sum_feats / n_real                           # (B, dim)

        # Max pooling (ignore padding)
        x_neg_inf = x.masked_fill(mask == 0, float("-inf"))
        max_pool  = x_neg_inf.max(dim=1).values                  # (B, dim)
        max_pool  = max_pool.nan_to_num(nan=0.0, posinf=0.0)

        # Concatenate and project
        graph_repr = torch.cat([mean_pool, max_pool], dim=-1)    # (B, 4*hidden)
        out        = self.readout(graph_repr)                     # (B, out_dim)
        return out


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.models.dataset import CRCMetastasisDataset, collate_fn
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("  Phase 4 — Step 2: GNN self-test")
    print("=" * 60)

    ds     = CRCMetastasisDataset()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=False)
    batch  = next(iter(loader))

    model = EMTGraphNet(node_in_dim=6, hidden_dim=64, out_dim=128, dropout=0.3)
    print(f"\n  Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        out = model(batch["node_features"],
                    batch["edge_index"],
                    torch.ones(8, dtype=torch.long) * batch["node_features"].shape[1])

    print(f"  Input  node_features : {tuple(batch['node_features'].shape)}")
    print(f"  Input  edge_index    : {tuple(batch['edge_index'].shape)}")
    print(f"  Output embedding     : {tuple(out.shape)}")
    print(f"  Output range         : [{out.min():.3f}, {out.max():.3f}]")
    print("\n  ✓ GNN OK")

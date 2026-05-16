"""
Phase 4 - Step 4: Hybrid model + physics-informed loss
=======================================================
Assembles the full MPS (Metastatic Proximity Score) model:

    GNN(gene graph)           →  graph_embedding  (128-dim)
    Transformer(seq)          →  seq_embedding    (128-dim)
    Tabular MLP(features)     →  tabular_embedding (64-dim)
          ↓ concatenate (320-dim)
    Fusion MLP                →  fused_embedding  (128-dim)
          ↓
    Bifurcation Classifier    →  MPS ∈ [0,1]

Physics-informed loss:
    Total loss = BCE(MPS, label)                    ← supervised signal
               + λ_phys × Physics_consistency_loss  ← bifurcation prior
               + λ_calib × Calibration_loss         ← uncertainty calibration

Physics consistency loss:
    Penalises predictions that contradict the ODE bifurcation model:
    - If physics_score > 0.7 (near mesenchymal attractor), MPS should be high
    - If physics_score < 0.3 (epithelial attractor), MPS should be low
    - In bistable zone (0.3–0.7), model is free to learn from data

Monte Carlo Dropout:
    At inference, run forward pass T times with dropout ON.
    Mean = MPS point estimate. Std = epistemic uncertainty.
    Confidence interval: MPS ± 1.96 × std (95% CI)

Usage:
    python src/models/model.py   # self-test
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.gnn         import EMTGraphNet
from src.models.transformer import TemporalTransformer


# ── Tabular MLP branch ────────────────────────────────────────────────────────

class TabularMLP(nn.Module):
    """
    Simple MLP for tabular feature encoding.
    Uses residual connections for better gradient flow.
    """
    def __init__(self,
                 in_dim:   int,
                 out_dim:  int   = 64,
                 dropout:  float = 0.3):
        super().__init__()
        hidden = in_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # Residual projection if dimensions differ
        self.residual = (nn.Linear(in_dim, out_dim, bias=False)
                         if in_dim != out_dim else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


# ── Fusion + Classifier ───────────────────────────────────────────────────────

class BifurcationClassifier(nn.Module):
    """
    Fusion head that combines GNN, Transformer, and tabular embeddings
    and outputs the Metastatic Proximity Score.

    Hidden layers use progressively decreasing width with dropout,
    following the bifurcation analogy: gradually 'committing' to
    one of two attractor states (metastatic vs non-metastatic).
    """
    def __init__(self,
                 fusion_in_dim: int,
                 hidden_dims:   list = [256, 128, 64],
                 dropout:       float = 0.3):
        super().__init__()

        layers = []
        in_d   = fusion_in_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_d, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h_dim

        self.hidden  = nn.Sequential(*layers)
        self.output  = nn.Linear(in_d, 1)   # single logit → sigmoid = MPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit (pre-sigmoid). Shape: (B, 1)"""
        return self.output(self.hidden(x))


# ── Full hybrid model ─────────────────────────────────────────────────────────

class MPSModel(nn.Module):
    """
    Metastatic Proximity Score (MPS) model.

    Combines three parallel branches then fuses to a single score.

    Forward returns:
        mps_logit  : (B, 1) pre-sigmoid output
        mps_score  : (B, 1) sigmoid(logit) ∈ [0,1]
    """

    def __init__(self,
                 n_tabular_features: int,
                 # GNN config
                 gnn_node_in_dim:    int   = 6,
                 gnn_hidden:         int   = 64,
                 gnn_out_dim:        int   = 128,
                 # Transformer config
                 transformer_d:      int   = 128,
                 transformer_heads:  int   = 4,
                 transformer_layers: int   = 3,
                 transformer_out:    int   = 128,
                 seq_len:            int   = 8,
                 # Tabular MLP config
                 tabular_out:        int   = 64,
                 # Fusion + classifier
                 hidden_dims:        list  = [256, 128, 64],
                 dropout:            float = 0.3):
        super().__init__()

        self.gnn = EMTGraphNet(
            node_in_dim = gnn_node_in_dim,
            hidden_dim  = gnn_hidden,
            out_dim     = gnn_out_dim,
            dropout     = dropout,
        )

        self.transformer = TemporalTransformer(
            n_features  = n_tabular_features,
            d_model     = transformer_d,
            n_heads     = transformer_heads,
            n_layers    = transformer_layers,
            out_dim     = transformer_out,
            max_seq_len = seq_len + 2,
            dropout     = dropout,
        )

        self.tabular_mlp = TabularMLP(
            in_dim  = n_tabular_features,
            out_dim = tabular_out,
            dropout = dropout,
        )

        fusion_dim = gnn_out_dim + transformer_out + tabular_out
        self.classifier = BifurcationClassifier(
            fusion_in_dim = fusion_dim,
            hidden_dims   = hidden_dims,
            dropout       = dropout,
        )

        self._log_model_info(fusion_dim)

    def _log_model_info(self, fusion_dim: int):
        total = sum(p.numel() for p in self.parameters())
        gnn_p = sum(p.numel() for p in self.gnn.parameters())
        tr_p  = sum(p.numel() for p in self.transformer.parameters())
        tab_p = sum(p.numel() for p in self.tabular_mlp.parameters())
        cls_p = sum(p.numel() for p in self.classifier.parameters())
        print(f"\n  MPSModel parameter breakdown:")
        print(f"    GNN              : {gnn_p:>8,}")
        print(f"    Transformer      : {tr_p:>8,}")
        print(f"    Tabular MLP      : {tab_p:>8,}")
        print(f"    Classifier head  : {cls_p:>8,}")
        print(f"    Fusion dim       : {fusion_dim}")
        print(f"    Total parameters : {total:>8,}")

    def forward(self,
                tabular:        torch.Tensor,   # (B, n_features)
                node_features:  torch.Tensor,   # (B, max_N, 6)
                edge_index:     torch.Tensor,   # (2, total_edges)
                temporal_seq:   torch.Tensor,   # (B, seq_len, n_features)
                n_nodes:        torch.Tensor,   # (B,) real nodes per graph
                ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Three parallel branches ───────────────────────────────────────
        gnn_emb   = self.gnn(node_features, edge_index, n_nodes)  # (B, 128)
        trans_emb = self.transformer(temporal_seq)                 # (B, 128)
        tab_emb   = self.tabular_mlp(tabular)                      # (B, 64)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = torch.cat([gnn_emb, trans_emb, tab_emb], dim=-1)   # (B, 320)

        # ── Bifurcation classifier ────────────────────────────────────────
        logit = self.classifier(fused)                              # (B, 1)
        score = torch.sigmoid(logit)                                # (B, 1)

        return logit, score

    def predict_with_uncertainty(self,
                                  tabular:       torch.Tensor,
                                  node_features: torch.Tensor,
                                  edge_index:    torch.Tensor,
                                  temporal_seq:  torch.Tensor,
                                  n_nodes:       torch.Tensor,
                                  n_samples:     int = 50
                                  ) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs the model T times with dropout ACTIVE (training=True on dropout
        layers) to sample from the approximate posterior.

        Returns:
            mps_mean  : (B, 1) — point estimate of MPS
            mps_std   : (B, 1) — epistemic uncertainty
            ci_lower  : (B, 1) — 95% confidence interval lower bound
            ci_upper  : (B, 1) — 95% confidence interval upper bound
        """
        # Enable dropout at inference via train mode (only for dropout layers)
        self.train()   # turns on dropout
        with torch.no_grad():
            samples = torch.stack([
                self.forward(tabular, node_features, edge_index,
                             temporal_seq, n_nodes)[1]
                for _ in range(n_samples)
            ], dim=0)   # (n_samples, B, 1)

        self.eval()

        mps_mean = samples.mean(dim=0)          # (B, 1)
        mps_std  = samples.std(dim=0)           # (B, 1)
        ci_lower = mps_mean - 1.96 * mps_std
        ci_upper = mps_mean + 1.96 * mps_std

        return mps_mean, mps_std, ci_lower.clamp(0, 1), ci_upper.clamp(0, 1)


# ── Physics-informed loss ─────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss = BCE + λ_phys × Physics consistency + λ_calib × Calibration

    Physics consistency loss:
        For patients with high physics_score (≥ 0.7): penalise low MPS
        For patients with low  physics_score (≤ 0.3): penalise high MPS
        For bistable zone (0.3–0.7): no physics penalty (let data decide)

        L_phys = mean( relu(0.7 - mps)  where physics ≥ 0.7 )
               + mean( relu(mps - 0.3)  where physics ≤ 0.3 )

    Calibration loss (soft ECE):
        Encourages predicted probabilities to match empirical frequencies.
        Uses differentiable bin-wise ECE approximation.
        L_calib = Σ_bins |mean_pred_in_bin - mean_label_in_bin|

    λ_phys   = 0.15 (physics prior weight — strong enough to guide,
                     weak enough not to override data signal)
    λ_calib  = 0.05 (calibration weight)
    """

    def __init__(self,
                 class_weights: torch.Tensor = None,
                 lambda_phys:   float = 0.15,
                 lambda_calib:  float = 0.05,
                 n_calib_bins:  int   = 5):
        super().__init__()
        self.lambda_phys  = lambda_phys
        self.lambda_calib = lambda_calib
        self.n_bins       = n_calib_bins

        # Binary cross-entropy with class weights for imbalanced data
        pos_weight = (class_weights[1] / class_weights[0]
                      if class_weights is not None
                      else torch.tensor(1.0))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def physics_consistency_loss(self,
                                  mps_score:    torch.Tensor,   # (B, 1)
                                  physics_feat: torch.Tensor    # (B, n_physics)
                                  ) -> torch.Tensor:
        """
        Penalise MPS predictions inconsistent with ODE bifurcation model.
        physics_score is assumed to be index 2 in physics_feat
        (attractor_proximity, bifurcation_score, physics_score, ...).
        """
        # Use physics_score (combined ODE signal) — index 2
        if physics_feat.shape[1] < 3:
            return torch.tensor(0.0, device=mps_score.device)

        phys = physics_feat[:, 2].unsqueeze(1)   # (B, 1) — raw (not scaled)

        # Map scaled physics back to [0,1] approximate range via sigmoid
        # (features were StandardScaler'd during dataset loading)
        phys_approx = torch.sigmoid(phys)

        mps = mps_score   # already in [0,1]

        # Penalise: high physics but low MPS
        near_mes   = (phys_approx >= 0.6).float()
        penalty_hi = F.relu(0.6 - mps) * near_mes

        # Penalise: low physics but high MPS
        near_epi   = (phys_approx <= 0.4).float()
        penalty_lo = F.relu(mps - 0.4) * near_epi

        return (penalty_hi + penalty_lo).mean()

    def calibration_loss(self,
                          mps_score: torch.Tensor,   # (B, 1)
                          labels:    torch.Tensor    # (B,)
                          ) -> torch.Tensor:
        """
        Soft differentiable calibration loss.
        Encourages P(metastatic | MPS=p) ≈ p across all confidence levels.
        """
        mps   = mps_score.squeeze(1)   # (B,)
        bins  = torch.linspace(0, 1, self.n_bins + 1, device=mps.device)
        loss  = torch.tensor(0.0, device=mps.device)
        count = 0

        for i in range(self.n_bins):
            lo, hi   = bins[i], bins[i + 1]
            # Soft bin membership (differentiable)
            in_bin   = torch.sigmoid(10 * (mps - lo)) * torch.sigmoid(10 * (hi - mps))
            bin_size = in_bin.sum()
            if bin_size > 0.5:
                pred_mean  = (mps * in_bin).sum() / bin_size
                label_mean = (labels.float() * in_bin).sum() / bin_size
                loss  += (pred_mean - label_mean).abs()
                count += 1

        return loss / max(count, 1)

    def forward(self,
                logits:      torch.Tensor,   # (B, 1) pre-sigmoid
                mps_score:   torch.Tensor,   # (B, 1) sigmoid output
                labels:      torch.Tensor,   # (B,)
                physics_feat: torch.Tensor   # (B, n_physics)
                ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss : scalar
            loss_dict  : breakdown for logging
        """
        # ── Supervised BCE loss ───────────────────────────────────────────
        bce_loss = self.bce(logits.squeeze(1), labels)

        # ── Physics consistency loss ──────────────────────────────────────
        phys_loss = self.physics_consistency_loss(mps_score, physics_feat)

        # ── Calibration loss ──────────────────────────────────────────────
        calib_loss = self.calibration_loss(mps_score, labels)

        # ── Total ─────────────────────────────────────────────────────────
        total = (bce_loss
                 + self.lambda_phys  * phys_loss
                 + self.lambda_calib * calib_loss)

        loss_dict = {
            "bce":         bce_loss.item(),
            "physics":     phys_loss.item(),
            "calibration": calib_loss.item(),
            "total":       total.item(),
        }

        return total, loss_dict


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(n_features:    int,
                class_weights: torch.Tensor = None,
                config:        dict         = None) -> tuple:
    """
    Build MPSModel + PhysicsInformedLoss from a config dict.
    Returns (model, loss_fn).
    """
    cfg = config or {}
    model = MPSModel(
        n_tabular_features  = n_features,
        gnn_hidden          = cfg.get("gnn_hidden", 64),
        gnn_out_dim         = cfg.get("gnn_out_dim", 128),
        transformer_d       = cfg.get("transformer_d", 128),
        transformer_heads   = cfg.get("transformer_heads", 4),
        transformer_layers  = cfg.get("transformer_layers", 3),
        transformer_out     = cfg.get("transformer_out", 128),
        tabular_out         = cfg.get("tabular_out", 64),
        hidden_dims         = cfg.get("hidden_dims", [256, 128, 64]),
        dropout             = cfg.get("dropout", 0.3),
    )

    loss_fn = PhysicsInformedLoss(
        class_weights = class_weights,
        lambda_phys   = cfg.get("lambda_phys", 0.15),
        lambda_calib  = cfg.get("lambda_calib", 0.05),
    )

    return model, loss_fn


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.models.dataset import CRCMetastasisDataset, collate_fn
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("  Phase 4 — Step 4: Full MPS Model self-test")
    print("=" * 60)

    ds     = CRCMetastasisDataset()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=False)
    batch  = next(iter(loader))

    n_features = batch["tabular"].shape[1]
    cw         = ds.get_class_weights()
    model, loss_fn = build_model(n_features, cw)

    model.eval()
    with torch.no_grad():
        # Approximate n_nodes from batch (use max_nodes for all)
        max_n  = batch["node_features"].shape[1]
        n_nodes = torch.full((8,), max_n, dtype=torch.long)

        logits, scores = model(
            batch["tabular"],
            batch["node_features"],
            batch["edge_index"],
            batch["temporal_seq"],
            n_nodes,
        )

    print(f"\n  Forward pass output:")
    print(f"    logits  : {tuple(logits.shape)}  range [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"    scores  : {tuple(scores.shape)}  range [{scores.min():.3f}, {scores.max():.3f}]")

    # Test physics-informed loss
    total, breakdown = loss_fn(logits, scores, batch["label"],
                               batch["physics_features"])
    print(f"\n  Loss breakdown:")
    for k, v in breakdown.items():
        print(f"    {k:<14}: {v:.6f}")

    # Test MC dropout uncertainty
    print(f"\n  MC Dropout uncertainty (10 samples):")
    mps_mean, mps_std, ci_lo, ci_hi = model.predict_with_uncertainty(
        batch["tabular"], batch["node_features"], batch["edge_index"],
        batch["temporal_seq"], n_nodes, n_samples=10
    )
    print(f"    MPS mean : {mps_mean.squeeze().tolist()[:4]}")
    print(f"    MPS std  : {mps_std.squeeze().tolist()[:4]}")
    print(f"    95% CI   : [{ci_lo.min():.3f}, {ci_hi.max():.3f}]")

    print("\n  ✓ Full model OK — ready for training")

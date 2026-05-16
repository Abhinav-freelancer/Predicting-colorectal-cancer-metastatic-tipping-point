"""
Phase 4 - Step 3: Temporal Transformer
=======================================
Processes pseudo-temporal patient sequences and extracts a
context-aware embedding capturing disease trajectory dynamics.

Architecture:
    Input: (batch, seq_len, n_features) — pseudo-time sequences
    Positional encoding: learnable position embeddings
    Linear projection: n_features → d_model
    N × TransformerEncoder layers (multi-head self-attention + FFN)
    Output token aggregation: mean of all positions
    Linear projection → transformer_out_dim

Design choices:
    - Learnable (not sinusoidal) positional encoding: better for short
      sequences (seq_len=8) where relative position matters more than
      absolute position
    - Pre-LayerNorm (more stable training than post-LayerNorm)
    - CLS token prepended: dedicated aggregation token learns to
      summarise the full sequence — standard BERT-style approach

Usage:
    python src/models/transformer.py   # self-test
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


# ── Positional encoding ───────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings.
    Better than sinusoidal for short sequences — the model can learn
    which positions in the disease trajectory are most informative.
    """
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS
        self.dropout       = nn.Dropout(dropout)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return self.dropout(x + self.pos_embedding(positions))


# ── Pre-LayerNorm Transformer block ──────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with Pre-LayerNorm.

    Pre-LN: LayerNorm applied BEFORE attention and FFN (not after).
    Advantages:
        - More stable gradients during early training
        - Less sensitive to learning rate
        - Better performance on small datasets

    Structure:
        x → LN → MultiHeadAttn → residual
        x → LN → FFN           → residual
    """
    def __init__(self,
                 d_model:     int,
                 n_heads:     int,
                 ffn_dim:     int,
                 dropout:     float = 0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True,   # (B, L, D) convention
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),             # GELU outperforms ReLU in transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self,
                x:           torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ── Full Temporal Transformer ─────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Transformer encoder that processes pseudo-temporal patient sequences.

    Architecture:
        Linear(n_features → d_model)     input projection
        Prepend CLS token                 learns global sequence summary
        LearnablePositionalEncoding       position-aware
        N × TransformerBlock              contextual representation
        Extract CLS token output          (B, d_model)
        Linear(d_model → out_dim)         final projection
    """

    def __init__(self,
                 n_features:  int,
                 d_model:     int   = 128,
                 n_heads:     int   = 4,
                 n_layers:    int   = 3,
                 ffn_dim:     int   = 256,
                 out_dim:     int   = 128,
                 max_seq_len: int   = 16,
                 dropout:     float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.out_dim = out_dim

        # Input projection: n_features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Learnable CLS token (one per item in batch, shared weights)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional encoding (+1 for CLS token position)
        self.pos_enc = LearnablePositionalEncoding(max_seq_len, d_model, dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final norm + output projection
        self.final_norm  = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                seq: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        seq          : (B, seq_len, n_features)
        padding_mask : (B, seq_len) bool, True = padded position to ignore
        Returns      : (B, out_dim)
        """
        B, L, _ = seq.shape

        # Project input features to d_model
        x = self.input_proj(seq)           # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, self.d_model)
        x   = torch.cat([cls, x], dim=1)   # (B, L+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Extend padding mask for CLS token (never masked)
        if padding_mask is not None:
            cls_mask  = torch.zeros(B, 1, dtype=torch.bool,
                                    device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        # Extract CLS token output (position 0) and normalise
        cls_out = self.final_norm(x[:, 0, :])   # (B, d_model)
        return self.output_proj(cls_out)          # (B, out_dim)

    def get_attention_weights(self,
                              seq: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract attention weights from all layers for interpretability.
        Returns list of (B, n_heads, seq_len+1, seq_len+1) tensors.
        """
        B, L, _ = seq.shape
        x = self.input_proj(seq)
        cls = self.cls_token.expand(B, 1, self.d_model)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_enc(x)

        attn_weights = []
        for layer in self.layers:
            x_norm   = layer.norm1(x)
            _, weights = layer.attn(x_norm, x_norm, x_norm,
                                    need_weights=True,
                                    average_attn_weights=False)
            attn_weights.append(weights)
            x = x + layer.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
            x = x + layer.ffn(layer.norm2(x))

        return attn_weights


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.models.dataset import CRCMetastasisDataset, collate_fn
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("  Phase 4 — Step 3: Temporal Transformer self-test")
    print("=" * 60)

    ds     = CRCMetastasisDataset()
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=False)
    batch  = next(iter(loader))

    n_features = batch["temporal_seq"].shape[-1]
    model = TemporalTransformer(
        n_features  = n_features,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 3,
        ffn_dim     = 256,
        out_dim     = 128,
        max_seq_len = 16,
        dropout     = 0.1,
    )
    print(f"\n  Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        out = model(batch["temporal_seq"])

    print(f"  Input  seq shape     : {tuple(batch['temporal_seq'].shape)}")
    print(f"  Output embedding     : {tuple(out.shape)}")
    print(f"  Output range         : [{out.min():.3f}, {out.max():.3f}]")
    print("\n  ✓ Transformer OK")

"""
GSE39582 External Validation
==============================
Loads trained ensemble checkpoints and evaluates on GSE39582 data.

Usage:
    python scripts/validate_gse39582.py \\
        --checkpoint-dir experiments/runs/phase4_run \\
        --gse-features data/processed/geo/gse39582_features.csv \\
        --gse-vst data/processed/geo/gse39582_rescaled.csv \\
        --ensemble-seeds "42,123,777"
"""

import sys, os, json, argparse, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.models.dataset import CRCMetastasisDataset, collate_fn, _load_symbol_to_ensg, build_gene_graph, build_temporal_sequence
from src.models.model   import build_model


class GEODataset(Dataset):
    """
    Minimal dataset wrapper for GEO feature matrix + VST expression.
    Must produce items compatible with MPSModel.forward():
      - tabular
      - node_features
      - edge_index
      - temporal_seq
      - n_nodes
    """
    def __init__(self, feature_csv: str, vst_csv: str, seq_len: int = 8):
        self.seq_len = seq_len

        df = pd.read_csv(feature_csv, index_col=0)
        self.labels = df["metastasis_label"].values.astype(float)
        feature_cols = [c for c in df.columns if c != "metastasis_label"]
        arr = df[feature_cols].values.astype(np.float32)
        # Within-patient percentile ranks — platform-agnostic
        for i in range(arr.shape[0]):
            arr[i] = rankdata(arr[i], method="average") / arr.shape[1]
        self.features_arr = arr
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)

        # Physics feature indices
        self.PHYSICS_COLS = [
            "attractor_proximity","bifurcation_score","physics_score",
            "fitted_T_ext","in_tipping_zone","epi_dist","mes_dist",
        ]
        self.physics_idx = [feature_cols.index(c)
                            for c in self.PHYSICS_COLS
                            if c in feature_cols]

        self.patient_ids = df.index.tolist()

        # EMT index for temporal ordering
        self.emt_index = (df["emt_index"].values.astype(float)
                          if "emt_index" in df.columns
                          else np.zeros(len(df)))

        # Load VST (rescaled GEO expression) and build graphs
        print("  Loading rescaled GEO expression for graph construction...")
        vst_full = pd.read_csv(vst_csv, index_col=0)

        print("  Pre-building gene graphs...")
        self.graphs = []
        for pid in self.patient_ids:
            col = vst_full[pid] if pid in vst_full.columns else \
                  pd.Series(np.zeros(len(vst_full)), index=vst_full.index)
            self.graphs.append(build_gene_graph(col))

        n_pos = int(self.labels.sum())
        print(f"  GEO Dataset: {len(self)} patients | {self.n_features} features | "
              f"{n_pos} metastatic / {len(self)-n_pos} non-metastatic")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features_arr[idx]
        label = self.labels[idx]
        graph = self.graphs[idx]

        # Build synthetic temporal sequence (nearest neighbours by EMT index)
        emt = self.emt_index
        emt_i = emt[idx]
        seq = build_temporal_sequence(idx, self.features_arr,
                                       self.emt_index, self.seq_len)

        # Physics features
        if self.physics_idx:
            physics = feat[self.physics_idx].astype(np.float32)
        else:
            physics = np.zeros(7, dtype=np.float32)

        node_features, edge_index = graph

        return {
            "tabular":          torch.tensor(feat, dtype=torch.float),
            "node_features":    node_features,
            "edge_index":       edge_index,
            "temporal_seq":     seq,
            "physics_features": torch.tensor(physics, dtype=torch.float),
            "label":            torch.tensor(label, dtype=torch.float),
        }


def _build_feature_mapper(ckpt_feature_names: list, geo_feature_names: list):
    """
    Build an index array to reorder/pad GEO features to match checkpoint ordering.
    Missing features -> 0 via index -1.
    """
    mapper = []
    geo_index = {name: i for i, name in enumerate(geo_feature_names)}
    for name in ckpt_feature_names:
        mapper.append(geo_index.get(name, -1))
    return np.array(mapper, dtype=np.int64)


def _apply_feature_mapper(tabular_batch: torch.Tensor, mapper: np.ndarray, device):
    """
    Reorder and pad a batch of tabular features using the mapper.
    -1 entries become zeros.
    """
    batch_size = tabular_batch.shape[0]
    n_ckpt = len(mapper)
    out = torch.zeros(batch_size, n_ckpt, dtype=tabular_batch.dtype, device=device)
    geo_np = tabular_batch.cpu().numpy()
    for j, src_idx in enumerate(mapper):
        if src_idx >= 0:
            out[:, j] = torch.tensor(geo_np[:, src_idx], dtype=out.dtype, device=device)
    return out


def load_ensemble_checkpoints(checkpoint_dir: str, n_folds: int,
                               seeds: list, model_config: dict,
                               geo_feature_names: list, device: torch.device):
    """Load all fold checkpoints per seed into a list of (seed, fold, model, mapper)."""
    ckpt_dir = Path(checkpoint_dir)
    models = []

    for seed in seeds:
        for fold in range(n_folds):
            ckpt_path = ckpt_dir / f"fold{fold}_seed{seed}_best.pt"
            if not ckpt_path.exists():
                ckpt_path = ckpt_dir / f"fold{fold}_best.pt"
                if not ckpt_path.exists():
                    print(f"  Warning: checkpoints for seed={seed} fold={fold} not found, skipping")
                    continue

            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_feature_names = state.get("feature_names")
            if ckpt_feature_names is None:
                # Fallback: infer n_features from the saved state dict
                sk = state["model_state"]
                tab_dim = sk["transformer.input_proj.0.weight"].shape[1]
                model_n_features = tab_dim
                mapper = _build_feature_mapper(
                    [f"feature_{i}" for i in range(model_n_features)], geo_feature_names
                )
                print(f"  Loaded seed={seed} fold={fold} (epoch {state.get('epoch', '?')})"
                      f"  tabular_dim={model_n_features} (inferred)")
            else:
                model_n_features = len(ckpt_feature_names)
                mapper = _build_feature_mapper(ckpt_feature_names, geo_feature_names)

            model, _ = build_model(model_n_features, config=model_config)
            model.load_state_dict(state["model_state"])
            model = model.to(device)
            model.eval()
            models.append((seed, fold, model, mapper))
            print(f"  Loaded seed={seed} fold={fold} (epoch {state.get('epoch', '?')})"
                  f"  tabular_dim={model_n_features}")

    print(f"  Total ensemble models loaded: {len(models)}")
    return models


def _apply_feature_mapper_temporal(temporal_batch: torch.Tensor, mapper: np.ndarray, device):
    """
    Reorder/pad temporal sequence features.
    temporal_batch: (B, L, n_geo_features) -> (B, L, n_ckpt_features)
    """
    B, L, _ = temporal_batch.shape
    n_ckpt = len(mapper)
    out = torch.zeros(B, L, n_ckpt, dtype=temporal_batch.dtype, device=device)
    temporal_np = temporal_batch.cpu().numpy()
    for j, src_idx in enumerate(mapper):
        if src_idx >= 0:
            out[:, :, j] = torch.tensor(temporal_np[:, :, src_idx],
                                        dtype=out.dtype, device=device)
    return out


def predict_ensemble(models, loader, device, mc_samples=10):
    """Ensemble prediction with MC dropout and feature remapping."""
    all_scores = []
    all_labels = []

    for batch in loader:
        tabular  = batch["tabular"].to(device)
        node_f   = batch["node_features"].to(device)
        edge_i   = batch["edge_index"].to(device)
        temporal = batch["temporal_seq"].to(device)
        labels   = batch["label"].numpy()

        n_nodes = torch.full((tabular.shape[0],),
                             node_f.shape[1],
                             dtype=torch.long, device=device)

        member_scores = []
        for _, _, model, mapper in models:
            model_tabular = _apply_feature_mapper(tabular, mapper, device)
            model_temporal = _apply_feature_mapper_temporal(temporal, mapper, device)

            with torch.no_grad():
                mc_preds = []
                for _ in range(mc_samples):
                    model.train()
                    _, scores = model(model_tabular, node_f, edge_i,
                                      model_temporal, n_nodes)
                    model.eval()
                    mc_preds.append(scores.cpu().numpy())
                member_mean = np.mean(mc_preds, axis=0)
                member_scores.append(member_mean)

        ensemble_scores = np.mean(member_scores, axis=0)
        all_scores.extend(ensemble_scores.squeeze())
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_scores)


def parse_args():
    parser = argparse.ArgumentParser(description="GSE39582 external validation")
    parser.add_argument("--checkpoint-dir", default="experiments/runs/phase4_run")
    parser.add_argument("--gse-features", default="data/processed/geo/gse39582_features.csv")
    parser.add_argument("--gse-vst", default="data/processed/geo/gse39582_rescaled.csv")
    parser.add_argument("--ensemble-seeds", type=str, default="42,123,777")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(s.strip()) for s in args.ensemble_seeds.split(",") if s.strip()]

    print("=" * 60)
    print("  GSE39582 External Validation")
    print("=" * 60)

    # Load GEO dataset
    print("\n  Building GEO dataset...")
    geo_dataset = GEODataset(args.gse_features, args.gse_vst)
    loader = DataLoader(geo_dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    # Model config (must match training config)
    model_config = {
        "gnn_hidden":         48,
        "gnn_out_dim":        96,
        "transformer_d":      96,
        "transformer_heads":  3,
        "transformer_layers": 2,
        "transformer_ffn_dim": 192,
        "transformer_out":    96,
        "tabular_out":        48,
        "cross_attn_heads":   3,
        "hidden_dims":        [128, 64],
        "dropout":            0.5,
        "lambda_phys":        0.25,
        "lambda_calib":       0.05,
    }

    # Load ensemble checkpoints
    print("\n  Loading ensemble checkpoints...")
    models = load_ensemble_checkpoints(
        args.checkpoint_dir, args.folds, seeds, model_config,
        geo_dataset.feature_names, device,
    )

    if len(models) == 0:
        print("  ERROR: No checkpoints loaded. Exiting.")
        sys.exit(1)

    # Predict
    print("\n  Running ensemble predictions...")
    labels, scores = predict_ensemble(models, loader, device, args.mc_samples)

    # Metrics
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    print(f"\n  GSE39582 Validation Results:")
    print(f"    Patients : {len(labels)}")
    print(f"    M1 rate  : {labels.mean():.2%}")
    print(f"    AUROC    : {auroc:.4f}")
    print(f"    AUPRC    : {auprc:.4f}")

    # Save results
    out_dir = Path(args.checkpoint_dir)
    results = {
        "dataset": "GSE39582",
        "n_patients": len(labels),
        "m1_rate": float(labels.mean()),
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "ensemble_seeds": seeds,
        "n_folds": args.folds,
        "n_ensemble_models": len(models),
    }
    results_path = out_dir / "gse39582_validation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Phase 4 - Step 5: Training loop
=================================
Full training pipeline with:
  - 5-fold stratified cross-validation
  - AdamW optimiser + cosine annealing LR scheduler
  - Early stopping (patience=15 on val AUROC)
  - MLflow experiment tracking
  - Gradient clipping (prevents exploding gradients)
  - Best model checkpointing per fold

Usage:
    python src/models/train.py
    python src/models/train.py --epochs 50 --batch-size 16 --folds 3
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.dataset     import CRCMetastasisDataset, collate_fn, get_stratified_folds
from src.models.model       import build_model
from sklearn.neighbors      import NearestNeighbors
from src.models.feature_select import MIFeatureSelector


# ── Mixup augmentation ─────────────────────────────────────────────────────

class MixupWrapper:
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    def __call__(self, batch):
        if __import__('numpy').random.random() >= self.prob:
            return batch
        B = batch['label'].shape[0]
        lam = __import__('numpy').random.beta(self.alpha, self.alpha)
        idx = __import__('torch').randperm(B)
        batch['label'] = lam * batch['label'] + (1 - lam) * batch['label'][idx]
        batch['tabular'] = lam * batch['tabular'] + (1 - lam) * batch['tabular'][idx]
        batch['node_features'] = lam * batch['node_features'] + (1 - lam) * batch['node_features'][idx]
        batch['temporal_seq'] = lam * batch['temporal_seq'] + (1 - lam) * batch['temporal_seq'][idx]
        batch['physics_features'] = lam * batch['physics_features'] + (1 - lam) * batch['physics_features'][idx]
        return batch

# â”€â”€ SMOTE-augmented dataset wrapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SMOTEDataset(torch.utils.data.Dataset):
    """
    Wraps a base dataset with SMOTE-oversampled tabular features.
    For synthetic samples, graph/sequence data is copied from the
    nearest real M1 (metastatic) neighbour in the training fold.
    """
    def __init__(self, base_dataset, train_indices, seed=42):
        self.base          = base_dataset
        self.train_indices = list(train_indices)

        X = base_dataset.features_arr[train_indices]
        y = base_dataset.labels[train_indices]

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.667, random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)

        self.n_orig  = len(train_indices)
        self.n_synth = len(X_res) - self.n_orig

        self.X_res = torch.tensor(X_res, dtype=torch.float)
        self.y_res = torch.tensor(y_res, dtype=torch.float)

        # Map each synthetic sample to its nearest M1 neighbour
        self.synth_source = []
        if self.n_synth > 0:
            m1_mask = (y == 1)
            if m1_mask.sum() > 0:
                m1_idx    = np.where(m1_mask)[0]
                m1_X      = X[m1_mask]
                synth_X   = X_res[self.n_orig:]
                nn        = NearestNeighbors(n_neighbors=1).fit(m1_X)
                _, nearest = nn.kneighbors(synth_X)
                self.synth_source = [train_indices[m1_idx[n[0]]] for n in nearest]
            else:
                self.synth_source = [train_indices[0]] * self.n_synth

    def __len__(self):
        return len(self.X_res)

    def __getitem__(self, idx):
        if idx < self.n_orig:
            return self.base[self.train_indices[idx]]
        src = self.synth_source[idx - self.n_orig]
        item = dict(self.base[src])
        item["tabular"] = self.X_res[idx]
        item["label"]   = self.y_res[idx]
        # Recompute physics features to match synthetic tabular
        if self.base.physics_idx:
            item["physics_features"] = self.X_res[idx][self.base.physics_idx]
        return item

# Optional MLflow â€” skip gracefully if not installed
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("  â„¹ MLflow not installed â€” logging to JSON only")


# â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute AUROC, AUPRC, and threshold-based metrics."""
    try:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auroc = auprc = 0.5

    # Threshold at 0.5 for binary metrics
    preds  = (scores >= 0.5).astype(int)
    tp     = ((preds == 1) & (labels == 1)).sum()
    fp     = ((preds == 1) & (labels == 0)).sum()
    fn     = ((preds == 0) & (labels == 1)).sum()
    tn     = ((preds == 0) & (labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "auroc":       round(float(auroc),       4),
        "auprc":       round(float(auprc),        4),
        "f1":          round(float(f1),           4),
        "precision":   round(float(precision),    4),
        "recall":      round(float(recall),       4),
        "specificity": round(float(specificity),  4),
    }


# â”€â”€ Training step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_epoch(model, loader, optimiser, loss_fn,
                device, max_norm=1.0,
                accumulation_steps=4,
                mixup_fn=None) -> dict:
    model.train()
    total_loss   = 0.0
    all_labels   = []
    all_scores   = []
    loss_details = {"bce": 0, "physics": 0, "calibration": 0}

    optimiser.zero_grad()

    for i, batch in enumerate(loader):
        # Apply mixup augmentation if provided
        if mixup_fn is not None:
            batch = mixup_fn(batch)

        # Move to device
        tabular   = batch["tabular"].to(device)
        node_f    = batch["node_features"].to(device)
        edge_i    = batch["edge_index"].to(device)
        temporal  = batch["temporal_seq"].to(device)
        physics   = batch["physics_features"].to(device)
        labels    = batch["label"].to(device)

        # Approximate n_nodes (use max padded size â€” safe for SAGEConv)
        n_nodes = torch.full((tabular.shape[0],),
                             node_f.shape[1],
                             dtype=torch.long, device=device)

        logits, scores = model(tabular, node_f, edge_i, temporal, n_nodes)
        loss, breakdown = loss_fn(logits, scores, labels, physics)

        (loss / accumulation_steps).backward()

        total_loss += loss.item()
        for k in loss_details:
            loss_details[k] += breakdown.get(k, 0)

        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.detach().squeeze(1).cpu().numpy())

        if (i + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimiser.step()
            optimiser.zero_grad()

    # Flush remaining gradients
    if (i + 1) % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimiser.step()
        optimiser.zero_grad()

    n = len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["loss"]         = round(total_loss / n, 5)
    metrics["loss_bce"]     = round(loss_details["bce"] / n, 5)
    metrics["loss_physics"] = round(loss_details["physics"] / n, 5)
    return metrics


# â”€â”€ Validation step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def val_epoch(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch in loader:
            tabular  = batch["tabular"].to(device)
            node_f   = batch["node_features"].to(device)
            edge_i   = batch["edge_index"].to(device)
            temporal = batch["temporal_seq"].to(device)
            physics  = batch["physics_features"].to(device)
            labels   = batch["label"].to(device)

            n_nodes = torch.full((tabular.shape[0],),
                                 node_f.shape[1],
                                 dtype=torch.long, device=device)

            logits, scores = model(tabular, node_f, edge_i, temporal, n_nodes)
            loss, _        = loss_fn(logits, scores, labels, physics)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.squeeze(1).cpu().numpy())

    metrics         = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["loss"] = round(total_loss / len(loader), 5)
    return metrics, np.array(all_labels), np.array(all_scores)


# â”€â”€ Training run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_fold(fold:       int,
               model:      nn.Module,
               loss_fn:    nn.Module,
               train_loader: DataLoader,
               val_loader:   DataLoader,
               epochs:     int,
               lr:         float,
               patience:   int,
               out_dir:    Path,
               device:     torch.device,
               accumulation_steps: int = 4,
               mixup_fn:    object = None,
               seed: int = None,
               feature_names: list = None) -> dict:
    """Train one cross-validation fold. Returns best val metrics."""

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_val_metrics = {}
    seed_tag = f"_seed{seed}" if seed is not None else ""
    best_ckpt_path   = out_dir / f"fold{fold}{seed_tag}_best.pt"
    history          = []

    print(f"\n  Fold {fold+1} training ({epochs} epochs, patience={patience}, "
          f"accum={accumulation_steps})")
    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAUROC':>8}  "
          f"{'VaLoss':>8}  {'VaAUROC':>8}  {'VaF1':>6}  {'LR':>8}")
    print(f"  {'-'*65}")

    mixup = mixup_fn  # None = no mixup; callable = apply mixup

    for epoch in range(epochs):
        tr = train_epoch(model, train_loader, optimiser, loss_fn, device,
                         accumulation_steps=accumulation_steps,
                         mixup_fn=mixup)
        va, va_labels, va_scores = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        current_lr = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, **{f"tr_{k}": v for k, v in tr.items()},
                         **{f"va_{k}": v for k, v in va.items()}})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {epoch+1:>4}  {tr['loss']:>8.4f}  {tr['auroc']:>8.4f}  "
                  f"{va['loss']:>8.4f}  {va['auroc']:>8.4f}  "
                  f"{va['f1']:>6.4f}  {current_lr:>8.6f}")

        # Checkpointing and early stopping on val loss
        if va["loss"] < best_val_loss:
            best_val_loss    = va["loss"]
            best_val_metrics = va.copy()
            best_val_metrics["best_epoch"]    = epoch
            best_val_metrics["val_labels"]    = va_labels.tolist()
            best_val_metrics["val_scores"]    = va_scores.tolist()
            patience_ctr     = 0
            ckpt = {
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimiser":    optimiser.state_dict(),
                "val_metrics":  va,
            }
            if feature_names is not None:
                ckpt["feature_names"] = feature_names
            torch.save(ckpt, best_ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stop at epoch {epoch+1} "
                      f"(no val loss improvement for {patience} epochs)")
                break

    # Save training history
    pd.DataFrame(history).to_csv(out_dir / f"fold{fold}_history.csv", index=False)
    best_auroc_val = best_val_metrics.get("auroc", 0.0)
    print(f"\n  Fold {fold+1} best: VaLoss={best_val_loss:.4f}  "
          f"VaAUROC={best_auroc_val:.4f}  "
          f"(epoch {best_val_metrics.get('best_epoch',0)+1})")

    return best_val_metrics


# â”€â”€ Cross-validation orchestrator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def cross_validate(dataset:     CRCMetastasisDataset,
                   n_folds:     int,
                   epochs:      int,
                   lr:          float,
                   patience:    int,
                   out_dir:     Path,
                   device:      torch.device,
                   model_config: dict,
                   use_smote:   bool = True,
                    accumulation_steps: int = 4,
                    mi_k:        int = 0,
                    use_mixup:   bool = False,
                    splits:      list = None,
                    seed:        int = None) -> dict:

    fold_results = []
    all_val_labels, all_val_scores = [], []

    cw = dataset.get_class_weights().to(device)

    # Save original state for restoration between folds
    orig_features_arr = dataset.features_arr.copy()
    orig_n_features = dataset.n_features
    orig_feature_names = dataset.feature_names.copy()

    for fold, train_loader, val_loader, tr, va in \
            get_stratified_folds(dataset, n_splits=n_folds,
                                 precomputed_splits=splits):

        # Restore original features for this fold
        dataset.features_arr = orig_features_arr.copy()
        dataset.n_features = orig_n_features
        dataset.feature_names = orig_feature_names.copy()
        dataset.physics_idx = [orig_feature_names.index(c)
                    for c in CRCMetastasisDataset.PHYSICS_COLS
                    if c in orig_feature_names]

        # Apply MI feature selection on training fold (no leakage)
        if mi_k > 0:
            X_tr = dataset.features_arr[tr]
            y_tr = dataset.labels[tr]
            selector = MIFeatureSelector(k=mi_k)
            selector.fit(X_tr, y_tr, dataset.feature_names)
            # Update dataset to use only selected features
            dataset.features_arr = dataset.features_arr[:, selector.selected_indices]
            dataset.n_features = len(selector.selected_indices)
            dataset.feature_names = selector.selected_names
            dataset.physics_idx = [dataset.feature_names.index(c)
                        for c in CRCMetastasisDataset.PHYSICS_COLS
                        if c in dataset.feature_names]
            print(f"  MI selection: {X_tr.shape[1]} -> {len(selector.selected_indices)} features")
            print(f"  Selected: {selector.selected_names[:6]}...")

        # Apply SMOTE oversampling on training fold (after MI selection)
        if use_smote:
            smote_ds = SMOTEDataset(dataset, tr, seed=42 + fold)
            train_loader = DataLoader(
                smote_ds, batch_size=train_loader.batch_size,
                shuffle=True, collate_fn=collate_fn, num_workers=0,
            )
            print(f"  SMOTE applied: {len(tr)} -> {len(smote_ds)} training samples")
            X_tr = dataset.features_arr[tr]
            y_tr = dataset.labels[tr]
            selector = MIFeatureSelector(k=mi_k)
            selector.fit(X_tr, y_tr, dataset.feature_names)
            # Update dataset to use only selected features
            dataset.features_arr = dataset.features_arr[:, selector.selected_indices]
            dataset.n_features = len(selector.selected_indices)
            dataset.feature_names = selector.selected_names
            dataset.physics_idx = [dataset.feature_names.index(c)
                        for c in CRCMetastasisDataset.PHYSICS_COLS
                        if c in dataset.feature_names]
            print(f"  MI selection: {X_tr.shape[1]} -> {len(selector.selected_indices)} features")
            print(f"  Selected: {selector.selected_names[:6]}...")

        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        model, loss_fn = build_model(dataset.n_features, cw, model_config)
        model          = model.to(device)
        loss_fn        = loss_fn.to(device)

        mixup_fn = MixupWrapper() if use_mixup else None
        best = train_fold(
            fold, model, loss_fn,
            train_loader, val_loader,
            epochs, lr, patience, out_dir, device,
            accumulation_steps=accumulation_steps,
            mixup_fn=mixup_fn,
            seed=seed,
            feature_names=list(dataset.feature_names),
        )

        fold_results.append(best)
        if "val_labels" in best:
            all_val_labels.extend(best.pop("val_labels"))
            all_val_scores.extend(best.pop("val_scores"))

    # Aggregate results
    agg = {}
    metric_keys = [k for k in fold_results[0] if k not in ("best_epoch",)]
    for key in metric_keys:
        vals = [f[key] for f in fold_results if key in f]
        if vals and isinstance(vals[0], (int, float)):
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"]  = round(float(np.std(vals)),  4)

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS ({n_folds} folds)")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<20} {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*40}")
    for key in ["auroc", "auprc", "f1", "precision", "recall"]:
        m = agg.get(f"{key}_mean", 0)
        s = agg.get(f"{key}_std",  0)
        print(f"  {key:<20} {m:>8.4f}  {s:>8.4f}")

    # Overall OOF (out-of-fold) AUROC
    oof_labels_arr = np.array(all_val_labels) if all_val_labels else np.array([])
    oof_scores_arr = np.array(all_val_scores) if all_val_scores else np.array([])
    if len(oof_labels_arr) > 1 and len(np.unique(oof_labels_arr)) > 1:
        oof_auroc = roc_auc_score(oof_labels_arr, oof_scores_arr)
        agg["oof_auroc"] = round(float(oof_auroc), 4)
        print(f"\n  Out-of-fold AUROC: {oof_auroc:.4f}")

    # Apply isotonic calibration to OOF scores
    if len(oof_labels_arr) > 1 and len(oof_scores_arr) > 1:
        try:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            oof_calibrated = calibrator.fit_transform(oof_scores_arr, oof_labels_arr)
            cal_auroc = roc_auc_score(oof_labels_arr, oof_calibrated)
            agg["calibrated_oof_auroc"] = round(float(cal_auroc), 4)
            print(f"  Calibrated OOF AUROC: {cal_auroc:.4f}")
        except Exception as e:
            print(f"  Calibration failed: {e}")

    # Store OOF predictions for ensemble
    agg["oof_labels"] = oof_labels_arr.tolist()
    agg["oof_scores"] = oof_scores_arr.tolist()

    return agg


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="Phase 4 - MPS model training")
    parser.add_argument("--phase4-input",  default="data/processed/temporal/phase4_input_clean.csv")
    parser.add_argument("--manifest-dir",  default="data/manifests")
    parser.add_argument("--vst-input",     default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--out-dir",       default="experiments/runs")
    parser.add_argument("--epochs",        type=int,   default=80)
    parser.add_argument("--folds",         type=int,   default=3)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lambda-phys",   type=float, default=0.25)
    parser.add_argument("--accumulation-steps", type=int,   default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--no-smote", action="store_true",
                        help="Disable SMOTE oversampling on training folds")
    parser.add_argument("--mixup", action="store_true",
                        help="Enable Mixup augmentation (default: off)")
    parser.add_argument("--mi-k",          type=int,   default=30,
                        help="Mutual information feature selection k (0=disable)")
    parser.add_argument("--ensemble-seeds", type=str, default="42",
                        help="Comma-separated seeds for ensemble training")
    parser.add_argument("--device",        default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / "phase4_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    print("=" * 60)
    print("  Phase 4 â€” MPS Model Training")
    print("=" * 60)

    # Dataset
    dataset = CRCMetastasisDataset(
        phase4_path   = args.phase4_input,
        manifest_path = f"{args.manifest_dir}/cohort_labeled.csv",
        vst_path      = args.vst_input,
    )

    model_config = {
        "gnn_hidden":         48,
        "gnn_out_dim":        96,
        "transformer_d":      96,
        "transformer_heads":  3,
        "transformer_layers": 2,
        "transformer_out":    96,
        "transformer_ffn_dim": 192,
        "tabular_out":        48,
        "cross_attn_heads":   3,
        "hidden_dims":        [128, 64],
        "dropout":            args.dropout,
        "lambda_phys":        args.lambda_phys,
        "lambda_calib":       0.05,
    }

    # MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(str(Path("experiments/mlflow")))
        mlflow.set_experiment("CRC_Metastasis_MPS")
        mlflow.start_run(run_name=f"phase4_cv{args.folds}fold")
        mlflow.log_params({**model_config, "epochs": args.epochs,
                           "lr": args.lr, "folds": args.folds})

    # Build model_config with training improvements
    model_config.update({
        "lambda_phys":    args.lambda_phys,
        "accumulation_steps": args.accumulation_steps,
        "focal_loss":     True,
        "focal_alpha":    0.75,
        "focal_gamma":    2.0,
        "label_smoothing": 0.1,
    })

    # Parse ensemble seeds
    seeds = [int(s.strip()) for s in args.ensemble_seeds.split(",") if s.strip()]
    print(f"\n  Ensemble seeds: {seeds}")

    # Compute fixed CV splits once (same folds across all ensemble seeds)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    labels_arr = dataset.labels.astype(int)
    indices_arr = np.arange(len(dataset))
    fixed_splits = list(skf.split(indices_arr, labels_arr))
    print(f"\n  Fixed CV splits: {args.folds} folds, "
          f"{len(fixed_splits[0][0])} train / {len(fixed_splits[0][1])} val per fold")

    # Run ensemble training (multiple seeds)
    all_ensemble_results = []
    all_oof_scores_by_seed = []
    all_oof_labels_common = None

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE MEMBER {seed_idx+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")

        # Set seed for reproducibility (affects model init, SMOTE, dropout)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Re-create dataset (to reset any fold-level modifications)
        dataset_fold = CRCMetastasisDataset(
            phase4_path   = args.phase4_input,
            manifest_path = f"{args.manifest_dir}/cohort_labeled.csv",
            vst_path      = args.vst_input,
        )

        # Run cross-validation for this seed with fixed splits
        results = cross_validate(
            dataset_fold, args.folds, args.epochs, args.lr,
            args.patience, out_dir, device, model_config,
            use_smote=not args.no_smote,
            accumulation_steps=args.accumulation_steps,
            mi_k=args.mi_k,
            use_mixup=args.mixup,
            splits=fixed_splits,
            seed=seed,
        )

        all_ensemble_results.append(results)

        # Collect OOF predictions per seed for ensemble averaging
        if "oof_scores" in results and "oof_labels" in results:
            all_oof_scores_by_seed.append(np.array(results["oof_scores"]))
            if all_oof_labels_common is None:
                all_oof_labels_common = np.array(results["oof_labels"])

    # Ensemble: average predictions across seeds, then recalibrate
    if len(all_oof_scores_by_seed) > 1:
        stacked = np.stack(all_oof_scores_by_seed, axis=1)
        ensemble_scores = np.mean(stacked, axis=1)
        ensemble_auroc = roc_auc_score(all_oof_labels_common, ensemble_scores)
        print(f"\n  Ensemble OOF AUROC (mean of {len(seeds)} seeds): {ensemble_auroc:.4f}")

        try:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrated = calibrator.fit_transform(ensemble_scores, all_oof_labels_common)
            cal_auroc = roc_auc_score(all_oof_labels_common, calibrated)
            print(f"  Ensemble + Calibration OOF AUROC: {cal_auroc:.4f}")
        except Exception as e:
            cal_auroc = ensemble_auroc
            print(f"  Calibration failed (using raw ensemble): {e}")

        results["ensemble_oof_auroc"] = round(float(ensemble_auroc), 4)
        results["calibrated_ensemble_oof_auroc"] = round(float(cal_auroc), 4)
        results["ensemble_seeds"] = seeds
        results["num_ensemble_members"] = len(seeds)

        oof_aurocs = [r.get("oof_auroc", 0) for r in all_ensemble_results]
        print(f"  Individual OOF AUROCs: {[f'{a:.4f}' for a in oof_aurocs]}")

    # Save results
    results_path = out_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({k: v for k, v in results.items()
                            if isinstance(v, float)})
        mlflow.end_run()

    print(f"\n  Phase 4 training complete")
    print(f"  Best OOF AUROC: {results.get('ensemble_oof_auroc_mean', results.get('oof_auroc', 'N/A'))}")
    print(f"  Next: python src/models/evaluate.py   (Phase 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()


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

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.models.dataset     import CRCMetastasisDataset, collate_fn, get_stratified_folds
from src.models.model       import build_model

# Optional MLflow — skip gracefully if not installed
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("  ℹ MLflow not installed — logging to JSON only")


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, loss_fn,
                device, max_norm=1.0) -> dict:
    model.train()
    total_loss   = 0.0
    all_labels   = []
    all_scores   = []
    loss_details = {"bce": 0, "physics": 0, "calibration": 0}

    for batch in loader:
        # Move to device
        tabular   = batch["tabular"].to(device)
        node_f    = batch["node_features"].to(device)
        edge_i    = batch["edge_index"].to(device)
        temporal  = batch["temporal_seq"].to(device)
        physics   = batch["physics_features"].to(device)
        labels    = batch["label"].to(device)

        # Approximate n_nodes (use max padded size — safe for SAGEConv)
        n_nodes = torch.full((tabular.shape[0],),
                             node_f.shape[1],
                             dtype=torch.long, device=device)

        optimiser.zero_grad()
        logits, scores = model(tabular, node_f, edge_i, temporal, n_nodes)
        loss, breakdown = loss_fn(logits, scores, labels, physics)

        loss.backward()
        # Gradient clipping — prevents explosions from stiff ODE-derived features
        nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimiser.step()

        total_loss += loss.item()
        for k in loss_details:
            loss_details[k] += breakdown.get(k, 0)

        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.detach().squeeze(1).cpu().numpy())

    n = len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["loss"]         = round(total_loss / n, 5)
    metrics["loss_bce"]     = round(loss_details["bce"] / n, 5)
    metrics["loss_physics"] = round(loss_details["physics"] / n, 5)
    return metrics


# ── Validation step ───────────────────────────────────────────────────────────

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


# ── Training run ──────────────────────────────────────────────────────────────

def train_fold(fold:       int,
               model:      nn.Module,
               loss_fn:    nn.Module,
               train_loader: DataLoader,
               val_loader:   DataLoader,
               epochs:     int,
               lr:         float,
               patience:   int,
               out_dir:    Path,
               device:     torch.device) -> dict:
    """Train one cross-validation fold. Returns best val metrics."""

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.01
    )

    best_auroc    = 0.0
    patience_ctr  = 0
    best_val_metrics = {}
    best_ckpt_path   = out_dir / f"fold{fold}_best.pt"
    history          = []

    print(f"\n  Fold {fold+1} training ({epochs} epochs, patience={patience})")
    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAUROC':>8}  "
          f"{'VaLoss':>8}  {'VaAUROC':>8}  {'VaF1':>6}  {'LR':>8}")
    print(f"  {'─'*65}")

    for epoch in range(epochs):
        tr = train_epoch(model, train_loader, optimiser, loss_fn, device)
        va, va_labels, va_scores = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        current_lr = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, **{f"tr_{k}": v for k, v in tr.items()},
                         **{f"va_{k}": v for k, v in va.items()}})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {epoch+1:>4}  {tr['loss']:>8.4f}  {tr['auroc']:>8.4f}  "
                  f"{va['loss']:>8.4f}  {va['auroc']:>8.4f}  "
                  f"{va['f1']:>6.4f}  {current_lr:>8.6f}")

        # Checkpointing and early stopping on val AUROC
        if va["auroc"] > best_auroc:
            best_auroc       = va["auroc"]
            best_val_metrics = va.copy()
            best_val_metrics["best_epoch"]    = epoch
            best_val_metrics["val_labels"]    = va_labels.tolist()
            best_val_metrics["val_scores"]    = va_scores.tolist()
            patience_ctr     = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimiser":    optimiser.state_dict(),
                "val_metrics":  va,
            }, best_ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stop at epoch {epoch+1} "
                      f"(no AUROC improvement for {patience} epochs)")
                break

    # Save training history
    pd.DataFrame(history).to_csv(out_dir / f"fold{fold}_history.csv", index=False)
    print(f"\n  Fold {fold+1} best: AUROC={best_auroc:.4f}  "
          f"(epoch {best_val_metrics.get('best_epoch',0)+1})")

    return best_val_metrics


# ── Cross-validation orchestrator ────────────────────────────────────────────

def cross_validate(dataset:     CRCMetastasisDataset,
                   n_folds:     int,
                   epochs:      int,
                   lr:          float,
                   patience:    int,
                   out_dir:     Path,
                   device:      torch.device,
                   model_config: dict) -> dict:

    fold_results = []
    all_val_labels, all_val_scores = [], []

    cw = dataset.get_class_weights().to(device)

    for fold, train_loader, val_loader, _, _ in \
            get_stratified_folds(dataset, n_splits=n_folds):

        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        model, loss_fn = build_model(dataset.n_features, cw, model_config)
        model          = model.to(device)
        loss_fn        = loss_fn.to(device)

        best = train_fold(
            fold, model, loss_fn,
            train_loader, val_loader,
            epochs, lr, patience, out_dir, device
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
    print(f"  {'─'*40}")
    for key in ["auroc", "auprc", "f1", "precision", "recall"]:
        m = agg.get(f"{key}_mean", 0)
        s = agg.get(f"{key}_std",  0)
        print(f"  {key:<20} {m:>8.4f}  {s:>8.4f}")

    # Overall OOF (out-of-fold) AUROC
    if all_val_labels and all_val_scores:
        oof_auroc = roc_auc_score(all_val_labels, all_val_scores)
        agg["oof_auroc"] = round(float(oof_auroc), 4)
        print(f"\n  Out-of-fold AUROC: {oof_auroc:.4f}")

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4 - MPS model training")
    parser.add_argument("--phase4-input",  default="data/processed/temporal/phase4_input.csv")
    parser.add_argument("--manifest-dir",  default="data/manifests")
    parser.add_argument("--vst-input",     default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--out-dir",       default="experiments/mlflow")
    parser.add_argument("--epochs",        type=int,   default=80)
    parser.add_argument("--folds",         type=int,   default=5)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lambda-phys",   type=float, default=0.15)
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
    print("  Phase 4 — MPS Model Training")
    print("=" * 60)

    # Dataset
    dataset = CRCMetastasisDataset(
        phase4_path   = args.phase4_input,
        manifest_path = f"{args.manifest_dir}/cohort_labeled.csv",
        vst_path      = args.vst_input,
    )

    model_config = {
        "gnn_hidden":        64,
        "gnn_out_dim":       128,
        "transformer_d":     128,
        "transformer_heads": 4,
        "transformer_layers": 3,
        "transformer_out":   128,
        "tabular_out":       64,
        "hidden_dims":       [256, 128, 64],
        "dropout":           args.dropout,
        "lambda_phys":       args.lambda_phys,
        "lambda_calib":      0.05,
    }

    # MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(str(Path("experiments/mlflow")))
        mlflow.set_experiment("CRC_Metastasis_MPS")
        mlflow.start_run(run_name=f"phase4_cv{args.folds}fold")
        mlflow.log_params({**model_config, "epochs": args.epochs,
                           "lr": args.lr, "folds": args.folds})

    # Run cross-validation
    results = cross_validate(
        dataset, args.folds, args.epochs, args.lr,
        args.patience, out_dir, device, model_config
    )

    # Save results
    results_path = out_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({k: v for k, v in results.items()
                            if isinstance(v, float)})
        mlflow.end_run()

    print(f"\n  Phase 4 training complete ✓")
    print(f"  Best OOF AUROC: {results.get('oof_auroc', 'N/A')}")
    print(f"  Next: python src/models/evaluate.py   (Phase 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()

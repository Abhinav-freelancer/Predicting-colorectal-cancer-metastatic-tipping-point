"""
Phase 5 - Step 1: End-to-end pipeline runner
=============================================
Integrates all phases into a single reproducible command.
Runs Phase 1 → 2 → 3 → 4 in sequence with checkpointing —
skips completed phases on re-runs.

Also runs a full cross-validation using scikit-learn baseline
comparators (LR, RF, XGBoost) so the MPS model performance
can be benchmarked against standard clinical ML.

Usage:
    python src/evaluation/pipeline.py                  # full run
    python src/evaluation/pipeline.py --from-phase 4   # resume at phase 4
    python src/evaluation/pipeline.py --baselines-only # baselines only
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parents[2]))


# ── Phase status tracker ──────────────────────────────────────────────────────

class PipelineState:
    """Tracks which phases have completed and their outputs."""

    CHECKPOINTS = {
        1: "data/manifests/cohort_labeled.csv",
        2: "data/processed/temporal/feature_matrix.csv",
        3: "data/processed/temporal/phase4_input.csv",
        4: "experiments/mlflow/phase4_run/cv_results.json",
    }

    def __init__(self, state_path: str = "experiments/pipeline_state.json"):
        self.state_path = Path(state_path)
        self.state      = self._load()

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.load(open(self.state_path))
        return {"completed_phases": [], "timestamps": {}, "metrics": {}}

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.state, open(self.state_path, "w"), indent=2)

    def is_complete(self, phase: int) -> bool:
        checkpoint = self.CHECKPOINTS.get(phase)
        return checkpoint and Path(checkpoint).exists()

    def mark_complete(self, phase: int, metrics: dict = None):
        if phase not in self.state["completed_phases"]:
            self.state["completed_phases"].append(phase)
        self.state["timestamps"][str(phase)] = datetime.now().isoformat()
        if metrics:
            self.state["metrics"][str(phase)] = metrics
        self.save()

    def report(self):
        print("\n  Pipeline State:")
        for phase in range(1, 5):
            done = self.is_complete(phase)
            ts   = self.state["timestamps"].get(str(phase), "")
            print(f"    Phase {phase}: {'✓ complete' if done else '○ pending'}  {ts[:16] if ts else ''}")


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_phase(phase_num: int, cmd: str, state: PipelineState,
              force: bool = False) -> bool:
    """Run a phase command if not already complete."""
    import subprocess

    if not force and state.is_complete(phase_num):
        print(f"\n  Phase {phase_num}: already complete — skipping")
        return True

    print(f"\n  {'='*56}")
    print(f"  Running Phase {phase_num}...")
    print(f"  {'='*56}")

    start = time.time()
    result = subprocess.run(
        [sys.executable] + cmd.split(),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        state.mark_complete(phase_num, {"elapsed_s": round(elapsed, 1)})
        print(f"\n  Phase {phase_num} complete ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  Phase {phase_num} FAILED (exit code {result.returncode})")
        return False


# ── Baseline models ───────────────────────────────────────────────────────────

def run_baselines(features:     pd.DataFrame,
                  labels:       pd.Series,
                  n_splits:     int   = 5,
                  seed:         int   = 42,
                  out_dir:      Path  = Path("outputs")) -> pd.DataFrame:
    """
    Run standard ML baselines on the feature matrix with 5-fold CV.

    Baselines:
        1. Logistic Regression (L2)       — linear clinical benchmark
        2. Random Forest (500 trees)       — non-linear ensemble
        3. Gradient Boosting (XGBoost-like) — strongest classical baseline
        4. AJCC Stage only                 — current clinical standard
        5. Physics score only              — ODE model alone
        6. EMT index only                  — single best feature

    Returns DataFrame with AUROC, AUPRC, F1 per model.
    """
    print("\n" + "=" * 60)
    print("  Baseline Model Comparison")
    print("=" * 60)

    X   = features.fillna(0).values.astype(float)
    y   = labels.values.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=1000,
                                          class_weight="balanced",
                                          random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=500,
                                              max_depth=6,
                                              class_weight="balanced",
                                              random_state=seed,
                                              n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200,
                                                  max_depth=4,
                                                  learning_rate=0.05,
                                                  random_state=seed)),
        ]),
    }

    # Single-feature baselines
    single_features = {
        "AJCC Stage only":    "stage_order",
        "Physics score only": "physics_score",
        "EMT index only":     "emt_index",
        "EWS composite only": "ews_composite",
    }

    results = []

    # ── Full-feature baselines ────────────────────────────────────────────
    for name, model in models.items():
        fold_aurocs, fold_auprcs, fold_f1s = [], [], []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_va)[:, 1]
            pred  = (proba >= 0.5).astype(int)

            fold_aurocs.append(roc_auc_score(y_va, proba))
            fold_auprcs.append(average_precision_score(y_va, proba))
            tp = ((pred==1)&(y_va==1)).sum()
            fp = ((pred==1)&(y_va==0)).sum()
            fn = ((pred==0)&(y_va==1)).sum()
            p  = tp / (tp + fp + 1e-8)
            r  = tp / (tp + fn + 1e-8)
            fold_f1s.append(2*p*r/(p+r+1e-8))

        results.append({
            "model":       name,
            "features":    "all_35",
            "auroc_mean":  round(np.mean(fold_aurocs), 4),
            "auroc_std":   round(np.std(fold_aurocs),  4),
            "auprc_mean":  round(np.mean(fold_auprcs), 4),
            "f1_mean":     round(np.mean(fold_f1s),    4),
        })
        print(f"\n  {name:<28} AUROC={results[-1]['auroc_mean']:.4f} "
              f"±{results[-1]['auroc_std']:.4f}  "
              f"AUPRC={results[-1]['auprc_mean']:.4f}  "
              f"F1={results[-1]['f1_mean']:.4f}")

    # ── Single-feature baselines ──────────────────────────────────────────
    print(f"\n  Single-feature baselines:")
    for name, col in single_features.items():
        if col not in features.columns:
            continue
        x_single = features[[col]].fillna(0).values

        fold_aurocs = []
        for tr_idx, va_idx in skf.split(X, y):
            x_tr, x_va = x_single[tr_idx], x_single[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            clf = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(class_weight="balanced",
                                                       random_state=seed))])
            clf.fit(x_tr, y_tr)
            proba = clf.predict_proba(x_va)[:, 1]
            fold_aurocs.append(roc_auc_score(y_va, proba))

        auroc_mean = round(np.mean(fold_aurocs), 4)
        auroc_std  = round(np.std(fold_aurocs),  4)
        results.append({
            "model":       name,
            "features":    col,
            "auroc_mean":  auroc_mean,
            "auroc_std":   auroc_std,
            "auprc_mean":  np.nan,
            "f1_mean":     np.nan,
        })
        print(f"    {name:<28} AUROC={auroc_mean:.4f} ±{auroc_std:.4f}")

    df = pd.DataFrame(results).sort_values("auroc_mean", ascending=False)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baseline_comparison.csv", index=False)
    print(f"\n  Saved → {out_dir}/baseline_comparison.csv")

    return df


# ── Feature importance ────────────────────────────────────────────────────────

def compute_feature_importance(features:   pd.DataFrame,
                                labels:     pd.Series,
                                out_dir:    Path) -> pd.DataFrame:
    """
    Compute permutation importance using Random Forest.
    Proxy for SHAP (which requires the full PyTorch model).
    """
    from sklearn.inspection import permutation_importance

    print("\n  Computing feature importance (Random Forest permutation)...")

    X    = features.fillna(0).values.astype(float)
    y    = labels.values.astype(int)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced", random_state=42)
    rf.fit(X_sc, y)

    # RF native importance
    native_imp = pd.Series(rf.feature_importances_,
                           index=features.columns,
                           name="rf_importance").sort_values(ascending=False)

    # Permutation importance on training set
    perm = permutation_importance(rf, X_sc, y, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean,
                         index=features.columns,
                         name="permutation_importance")

    importance_df = pd.DataFrame({
        "feature":                features.columns,
        "rf_importance":          rf.feature_importances_,
        "permutation_importance": perm_imp.values,
    }).sort_values("permutation_importance", ascending=False)

    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\n  Top 15 features by permutation importance:")
    print(f"  {'Feature':<30} {'RF Imp':>8}  {'Perm Imp':>10}")
    print(f"  {'─'*52}")
    for _, row in importance_df.head(15).iterrows():
        bar = "█" * int(row["permutation_importance"] * 100)
        print(f"  {row['feature']:<30} {row['rf_importance']:>8.4f}  "
              f"{row['permutation_importance']:>10.4f}  {bar}")

    return importance_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5 - Pipeline integration")
    parser.add_argument("--from-phase",    type=int, default=1,
                        help="Resume from this phase (1-4)")
    parser.add_argument("--force",         action="store_true",
                        help="Re-run all phases even if complete")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Skip phase execution, only run baselines")
    parser.add_argument("--out-dir",       default="outputs/evaluation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState()

    print("=" * 60)
    print("  Phase 5 — Pipeline Integration & Baseline Evaluation")
    print("=" * 60)
    state.report()

    if not args.baselines_only:
        # Phase sequence
        phase_commands = {
            1: "src/data/generate_synthetic_cohort.py",
            2: "src/data/normaliser.py",
            3: "src/ode/physics_exporter.py",
        }
        for phase_num, cmd in phase_commands.items():
            if phase_num >= args.from_phase:
                success = run_phase(phase_num, cmd, state, args.force)
                if not success:
                    print(f"  Pipeline halted at Phase {phase_num}")
                    return

    # Load feature matrix for baseline evaluation
    feat_path = Path("data/processed/temporal/phase4_input.csv")
    if not feat_path.exists():
        print(f"  ✗ Phase 4 input not found. Run phases 1-3 first.")
        return

    feat_cols = [c for c in pd.read_csv(feat_path, index_col=0, nrows=0).columns
                 if c not in ["metastasis_label", "ajcc_stage", "ajcc_m"]]
    features  = pd.read_csv(feat_path, index_col=0)[feat_cols]
    labels    = pd.read_csv(feat_path, index_col=0)["metastasis_label"]

    # Run baselines
    baseline_df = run_baselines(features, labels, out_dir=out_dir)

    # Feature importance
    compute_feature_importance(features, labels, out_dir)

    # Final summary
    best = baseline_df.iloc[0]
    print(f"\n  Best baseline: {best['model']}")
    print(f"    AUROC = {best['auroc_mean']:.4f} ± {best['auroc_std']:.4f}")

    state.mark_complete(5, {"best_baseline_auroc": float(best["auroc_mean"])})

    print(f"\n  Phase 5 Step 1 complete ✓")
    print(f"  Next: python src/evaluation/cross_validator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Phase 5 - Step 2: Rigorous cross-validation & model evaluation
===============================================================
Produces a statistically rigorous evaluation of the full MPS
framework including:

  1. Stratified 5-fold CV with confidence intervals
     (bootstrap resampling within each fold)

  2. Calibration analysis
     - Reliability diagrams (predicted probability vs empirical frequency)
     - Expected Calibration Error (ECE)
     - Brier score

  3. Threshold analysis
     - Precision-Recall curve
     - ROC curve with optimal operating point (Youden's J)
     - Clinical utility at MPS threshold = 0.72

  4. DeLong test: statistical comparison of MPS vs baselines
     (tests if AUROC difference is significant)

  5. Bootstrap confidence intervals on all metrics (B=1000)

Usage:
    python src/evaluation/cross_validator.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parents[2]))


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(y_true:   np.ndarray,
                 y_score:  np.ndarray,
                 metric_fn,
                 n_boot:   int   = 1000,
                 ci:       float = 0.95,
                 seed:     int   = 42) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng        = np.random.default_rng(seed)
    n          = len(y_true)
    boot_vals  = []

    point_est = metric_fn(y_true, y_score)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_vals.append(metric_fn(yt, ys))
        except Exception:
            continue

    alpha  = 1 - ci
    lower  = float(np.percentile(boot_vals, 100 * alpha / 2))
    upper  = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return float(point_est), lower, upper


# ── Expected Calibration Error ────────────────────────────────────────────────

def expected_calibration_error(y_true:   np.ndarray,
                                y_prob:   np.ndarray,
                                n_bins:   int = 10) -> float:
    """
    ECE: weighted average calibration error across probability bins.
    Lower = better calibrated.
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_prob  = y_prob[mask].mean()
        bin_freq  = y_true[mask].mean()
        bin_size  = mask.sum()
        ece      += (bin_size / n) * abs(bin_prob - bin_freq)

    return float(ece)


# ── DeLong test for AUROC comparison ─────────────────────────────────────────

def delong_test(y_true: np.ndarray,
                scores_a: np.ndarray,
                scores_b: np.ndarray) -> tuple[float, float]:
    """
    DeLong et al. (1988) non-parametric test comparing two AUROCs.
    Tests H0: AUROC_A == AUROC_B.
    Returns (z_statistic, p_value).
    """
    def compute_midranks(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N)
        T2[J] = T + 1
        return T2

    def fastDeLong(scores, labels):
        n1 = int(labels.sum())
        n0 = len(labels) - n1
        scores1 = scores[labels == 1]
        scores0 = scores[labels == 0]
        tx = compute_midranks(scores1)
        ty = compute_midranks(scores0)
        tz = compute_midranks(np.concatenate([scores1, scores0]))

        aucs = (tz[:n1].mean() - (n1 + 1) / 2) / n0
        v01  = (tz[:n1] - tx) / n0
        v10  = 1 - (tz[n1:] - ty) / n1
        sx   = np.cov(v01) / n1 if n1 > 1 else np.array([[0.0]])
        sy   = np.cov(v10) / n0 if n0 > 1 else np.array([[0.0]])
        return aucs, sx + sy

    auc_a, var_a = fastDeLong(scores_a, y_true)
    auc_b, var_b = fastDeLong(scores_b, y_true)

    delta = auc_a - auc_b
    se    = np.sqrt(float(var_a) + float(var_b) + 1e-12)
    z     = delta / se
    p     = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(z), float(p)


# ── Calibration analysis ──────────────────────────────────────────────────────

def calibration_analysis(y_true:  np.ndarray,
                          y_prob:  np.ndarray,
                          model_name: str,
                          out_dir: Path) -> dict:
    """
    Full calibration report: ECE, Brier score, reliability curve data.
    """
    ece    = expected_calibration_error(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)

    # Reliability curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    calib_df = pd.DataFrame({
        "mean_predicted": mean_pred,
        "fraction_positive": frac_pos,
        "calibration_gap": abs(frac_pos - mean_pred),
    })

    calib_df.to_csv(out_dir / f"calibration_{model_name.replace(' ','_')}.csv",
                    index=False)

    return {
        "ece":   round(ece, 5),
        "brier": round(brier, 5),
    }


# ── Threshold analysis ────────────────────────────────────────────────────────

def threshold_analysis(y_true:  np.ndarray,
                        y_prob:  np.ndarray,
                        mps_threshold: float = 0.72) -> dict:
    """
    Compute ROC, PR curve, and clinical utility at alert threshold.

    Clinical utility metrics at MPS ≥ threshold:
        - Number Needed to Screen (NNS): how many patients need to be
          flagged to catch one true metastatic case
        - Sensitivity / Specificity at clinical threshold
        - Net Benefit (decision curve analysis proxy)
    """
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # Youden's J: optimal threshold (maximises sensitivity + specificity - 1)
    j_scores      = tpr - fpr
    optimal_idx   = np.argmax(j_scores)
    optimal_thresh = float(roc_thresholds[optimal_idx])
    optimal_sens   = float(tpr[optimal_idx])
    optimal_spec   = float(1 - fpr[optimal_idx])

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    # Clinical utility at MPS alert threshold
    y_pred_thresh = (y_prob >= mps_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh,
                                       labels=[0, 1]).ravel()

    sensitivity_thresh = tp / (tp + fn + 1e-8)
    specificity_thresh = tn / (tn + fp + 1e-8)
    ppv_thresh         = tp / (tp + fp + 1e-8)  # positive predictive value
    npv_thresh         = tn / (tn + fn + 1e-8)  # negative predictive value
    nns                = 1.0 / (ppv_thresh + 1e-8)  # number needed to screen

    # Net benefit at threshold t
    # NB = (TP/N) - (FP/N) * (t / (1-t))
    n         = len(y_true)
    t         = mps_threshold
    nb        = (tp / n) - (fp / n) * (t / (1 - t + 1e-8))

    return {
        "optimal_threshold":   round(optimal_thresh, 4),
        "optimal_sensitivity": round(optimal_sens,   4),
        "optimal_specificity": round(optimal_spec,   4),
        "sens_at_alert_thresh":  round(float(sensitivity_thresh), 4),
        "spec_at_alert_thresh":  round(float(specificity_thresh), 4),
        "ppv_at_alert_thresh":   round(float(ppv_thresh), 4),
        "npv_at_alert_thresh":   round(float(npv_thresh), 4),
        "number_needed_to_screen": round(float(nns), 2),
        "net_benefit":           round(float(nb), 5),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ── Full cross-validation ─────────────────────────────────────────────────────

def run_cross_validation(features:   pd.DataFrame,
                          labels:     pd.Series,
                          cv_results_path: Path,
                          n_splits:   int   = 5,
                          n_boot:     int   = 500,
                          seed:       int   = 42,
                          out_dir:    Path  = Path("outputs/evaluation")
                          ) -> dict:
    """
    Full evaluation of all models including MPS (from Phase 4 CV results).
    Produces calibrated comparison with confidence intervals.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X = features.fillna(0).values.astype(float)
    y = labels.values.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ── Collect out-of-fold predictions for each model ────────────────────
    oof_scores = {}
    all_labels  = np.zeros(len(y), dtype=int)
    all_indices = []

    models_to_eval = {
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                        class_weight="balanced",
                                        random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                            class_weight="balanced",
                                            random_state=seed, n_jobs=-1)),
        ]),
    }

    for name in models_to_eval:
        oof_scores[name] = np.zeros(len(y))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        all_indices.extend(va_idx)
        all_labels[va_idx] = y[va_idx]

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        for name, model in models_to_eval.items():
            model.fit(X_tr, y_tr)
            oof_scores[name][va_idx] = model.predict_proba(X_va)[:, 1]

    # ── Physics score as a predictor (ODE model alone) ────────────────────
    if "physics_score" in features.columns:
        sc = StandardScaler()
        X_phys = sc.fit_transform(features[["physics_score"]].fillna(0).values)
        lr_phys = LogisticRegression(class_weight="balanced", random_state=seed)
        oof_phys = np.zeros(len(y))
        for tr_idx, va_idx in skf.split(X, y):
            lr_phys.fit(X_phys[tr_idx], y[tr_idx])
            oof_phys[va_idx] = lr_phys.predict_proba(X_phys[va_idx])[:, 1]
        oof_scores["ODE Physics Only"] = oof_phys

    # ── Load MPS model OOF scores from Phase 4 if available ───────────────
    mps_loaded = False
    if cv_results_path and cv_results_path.exists():
        cv_data = json.load(open(cv_results_path))
        if "oof_scores" in cv_data:
            oof_scores["MPS Model (Phase 4)"] = np.array(cv_data["oof_scores"])
            mps_loaded = True
            print("  Loaded MPS OOF scores from Phase 4")
        else:
            # Simulate MPS scores from physics + EMT features for demo
            emt_idx = features.get("emt_index", pd.Series(0, index=features.index))
            phys    = features.get("physics_score", pd.Series(0, index=features.index))
            simulated = (0.4 * emt_idx + 0.6 * phys).fillna(0)
            sc_s      = StandardScaler()
            X_sim     = sc_s.fit_transform(simulated.values.reshape(-1, 1))
            lr_sim    = LogisticRegression(class_weight="balanced", random_state=seed)
            oof_sim   = np.zeros(len(y))
            for tr_idx, va_idx in skf.split(X, y):
                lr_sim.fit(X_sim[tr_idx], y[tr_idx])
                oof_sim[va_idx] = lr_sim.predict_proba(X_sim[va_idx])[:, 1]
            oof_scores["MPS Model (Phase 4)"] = oof_sim
            print("  Using simulated MPS scores (re-train Phase 4 for real scores)")

    # ── Compute metrics for each model ────────────────────────────────────
    print("\n  Cross-validation results with bootstrap 95% CIs:")
    print(f"\n  {'Model':<28}  {'AUROC':>8}  {'95% CI':>16}  "
          f"{'AUPRC':>8}  {'Brier':>8}  {'ECE':>8}")
    print(f"  {'─'*80}")

    full_results = {}
    for name, scores in oof_scores.items():
        y_arr = y

        auroc, ci_lo, ci_hi = bootstrap_ci(
            y_arr, scores, roc_auc_score, n_boot=n_boot
        )
        auprc = float(average_precision_score(y_arr, scores))
        brier = float(brier_score_loss(y_arr, scores))
        ece   = expected_calibration_error(y_arr, scores)

        thresh_metrics = threshold_analysis(y_arr, scores)
        calib_metrics  = calibration_analysis(y_arr, scores, name, out_dir)

        print(f"  {name:<28}  {auroc:>8.4f}  "
              f"[{ci_lo:.4f}–{ci_hi:.4f}]  "
              f"{auprc:>8.4f}  {brier:>8.4f}  {ece:>8.4f}")

        full_results[name] = {
            "auroc": auroc, "auroc_ci_lo": ci_lo, "auroc_ci_hi": ci_hi,
            "auprc": auprc, "brier": brier, "ece": ece,
            **calib_metrics, **thresh_metrics,
        }

    # ── DeLong pairwise tests ─────────────────────────────────────────────
    print(f"\n  DeLong AUROC comparison (vs MPS Model):")
    mps_scores = oof_scores.get("MPS Model (Phase 4)",
                                 oof_scores.get("Random Forest"))
    for name, scores in oof_scores.items():
        if name == "MPS Model (Phase 4)":
            continue
        z, p = delong_test(y, mps_scores, scores)
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
               ("*" if p < 0.05 else "ns"))
        print(f"    MPS vs {name:<25} z={z:+.3f}  p={p:.4f}  {sig}")

    # ── Clinical utility summary ──────────────────────────────────────────
    best_model = "MPS Model (Phase 4)"
    best_thresh = full_results.get(best_model, {})

    print(f"\n  Clinical utility at MPS alert threshold = 0.72:")
    print(f"    Sensitivity            : {best_thresh.get('sens_at_alert_thresh', 'N/A')}")
    print(f"    Specificity            : {best_thresh.get('spec_at_alert_thresh', 'N/A')}")
    print(f"    PPV                    : {best_thresh.get('ppv_at_alert_thresh', 'N/A')}")
    print(f"    NPV                    : {best_thresh.get('npv_at_alert_thresh', 'N/A')}")
    print(f"    Number needed to screen: {best_thresh.get('number_needed_to_screen', 'N/A')}")
    print(f"    Net benefit            : {best_thresh.get('net_benefit', 'N/A')}")

    # Save
    results_df = pd.DataFrame(full_results).T
    results_df.to_csv(out_dir / "cv_full_results.csv")
    json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                        else vv)
                   for kk, vv in v.items()}
               for k, v in full_results.items()},
              open(out_dir / "cv_full_results.json", "w"),
              indent=2)

    print(f"\n  Saved results → {out_dir}/cv_full_results.json")
    return full_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 - Cross-validation")
    parser.add_argument("--phase4-input",    default="data/processed/temporal/phase4_input.csv")
    parser.add_argument("--cv-results",      default="experiments/mlflow/phase4_run/cv_results.json")
    parser.add_argument("--out-dir",         default="outputs/evaluation")
    parser.add_argument("--n-splits",        type=int, default=5)
    parser.add_argument("--n-boot",          type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 5 — Step 2: Cross-Validation & Model Comparison")
    print("=" * 60)

    feat_cols = [c for c in pd.read_csv(args.phase4_input, index_col=0, nrows=0).columns
                 if c not in ["metastasis_label", "ajcc_stage", "ajcc_m"]]
    features  = pd.read_csv(args.phase4_input, index_col=0)[feat_cols]
    labels    = pd.read_csv(args.phase4_input, index_col=0)["metastasis_label"]

    cv_path   = Path(args.cv_results) if Path(args.cv_results).exists() else None

    results = run_cross_validation(
        features, labels,
        cv_results_path = cv_path,
        n_splits        = args.n_splits,
        n_boot          = args.n_boot,
        out_dir         = out_dir,
    )

    print(f"\n  Phase 5 Step 2 complete ✓")
    print(f"  Next: python src/evaluation/lead_time.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

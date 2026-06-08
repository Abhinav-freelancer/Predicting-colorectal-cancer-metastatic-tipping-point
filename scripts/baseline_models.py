"""
Baseline Models - Full 38-feature benchmark
============================================
Loads the complete Phase 4 feature matrix (35 tabular + 3 ODE physics),
trains 4 models with SMOTE + hyperparameter tuning, builds a stacking
ensemble, reports bootstrap confidence intervals, and plots ROC curves.

Also applies Platt scaling calibration and reports ECE.

Usage:
    python scripts/baseline_models.py
    python scripts/baseline_models.py --data data/processed/temporal/phase4_input.csv
"""

import sys, json, argparse, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import xgboost as xgb

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

FEATURE_GROUPS = {
    "EMT Signatures": [
        "epithelial", "mesenchymal", "tgfb_pathway", "wnt_pathway",
        "proliferation", "cytotoxic_t", "immune_suppression", "hypoxia",
        "emt_index", "immune_balance", "invasion_potential",
        "em_ratio",
    ],
    "Early Warning Signals": [
        "ews_var_epithelial", "ews_var_mesenchymal", "ews_skew_emt",
        "ews_kurt_emt", "ews_cv_mesenchymal", "ews_em_ratio", "ews_composite",
        "critical_slowing_down_index", "variance_skewness_ratio",
        "trajectory_position", "ews_ac1_slope", "ews_var_slope",
        "ews_skew_slope", "ews_anomaly", "spatial_synchrony",
    ],
    "Clinical / Staging": [
        "stage_order", "ajcc_t_encoded", "ajcc_n_encoded",
        "gender_encoded", "vital_status_encoded", "age_at_index",
        "days_to_last_fu_rna",
    ],
    "ODE Physics": [
        "fitted_T_ext", "attractor_proximity", "bifurcation_score",
        "physics_score", "in_tipping_zone", "n_attractors", "is_bistable",
        "epi_dist", "mes_dist", "current_state_encoded",
        "phase_portrait_angle",
        "cdh1_vim_ratio", "snai1_expression", "hif1a_expression",
        "tgfb1_expression", "zeb1_mir200b_ratio", "emt_tf_activity",
        "emt_switch_index", "tgf_hypoxia_synergy", "tgfb_snai1_axis",
        "vim_expression", "mir200b_expression",
        "epith_program", "em_balance", "epithelial_integrity",
        "tgf_hypoxia_loop",
        "snai1_zeb1_motif", "tgfb_emt_tf_axis", "cdh1_mir200b_motif",
    ],
    "Interaction Features": [
        "tgfb_x_bifurcation", "emt_x_tipping",
        "mesenchymal_dominance", "ews_composite_v2",
    ],
}

# Sub-groups for group-specific baseline comparisons
GROUP_FEATURE_LISTS = {
    "EMT": [
        "epithelial", "mesenchymal", "tgfb_pathway", "wnt_pathway",
        "proliferation", "cytotoxic_t", "immune_suppression", "hypoxia",
        "emt_index", "immune_balance", "invasion_potential",
        "em_ratio",
    ],
    "Physics": [
        "fitted_T_ext", "attractor_proximity", "bifurcation_score",
        "physics_score", "in_tipping_zone", "n_attractors", "is_bistable",
        "epi_dist", "mes_dist", "current_state_encoded",
        "phase_portrait_angle",
        "tgfb_x_bifurcation", "emt_x_tipping",
        "meso_epi_dist_ratio", "tipping_urgency",
        "cdh1_vim_ratio", "snai1_expression", "hif1a_expression",
        "tgfb1_expression", "zeb1_mir200b_ratio", "emt_tf_activity",
        "emt_switch_index", "tgf_hypoxia_synergy", "tgfb_snai1_axis",
        "vim_expression", "mir200b_expression",
        "epith_program", "em_balance", "epithelial_integrity",
        "tgf_hypoxia_loop",
        "snai1_zeb1_motif", "tgfb_emt_tf_axis", "cdh1_mir200b_motif",
    ],
    "EWS": [
        "ews_var_epithelial", "ews_var_mesenchymal", "ews_skew_emt",
        "ews_kurt_emt", "ews_cv_mesenchymal", "ews_em_ratio", "ews_composite",
        "ews_composite_v2",
        "critical_slowing_down_index", "variance_skewness_ratio",
        "trajectory_position", "ews_ac1_slope", "ews_var_slope",
        "ews_skew_slope", "ews_anomaly", "spatial_synchrony",
    ],
}


def build_physics_group_features(df):
    """Build physics group feature matrix including derived interactions."""
    wanted = [c for c in GROUP_FEATURE_LISTS["Physics"] if c in df.columns]
    out = df[wanted].copy()
    if "mes_dist" in out.columns and "epi_dist" in out.columns:
        out["meso_epi_dist_ratio"] = out["mes_dist"] / (out["epi_dist"] + 1e-6)
    if "bifurcation_score" in out.columns and "in_tipping_zone" in out.columns:
        out["tipping_urgency"] = out["bifurcation_score"] * out["in_tipping_zone"]
    return out


def build_ews_group_features(df):
    """Build EWS group feature matrix including derived interactions."""
    wanted = [c for c in GROUP_FEATURE_LISTS["EWS"] if c in df.columns]
    out = df[wanted].copy()
    return out


ALL_FEATURES = list(itertools.chain(*FEATURE_GROUPS.values()))
GROUP_BUILDERS = {
    "Physics": build_physics_group_features,
    "EWS": build_ews_group_features,
}


def load_data(path: str) -> tuple:
    df = pd.read_csv(path, index_col=0)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    available = [c for c in ALL_FEATURES if c not in df.columns]
    if available:
        print(f"  Missing features (will be excluded): {available}")
    X = df[feature_cols].values.astype(float)
    y = df["metastasis_label"].values.astype(int)
    print(f"  Loaded {len(df)} patients, {len(feature_cols)} features")
    n_m1 = y.sum()
    print(f"  M1: {n_m1} ({100*n_m1/len(y):.1f}%), M0: {len(y)-n_m1} ({100*(len(y)-n_m1)/len(y):.1f}%)")
    return X, y, feature_cols, df


def knn_impute(X: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(X)
    if not nan_mask.any():
        return X
    n_nan = nan_mask.sum() / X.size * 100
    print(f"  Imputing {n_nan:.1f}% missing values via KNN (k=5)...")
    return KNNImputer(n_neighbors=5).fit_transform(X)


def bootstrap_ci(y_true, y_pred, n_boot=2000, alpha=0.95):
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        scores.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (1 + alpha) / 2 * 100)
    return float(lower), float(upper), scores


def ece_score(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])
    ece_val = 0.0
    for i in range(n_bins):
        mask = idx == i
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece_val += mask.sum() * abs(bin_acc - bin_conf)
    return ece_val / len(y_true)


def train_within_fold(X_tr, y_tr, X_va, y_va, model_fn, model_kwargs):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    smote = SMOTE(random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_sc, y_tr)
    model = model_fn(**model_kwargs)
    model.fit(X_tr_res, y_tr_res)
    scores = model.predict_proba(X_va_sc)[:, 1]
    auroc = roc_auc_score(y_va, scores)
    auprc = average_precision_score(y_va, scores)
    return auroc, auprc, scores, model, scaler


def tune_lr(X, y, n_trials=30):
    if n_trials < 1 or not HAS_OPTUNA:
        return {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced", "max_iter": 5000, "random_state": 42}

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "saga" if penalty == "l1" else "lbfgs"
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            sm = SMOTE(random_state=42)
            X_r, y_r = sm.fit_resample(X_tr, y[tr])
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                      class_weight="balanced", max_iter=5000, random_state=42)
            clf.fit(X_r, y_r)
            scores.append(roc_auc_score(y[va], clf.predict_proba(X_va)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  LR best: C={study.best_params['C']:.4f}, penalty={study.best_params['penalty']}, AUROC={study.best_value:.4f}")
    solver = "saga" if study.best_params["penalty"] == "l1" else "lbfgs"
    return {**study.best_params, "solver": solver, "class_weight": "balanced", "max_iter": 5000, "random_state": 42}


def tune_rf(X, y, n_trials=20):
    if n_trials < 1 or not HAS_OPTUNA:
        best_score, best_params = 0, {}
        for n_est in [200, 400, 600]:
            for md in [3, 5, 7, 9]:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                for tr, va in skf.split(X, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[tr])
                    X_va = scaler.transform(X[va])
                    sm = SMOTE(random_state=42)
                    X_r, y_r = sm.fit_resample(X_tr, y[tr])
                    clf = RandomForestClassifier(n_estimators=n_est, max_depth=md,
                                                  class_weight="balanced", random_state=42, n_jobs=-1)
                    clf.fit(X_r, y_r)
                    scores.append(roc_auc_score(y[va], clf.predict_proba(X_va)[:, 1]))
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"n_estimators": n_est, "max_depth": md}
        print(f"  RF best: {best_params}, AUROC={best_score:.4f}")
        return {**best_params, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}

    def objective(trial):
        n_est = trial.suggest_int("n_estimators", 100, 800, step=50)
        md = trial.suggest_int("max_depth", 2, 12)
        mss = trial.suggest_int("min_samples_split", 2, 20)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            sm = SMOTE(random_state=42)
            X_r, y_r = sm.fit_resample(X_tr, y[tr])
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=md,
                                          min_samples_split=mss,
                                          class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_r, y_r)
            scores.append(roc_auc_score(y[va], clf.predict_proba(X_va)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  RF best: n_est={study.best_params['n_estimators']}, max_depth={study.best_params['max_depth']}, AUROC={study.best_value:.4f}")
    return {**study.best_params, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}


def tune_xgb(X, y, n_trials=40):
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    if n_trials < 1 or not HAS_OPTUNA:
        return {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
                "scale_pos_weight": scale_pos, "random_state": 42, "verbosity": 0, "use_label_encoder": False}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, scale_pos * 3),
            "random_state": 42, "verbosity": 0, "use_label_encoder": False,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            sm = SMOTE(random_state=42)
            X_r, y_r = sm.fit_resample(X_tr, y[tr])
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_r, y_r)
            scores.append(roc_auc_score(y[va], clf.predict_proba(X_va)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGB best: lr={study.best_params['learning_rate']:.4f}, depth={study.best_params['max_depth']}, scale_pos={study.best_params['scale_pos_weight']:.2f}, AUROC={study.best_value:.4f}")
    return {**study.best_params, "random_state": 42, "verbosity": 0, "use_label_encoder": False}


def tune_svm(X, y, n_trials=20):
    if n_trials < 1 or not HAS_OPTUNA:
        return {"C": 1.0, "gamma": "scale", "kernel": "rbf", "probability": True, "class_weight": "balanced", "random_state": 42}

    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            sm = SMOTE(random_state=42)
            X_r, y_r = sm.fit_resample(X_tr, y[tr])
            clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True,
                       class_weight="balanced", random_state=42)
            clf.fit(X_r, y_r)
            scores.append(roc_auc_score(y[va], clf.predict_proba(X_va)[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  SVM best: C={study.best_params['C']:.4f}, AUROC={study.best_value:.4f}")
    return {**study.best_params, "kernel": "rbf", "probability": True, "class_weight": "balanced", "random_state": 42}


def run_cv_model(X, y, model_fn, model_kwargs, name, n_splits=5):
    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  {'='*50}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aurocs, fold_auprcs = [], []
    all_labels, all_scores = [], []
    fold_models = []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        auroc, auprc, scores, model, scaler = train_within_fold(
            X[tr], y[tr], X[va], y[va], model_fn, model_kwargs
        )
        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)
        all_labels.extend(y[va])
        all_scores.extend(scores)
        fold_models.append((model, scaler))
        print(f"    Fold {fold+1}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    oof_auroc = roc_auc_score(all_labels, all_scores)
    oof_auprc = average_precision_score(all_labels, all_scores)
    lower, upper, boot_scores = bootstrap_ci(np.array(all_labels), np.array(all_scores))

    print(f"    Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    print(f"    OOF AUROC:  {oof_auroc:.4f}  (95% CI: {lower:.4f} - {upper:.4f})")
    print(f"    OOF AUPRC:  {oof_auprc:.4f}")

    return {
        "name": name,
        "fold_aurocs": [round(x, 4) for x in fold_aurocs],
        "mean_auroc": round(float(mean_auroc), 4),
        "std_auroc": round(float(std_auroc), 4),
        "oof_auroc": round(float(oof_auroc), 4),
        "oof_auroc_ci": (round(float(lower), 4), round(float(upper), 4)),
        "oof_auprc": round(float(oof_auprc), 4),
        "all_labels": all_labels,
        "all_scores": all_scores,
        "boot_scores": boot_scores,
        "models": fold_models,
    }


def run_stacking_ensemble(base_results, X, y, n_splits=5):
    print(f"\n  {'='*50}")
    print(f"  Stacking Ensemble (LR meta-learner)")
    print(f"  {'='*50}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_labels, all_scores = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        base_preds_tr, base_preds_va = [], []

        for result in base_results:
            model, scaler = result["models"][fold]
            X_tr_sc = scaler.transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            base_preds_tr.append(model.predict_proba(X_tr_sc)[:, 1].reshape(-1, 1))
            base_preds_va.append(model.predict_proba(X_va_sc)[:, 1].reshape(-1, 1))

        meta_X_tr = np.hstack(base_preds_tr)
        meta_X_va = np.hstack(base_preds_va)

        sm = SMOTE(random_state=42)
        meta_X_tr_res, y_tr_res = sm.fit_resample(meta_X_tr, y_tr)

        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
        meta.fit(meta_X_tr_res, y_tr_res)
        scores = meta.predict_proba(meta_X_va)[:, 1]
        auroc = roc_auc_score(y_va, scores)
        all_labels.extend(y_va)
        all_scores.extend(scores)
        print(f"    Fold {fold+1}: AUROC={auroc:.4f}")

    oof_auroc = roc_auc_score(all_labels, all_scores)
    oof_auprc = average_precision_score(all_labels, all_scores)
    lower, upper, boot_scores = bootstrap_ci(np.array(all_labels), np.array(all_scores))

    print(f"    OOF AUROC:  {oof_auroc:.4f}  (95% CI: {lower:.4f} - {upper:.4f})")
    print(f"    OOF AUPRC:  {oof_auprc:.4f}")

    return {
        "name": "Stacking Ensemble",
        "oof_auroc": round(float(oof_auroc), 4),
        "oof_auroc_ci": (round(float(lower), 4), round(float(upper), 4)),
        "oof_auprc": round(float(oof_auprc), 4),
        "all_labels": all_labels,
        "all_scores": all_scores,
        "boot_scores": boot_scores,
    }


def calibrate_best_model(best_model_results, best_params, X, y, best_name, out_dir):
    print(f"\n{'='*60}")
    print("  Phase 3 -- Calibration (Platt Scaling)")
    print(f"{'='*60}")
    print(f"\n  Calibrating: {best_name}")

    if "xgb" in best_name.lower():
        def base_fn(**kw):
            return xgb.XGBClassifier(**kw)
    elif "rf" in best_name.lower() or "random" in best_name.lower():
        def base_fn(**kw):
            return RandomForestClassifier(**kw)
    elif "svm" in best_name.lower() or "svc" in best_name.lower():
        def base_fn(**kw):
            return SVC(**kw)
    else:
        def base_fn(**kw):
            return LogisticRegression(**kw)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_labels_cal, all_scores_raw, all_scores_cal = [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        sm = SMOTE(random_state=42)
        X_r, y_r = sm.fit_resample(X_tr_sc, y_tr)

        base = base_fn(**{k: v for k, v in best_params.items() if k not in ["models"]})
        base.fit(X_r, y_r)
        raw_scores = base.predict_proba(X_va_sc)[:, 1]

        # Calibrate on original (non-resampled) training data
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(X_tr_sc, y_tr)
        cal_scores = cal.predict_proba(X_va_sc)[:, 1]

        all_labels_cal.extend(y_va)
        all_scores_raw.extend(raw_scores)
        all_scores_cal.extend(cal_scores)

    raw_auroc = roc_auc_score(all_labels_cal, all_scores_raw)
    cal_auroc = roc_auc_score(all_labels_cal, all_scores_cal)
    raw_ece = ece_score(np.array(all_labels_cal), np.array(all_scores_raw))
    cal_ece = ece_score(np.array(all_labels_cal), np.array(all_scores_cal))

    print(f"  Raw AUROC:      {raw_auroc:.4f}")
    print(f"  Calibrated AUROC: {cal_auroc:.4f}")
    print(f"  Raw ECE:        {raw_ece:.4f}")
    print(f"  Calibrated ECE:  {cal_ece:.4f}")

    # Calibration curve plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for scores, label, color in [
        (all_scores_raw, "Raw", "#1f77b4"),
        (all_scores_cal, "Platt-scaled", "#2ca02c"),
    ]:
        prob_true, prob_pred = calibration_curve(all_labels_cal, scores, n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curves - {best_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "calibration_curve.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Calibration plot saved -> {path}")

    return {
        "model": best_name,
        "raw_auroc": round(float(raw_auroc), 4),
        "calibrated_auroc": round(float(cal_auroc), 4),
        "raw_ece": round(float(raw_ece), 4),
        "calibrated_ece": round(float(cal_ece), 4),
    }


def run_group_baseline(X, y, feature_cols, group_name, group_feature_names, n_splits=5, n_trials=30, df=None):
    # Use group builder if available
    builder = GROUP_BUILDERS.get(group_name)
    if builder is not None and df is not None:
        built_df = builder(df)
        n_f = len(built_df.columns)
        X_g = built_df.values.astype(float)
    else:
        indices = [i for i, c in enumerate(feature_cols) if c in group_feature_names]
        if len(indices) < 2:
            print(f"  Skipping {group_name}: only {len(indices)} features available")
            return None
        X_g = X[:, indices]
        n_f = X_g.shape[1]
    print(f"\n  {'='*50}")
    print(f"  Group: {group_name} ({n_f} features)")
    print(f"  {'='*50}")
    lr_params = tune_lr(X_g, y, n_trials=min(n_trials, 20))
    xgb_params = tune_xgb(X_g, y, n_trials=min(n_trials, 30))
    lr_res = run_cv_model(X_g, y, LogisticRegression, lr_params,
                          f"LR [{group_name}]", n_splits)
    xgb_res = run_cv_model(X_g, y, xgb.XGBClassifier, xgb_params,
                           f"XGB [{group_name}]", n_splits)
    print(f"\n  --- {group_name} Summary ---")
    print(f"    LR  OOF AUROC={lr_res['oof_auroc']:.4f}  ({lr_res['oof_auroc_ci'][0]:.4f}-{lr_res['oof_auroc_ci'][1]:.4f})")
    print(f"    XGB OOF AUROC={xgb_res['oof_auroc']:.4f}  ({xgb_res['oof_auroc_ci'][0]:.4f}-{xgb_res['oof_auroc_ci'][1]:.4f})")
    return {"group": group_name, "n_features": n_f, "lr": lr_res, "xgb": xgb_res}


def plot_roc_curves(results_list, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, res in enumerate(results_list):
        labels = np.array(res["all_labels"])
        scores = np.array(res["all_scores"])
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{res['name']} (AUROC={res['oof_auroc']:.4f})")

        boot_scores = res.get("boot_scores", [])
        if boot_scores:
            boot_tprs = []
            rng = np.random.default_rng(42)
            n = len(labels)
            for _ in range(500):
                idx = rng.integers(0, n, size=n)
                if labels[idx].sum() == 0 or labels[idx].sum() == n:
                    continue
                f, t, _ = roc_curve(labels[idx], scores[idx])
                boot_tprs.append(np.interp(np.linspace(0, 1, 100), f, t))
            if boot_tprs:
                tprs_arr = np.array(boot_tprs)
                tpr_mean = np.mean(tprs_arr, axis=0)
                tpr_std = np.std(tprs_arr, axis=0)
                fpr_grid = np.linspace(0, 1, 100)
                ax.fill_between(fpr_grid,
                                np.clip(tpr_mean - 1.96 * tpr_std, 0, 1),
                                np.clip(tpr_mean + 1.96 * tpr_std, 0, 1),
                                color=colors[i % len(colors)], alpha=0.08)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUROC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Baseline Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "baseline_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  ROC plot saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline Models - Full 38-feature benchmark")
    parser.add_argument("--data", default="data/processed/temporal/phase4_input.csv")
    parser.add_argument("--out-dir", default="experiments/runs/phase4_run")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per model (0 = skip tuning)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Baseline Models - Full 38-Feature Benchmark")
    print("=" * 60)

    print("\n  Loading data...")
    X, y, feature_cols, df = load_data(args.data)
    X = knn_impute(X)

    n_trials = args.n_trials if HAS_OPTUNA else 0

    # -- Phase 1: Hyperparameter tuning -----------------------------------
    print(f"\n{'='*60}")
    print("  Phase 1 -- Hyperparameter Tuning (3-fold inner CV)")
    print(f"{'='*60}")
    lr_params = tune_lr(X, y, n_trials=min(n_trials, 30))
    rf_params = tune_rf(X, y, n_trials=min(n_trials, 20))
    xgb_params = tune_xgb(X, y, n_trials=min(n_trials, 40))
    svm_params = tune_svm(X, y, n_trials=min(n_trials, 20))

    # -- Phase 2: 5-fold CV with SMOTE ----------------------------------
    print(f"\n{'='*60}")
    print("  Phase 2 -- 5-Fold Cross-Validation with SMOTE")
    print(f"{'='*60}")

    results = {}
    lr_res = run_cv_model(X, y, LogisticRegression, lr_params, "Logistic Regression", args.n_folds)
    results["logistic_regression"] = lr_res

    rf_res = run_cv_model(X, y, RandomForestClassifier, rf_params, "Random Forest", args.n_folds)
    results["random_forest"] = rf_res

    xgb_res = run_cv_model(X, y, xgb.XGBClassifier, xgb_params, "XGBoost", args.n_folds)
    results["xgboost"] = xgb_res

    svm_res = run_cv_model(X, y, SVC, svm_params, "SVM (RBF)", args.n_folds)
    results["svm"] = svm_res

    # Stacking ensemble
    base_results = [lr_res, rf_res, xgb_res, svm_res]
    ensemble_res = run_stacking_ensemble(base_results, X, y, args.n_folds)
    results["stacking_ensemble"] = ensemble_res

    # ROC plot
    all_results_for_plot = [lr_res, rf_res, xgb_res, svm_res, ensemble_res]
    plot_roc_curves(all_results_for_plot, out_dir)

    # -- Phase 3: Calibration --------------------------------------------
    best_individual = max([k for k in results if k in ["logistic_regression","random_forest","xgboost","svm"]],
                          key=lambda k: results[k]["oof_auroc"])
    best_name = max(results, key=lambda k: results[k]["oof_auroc"])
    best_params_map = {
        "logistic_regression": lr_params,
        "random_forest": rf_params,
        "xgboost": xgb_params,
        "svm": svm_params,
    }
    calib_name = best_name if best_name in best_params_map else best_individual
    calib_res = calibrate_best_model(
        results[calib_name], best_params_map[calib_name],
        X, y, calib_name, out_dir
    )
    results["calibration"] = calib_res

    # -- Phase 4: Group-specific baselines --------------------------------
    print(f"\n{'='*60}")
    print("  Phase 4 -- Group-Specific Baselines (LR + XGBoost per group)")
    print(f"{'='*60}")
    group_results = {}
    for gname, gfeats in GROUP_FEATURE_LISTS.items():
        gr = run_group_baseline(X, y, feature_cols, gname, gfeats, args.n_folds, n_trials, df=df)
        if gr is not None:
            group_results[gname] = gr
    if group_results:
        print(f"\n  {'--- Group Comparison ---':^60}")
        print(f"  {'Group':<12} {'#Feat':>6} {'LR AUROC':>10} {'LR CI':>18} {'XGB AUROC':>10} {'XGB CI':>18}")
        print(f"  {'-'*74}")
        for gname, gr in group_results.items():
            lr_ci = f"({gr['lr']['oof_auroc_ci'][0]:.4f}-{gr['lr']['oof_auroc_ci'][1]:.4f})"
            xgb_ci = f"({gr['xgb']['oof_auroc_ci'][0]:.4f}-{gr['xgb']['oof_auroc_ci'][1]:.4f})"
            print(f"  {gname:<12} {gr['n_features']:>6} {gr['lr']['oof_auroc']:>10.4f} {lr_ci:>18} {gr['xgb']['oof_auroc']:>10.4f} {xgb_ci:>18}")

    # -- Summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"\n  {'Model':<22} {'Mean AUROC':>10} {'OOF AUROC':>10} {'95% CI':>20} {'AUPRC':>8}")
    print(f"  {'='*70}")
    for res in all_results_for_plot:
        ci = res.get("oof_auroc_ci", (0, 0))
        ci_str = f"({ci[0]:.4f}-{ci[1]:.4f})"
        mean_str = f"{res.get('mean_auroc', 0):.4f}+-{res.get('std_auroc', 0):.4f}" if "mean_auroc" in res else "     -"
        print(f"  {res['name']:<22} {mean_str:>10} {res['oof_auroc']:>10.4f} {ci_str:>20} {res['oof_auprc']:>8.4f}")

    print(f"\n  {'='*70}")
    print(f"  Calibration ({calib_res['model']}):")
    print(f"    Raw AUROC: {calib_res['raw_auroc']:.4f} -> Calibrated AUROC: {calib_res['calibrated_auroc']:.4f}")
    print(f"    Raw ECE:   {calib_res['raw_ece']:.4f} -> Calibrated ECE:   {calib_res['calibrated_ece']:.4f}")

    # -- Save results --------------------------------------------------------
    results_json = {}
    for name, res in results.items():
        if name == "calibration":
            results_json[name] = {
                "model": res["model"],
                "raw_auroc": res["raw_auroc"],
                "calibrated_auroc": res["calibrated_auroc"],
                "raw_ece": res["raw_ece"],
                "calibrated_ece": res["calibrated_ece"],
            }
        else:
            results_json[name] = {
                "oof_auroc": res["oof_auroc"],
                "oof_auprc": res["oof_auprc"],
            }
            if "oof_auroc_ci" in res:
                results_json[name]["oof_auroc_ci"] = list(res["oof_auroc_ci"])
            if "fold_aurocs" in res:
                results_json[name]["fold_aurocs"] = res["fold_aurocs"]
                results_json[name]["mean_auroc"] = res["mean_auroc"]
                results_json[name]["std_auroc"] = res["std_auroc"]

    results_json["features"] = feature_cols
    results_json["n_features"] = len(feature_cols)
    results_json["n_patients"] = len(y)
    results_json["n_metastatic"] = int(y.sum())

    results_path = out_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    # -- Save canonical comparison CSV for dashboard ---------------------------
    canonical_dir = Path("outputs/evaluation")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "baseline_comparison.csv"

    model_map = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "Gradient Boosting",
        "svm":                 "SVM (RBF)",
        "stacking_ensemble":   "Stacking Ensemble",
    }
    rows = []
    for key, label in model_map.items():
        r = results.get(key)
        if r:
            rows.append({
                "model": label,
                "auroc_mean": r.get("oof_auroc", 0),
                "auroc_std": r.get("std_auroc", 0),
            })
    # Add group baselines (best of LR/XGB per group)
    for gname, gr in group_results.items():
        best = gr["lr"] if gr["lr"]["oof_auroc"] >= gr["xgb"]["oof_auroc"] else gr["xgb"]
        rows.append({
            "model": f"{gname} group",
            "auroc_mean": best["oof_auroc"],
            "auroc_std": best.get("std_auroc", 0),
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(canonical_path, index=False)
    print(f"  Comparison CSV -> {canonical_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

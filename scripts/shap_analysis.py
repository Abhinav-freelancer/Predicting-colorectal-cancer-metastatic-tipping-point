"""
SHAP Analysis — Feature Importance for Best Baseline Model
============================================================
Trains XGBoost on full data + SMOTE, runs TreeExplainer,
prints top-10 features, and saves beeswarm summary plot.

Flags if ODE physics features are absent from the top 10,
which would indicate issues with the ODE fitting pipeline.

Usage:
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --data data/processed/temporal/phase4_input.csv
"""

import sys, argparse, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap

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
        "critical_slowing_down_index", "variance_skewness_ratio",
        "ews_composite_v2",
        "trajectory_position", "ews_ac1_slope", "ews_var_slope",
        "ews_skew_slope", "ews_anomaly", "spatial_synchrony",
    ],
}
ALL_FEATURES = list(itertools.chain(*FEATURE_GROUPS.values()))

PHYSICS_FEATURES = [
    "attractor_proximity", "bifurcation_score",
    "physics_score", "in_tipping_zone", "fitted_T_ext",
]


def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for CRC Metastasis Model")
    parser.add_argument("--data", default="data/processed/temporal/phase4_input.csv")
    parser.add_argument("--out-dir", default="outputs/shap")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials for XGBoost tuning")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SHAP Analysis — Feature Importance")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading data...")
    df = pd.read_csv(args.data, index_col=0)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  Missing features: {missing}")

    X = KNNImputer(n_neighbors=5).fit_transform(df[feature_cols].values.astype(float))
    y = df["metastasis_label"].values.astype(int)

    n_m1 = y.sum()
    print(f"  Patients: {len(y)}, Features: {len(feature_cols)}")
    print(f"  M1: {n_m1}/{len(y)} ({100*n_m1/len(y):.1f}%)")

    # ── Preprocess + SMOTE ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_sc, y)
    print(f"  After SMOTE: {len(y_res)} samples ({y_res.sum()} positive)")

    # ── Train XGBoost ────────────────────────────────────────────────────
    print("\n  Training XGBoost...")
    scale_pos = (y == 0).sum() / max((y == 1).sum(), 1)

    try:
        import optuna

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, scale_pos * 3),
                "random_state": 42, "verbosity": 0, "use_label_encoder": False,
            }
            from sklearn.model_selection import cross_val_score
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_res, y_res, cv=3,
                                     scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
        best_params = study.best_params
        print(f"  Best params: {best_params}")
        best_params["random_state"] = 42
        best_params["verbosity"] = 0
        best_params["use_label_encoder"] = False
    except ImportError:
        best_params = {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos,
            "random_state": 42, "verbosity": 0, "use_label_encoder": False,
        }
        print(f"  Optuna not available — using default params: {best_params}")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_res, y_res)

    # ── SHAP ────────────────────────────────────────────────────────────
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-10:][::-1]

    # ── Print top-10 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Top 10 Features by Mean |SHAP Value|")
    print(f"{'='*60}")
    print(f"  {'#':<3} {'Feature':<30} {'Mean |SHAP|':>12} {'Group'}")
    print(f"  {'─'*70}")
    for rank, idx in enumerate(top_idx):
        feat = feature_cols[idx]
        group = next((g for g, cols in FEATURE_GROUPS.items() if feat in cols), "Other")
        print(f"  {rank+1:<3} {feat:<30} {mean_shap[idx]:>12.6f}  {group}")

    # ── Check physics features ──────────────────────────────────────────
    physics_in_top = [f for f in PHYSICS_FEATURES
                      if f in feature_cols
                      and feature_cols.index(f) in top_idx]
    if not physics_in_top:
        print(f"\n  ⚠ WARNING: No ODE physics features in top 10!")
        print(f"     Possible causes:")
        print(f"      1. physics_exporter.py not run — all physics values are 0.5 defaults")
        print(f"      2. ODE fitting failed — check param_fitter.py output")
        print(f"      3. Physics features are genuinely uninformative on this data")
        print(f"     Action: verify {args.data} contains non-constant physics columns")
    else:
        print(f"\n  ✓ Physics features in top 10: {physics_in_top}")

    # ── Save top-10 CSV ─────────────────────────────────────────────────
    top_df = pd.DataFrame({
        "rank": range(1, 11),
        "feature": [feature_cols[i] for i in top_idx],
        "mean_abs_shap": mean_shap[top_idx],
        "group": [next((g for g, cols in FEATURE_GROUPS.items()
                         if feature_cols[i] in cols), "Other")
                  for i in top_idx],
    })
    top_df.to_csv(out_dir / "shap_top10.csv", index=False)
    print(f"\n  Top-10 CSV saved -> {out_dir / 'shap_top10.csv'}")

    # ── Beeswarm plot ───────────────────────────────────────────────────
    print("\n  Generating beeswarm plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sc, feature_names=feature_cols,
                       max_display=15, show=False)
    plt.tight_layout()
    path = out_dir / "shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm plot saved -> {path}")

    # ── Top-10 horizontal bar ───────────────────────────────────────────
    print("  Generating top-10 bar plot...")
    plt.figure(figsize=(8, 5))
    colors = []
    for feat in [feature_cols[i] for i in top_idx]:
        if feat in PHYSICS_FEATURES:
            colors.append("#d62728")
        elif feat in ["tgfb_x_bifurcation", "emt_x_tipping",
                        "mesenchymal_dominance", "ews_composite_v2"]:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")
    plt.barh(range(10), mean_shap[top_idx][::-1], color=colors[::-1])
    plt.yticks(range(10), [feature_cols[i] for i in top_idx][::-1])
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Top 10 Features — XGBoost SHAP Importance")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="EMT / EWS / Clinical"),
        Patch(facecolor="#d62728", label="ODE Physics"),
        Patch(facecolor="#ff7f0e", label="Interaction"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    path = out_dir / "shap_top10_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Top-10 bar plot saved -> {path}")

    # ── Summary feature importance (all features) ───────────────────────
    all_imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_shap,
        "group": [next((g for g, cols in FEATURE_GROUPS.items()
                         if f in cols), "Other")
                  for f in feature_cols],
    }).sort_values("mean_abs_shap", ascending=False)
    all_imp.to_csv(out_dir / "shap_all_features.csv", index=False)
    print(f"  Full importance CSV saved -> {out_dir / 'shap_all_features.csv'}")

    print(f"\n  {'─'*60}")
    print(f"  Group summary (mean |SHAP| per group, full model):")
    print(f"  {'─'*60}")
    for group in FEATURE_GROUPS:
        group_features = [f for f in FEATURE_GROUPS[group] if f in feature_cols]
        if group_features:
            group_mean = all_imp[all_imp["feature"].isin(group_features)]["mean_abs_shap"].mean()
            print(f"    {group:<25}: {group_mean:.6f}")

    # ── Per-group SHAP analysis ────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Per-group SHAP analysis (models trained on each group only)")
    print(f"  {'─'*60}")
    for gname, gfeats in GROUP_FEATURE_LISTS.items():
        gcols = [c for c in gfeats if c in df.columns]
        if len(gcols) < 2:
            continue
        X_g = KNNImputer(n_neighbors=5).fit_transform(df[gcols].values.astype(float))
        scaler_g = StandardScaler()
        X_g_sc = scaler_g.fit_transform(X_g)
        sm_g = SMOTE(random_state=42)
        X_g_res, y_g_res = sm_g.fit_resample(X_g_sc, y)
        model_g = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     scale_pos_weight=scale_pos, random_state=42,
                                     verbosity=0, use_label_encoder=False)
        model_g.fit(X_g_res, y_g_res)
        explainer_g = shap.TreeExplainer(model_g)
        shap_g = explainer_g.shap_values(X_g_sc)
        if isinstance(shap_g, list):
            shap_g = shap_g[1]
        mean_g = np.abs(shap_g).mean(axis=0)
        top3_idx = np.argsort(mean_g)[-3:][::-1]
        print(f"\n  {gname} ({len(gcols)} features):")
        for rk, idx in enumerate(top3_idx):
            print(f"    {rk+1}. {gcols[idx]:<30} {mean_g[idx]:.6f}")
        # Save per-group SHAP
        group_shap_df = pd.DataFrame({"feature": gcols, "mean_abs_shap": mean_g})\
            .sort_values("mean_abs_shap", ascending=False)
        group_shap_df.to_csv(out_dir / f"shap_group_{gname.lower()}.csv", index=False)
        print(f"       -> {out_dir / f'shap_group_{gname.lower()}.csv'}")

    print("=" * 60)


if __name__ == "__main__":
    main()

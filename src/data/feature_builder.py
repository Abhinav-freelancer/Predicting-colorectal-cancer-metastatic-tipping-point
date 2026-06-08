"""
Phase 2 - Step 4: Feature builder
===================================
Merges all Phase 2 outputs into a single patient feature matrix
(the temporal state vector ΔΨ) ready for Phase 3 (ODE model)
and Phase 4 (GNN + Transformer).

Output feature matrix columns:
  ── EMT signature scores (11 features)
      epithelial, mesenchymal, tgfb_pathway, wnt_pathway,
      proliferation, cytotoxic_t, immune_suppression, hypoxia,
      emt_index, immune_balance, invasion_potential

  ── Early warning signals (7 features)
      ews_var_epithelial, ews_var_mesenchymal, ews_skew_emt,
      ews_kurt_emt, ews_cv_mesenchymal, ews_em_ratio, ews_composite

  ── Clinical features (7 features)
      age_at_index, gender_encoded, ajcc_t_encoded, ajcc_n_encoded,
      vital_status_encoded, days_to_last_fu, stage_order

  ── Labels
      metastasis_label (0/1), ajcc_stage, ajcc_m

Usage:
    python src/data/feature_builder.py
"""

import argparse
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Ordinal encoders ──────────────────────────────────────────────────────────
STAGE_ORDER = {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}

T_ORDER  = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
N_ORDER  = {"N0": 0, "N1": 1, "N2": 2}


def encode_clinical(manifest: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical clinical variables as ordinal integers."""
    clin = manifest[["case_id", "metastasis_label",
                     "ajcc_stage", "ajcc_m", "ajcc_t", "ajcc_n",
                     "gender_rna", "vital_status_rna",
                     "age_at_index", "days_to_last_fu_rna"]].copy()

    clin = clin.set_index("case_id")

    clin["stage_order"]          = clin["ajcc_stage"].map(STAGE_ORDER).fillna(1)
    clin["ajcc_t_encoded"]       = clin["ajcc_t"].map(T_ORDER).fillna(1)
    clin["ajcc_n_encoded"]       = clin["ajcc_n"].map(N_ORDER).fillna(0)
    clin["gender_encoded"]       = (clin["gender_rna"].str.lower() == "male").astype(int)
    clin["vital_status_encoded"] = (clin["vital_status_rna"].str.lower() == "dead").astype(int)

    # Normalise continuous clinical features
    for col in ["age_at_index", "days_to_last_fu_rna"]:
        clin[col] = pd.to_numeric(clin[col], errors="coerce").fillna(clin[col].median())

    keep = ["metastasis_label", "ajcc_stage", "ajcc_m",
            "stage_order", "ajcc_t_encoded", "ajcc_n_encoded",
            "gender_encoded", "vital_status_encoded",
            "age_at_index", "days_to_last_fu_rna"]

    return clin[keep]


def build_feature_matrix(
    emt_scores:   pd.DataFrame,
    ews_scores:   pd.DataFrame,
    clinical:     pd.DataFrame,
    cnv_features: Optional[pd.DataFrame] = None,
    trajectory:   Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Inner-join all feature tables on patient ID.
    Patients without all required tables are dropped with a warning.
    """
    # Align indices
    emt_scores.index.name  = "patient_id"
    ews_scores.index.name  = "patient_id"
    clinical.index.name    = "patient_id"

    merged = emt_scores.join(ews_scores,  how="inner", rsuffix="_ews")
    merged = merged.join(clinical,        how="inner", rsuffix="_clin")

    # Merge trajectory projection features (cohort pseudo-time EWS)
    TRAJECTORY_FEATURES = [
        "trajectory_position", "ews_ac1_slope", "ews_var_slope",
        "ews_skew_slope", "ews_anomaly", "spatial_synchrony",
    ]
    if trajectory is not None and len(trajectory) > 0:
        traj = trajectory.set_index("patient_id")
        traj_avail = [c for c in TRAJECTORY_FEATURES if c in traj.columns]
        if traj_avail:
            merged = merged.join(traj[traj_avail], how="left")

    # Merge CNV features if provided (left join — CNV data may be partial)
    if cnv_features is not None and len(cnv_features) > 0:
        cnv_idx = cnv_features.set_index("case_id")
        cnv_idx.index.name = "patient_id"
        cnv_cols = [c for c in cnv_idx.columns]
        cnv_existing = [c for c in cnv_cols if c in merged.columns]
        if cnv_existing:
            cnv_idx = cnv_idx.drop(columns=cnv_existing)
        merged = merged.join(cnv_idx, how="left")
        # Fill missing CNV values with 0 (no CNV data = neutral)
        for c in cnv_idx.columns:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0.0)
        print(f"    CNV features : {len(cnv_cols)} added to feature matrix")

    # Rename duplicate columns
    dup_cols = [c for c in merged.columns if c.endswith("_ews") or c.endswith("_clin")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # ── Biologically motivated interaction features (Phase 2 only) ──────
    # Mesenchymal dominance: net shift along the E-M axis.
    if "mesenchymal" in merged.columns and "epithelial" in merged.columns:
        merged["mesenchymal_dominance"] = merged["mesenchymal"] - merged["epithelial"]

    # EWS composite v2: mean of the strongest variance-based signals.
    ews_vars = ["ews_var_epithelial", "ews_var_mesenchymal", "ews_em_ratio"]
    ews_present = [c for c in ews_vars if c in merged.columns]
    if ews_present:
        merged["ews_composite_v2"] = merged[ews_present].mean(axis=1)

    # Epithelial-mesenchymal ratio (inverse of ews_em_ratio for interpretability)
    if "epithelial" in merged.columns and "mesenchymal" in merged.columns:
        merged["em_ratio"] = merged["epithelial"] / (merged["mesenchymal"] + 1e-6)

    # Critical slowing down index: proxy for lag1_autocorr x rolling_variance
    ews_vm = "ews_var_mesenchymal"
    ews_ve = "ews_var_epithelial"
    if ews_vm in merged.columns and ews_ve in merged.columns:
        merged["critical_slowing_down_index"] = merged[ews_vm] * merged[ews_ve]

    # Variance-skewness ratio: asymmetry of fluctuations
    if "ews_var_mesenchymal" in merged.columns and "ews_skew_emt" in merged.columns:
        merged["variance_skewness_ratio"] = merged["ews_var_mesenchymal"] / (merged["ews_skew_emt"] + 1e-6)

    return merged


def impute_and_scale(df: pd.DataFrame,
                     label_cols: list,
                     scale: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate features from labels, impute NaN with column median,
    optionally standardise feature columns (zero mean, unit variance).

    Returns:
        features_scaled : pd.DataFrame  (model input)
        labels          : pd.DataFrame  (metastasis_label + metadata)
    """
    label_df   = df[label_cols].copy()
    feature_df = df.drop(columns=label_cols).copy()

    # Impute
    for col in feature_df.columns:
        med = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(med)

    if scale:
        scaler   = StandardScaler()
        scaled   = scaler.fit_transform(feature_df.values)
        feat_out = pd.DataFrame(scaled, index=feature_df.index,
                                columns=feature_df.columns)
        # Save scaler params
        scaler_df = pd.DataFrame({
            "feature": feature_df.columns,
            "mean":    scaler.mean_,
            "std":     scaler.scale_,
        })
    else:
        feat_out  = feature_df
        scaler_df = pd.DataFrame()

    return feat_out, label_df, scaler_df


def feature_summary(features: pd.DataFrame,
                    labels: pd.DataFrame,
                    out_dir: Path) -> None:
    """Print and save a summary of the final feature matrix."""
    n          = len(features)
    n_features = features.shape[1]
    n_meta     = (labels["metastasis_label"] == 1).sum()
    n_nonmeta  = (labels["metastasis_label"] == 0).sum()

    print(f"\n  ┌─ Final feature matrix ─────────────────────────────┐")
    print(f"  │  Patients          : {n:<6}                         │")
    print(f"  │  Features          : {n_features:<6}                         │")
    print(f"  │  Metastatic (1)    : {n_meta:<6} ({100*n_meta/n:.1f}%)              │")
    print(f"  │  Non-metastatic(0) : {n_nonmeta:<6} ({100*n_nonmeta/n:.1f}%)              │")
    print(f"  │  NaN remaining     : {int(features.isna().sum().sum()):<6}                         │")
    print(f"  └───────────────────────────────────────────────────┘")

    # Feature group breakdown
    print(f"\n  Feature groups:")
    groups = {
        "EMT scores":    [c for c in features.columns if not c.startswith("ews_")
                          and c not in ["stage_order","age_at_index","days_to_last_fu_rna",
                                        "gender_encoded","vital_status_encoded",
                                        "ajcc_t_encoded","ajcc_n_encoded"]],
        "EWS signals":   [c for c in features.columns if c.startswith("ews_")],
        "Clinical":      ["stage_order","age_at_index","days_to_last_fu_rna",
                          "gender_encoded","vital_status_encoded",
                          "ajcc_t_encoded","ajcc_n_encoded"],
    }
    for grp, cols in groups.items():
        found = [c for c in cols if c in features.columns]
        print(f"    {grp:<18} : {len(found)} features")

    # Correlation with label
    print(f"\n  Top 10 features by |correlation| with metastasis label:")
    numeric_labels = labels["metastasis_label"].astype(float)
    corrs = features.corrwith(numeric_labels).abs().dropna().sort_values(ascending=False)
    for feat, corr in corrs.head(10).items():
        bar = "█" * int(corr * 30)
        print(f"    {feat:<30} {corr:.4f}  {bar}")

    # Save correlation report
    corr_full = features.corrwith(numeric_labels).dropna().sort_values(ascending=False)
    corr_full.to_csv(out_dir / "feature_label_correlations.csv", header=["pearson_r"])


def main():
    parser = argparse.ArgumentParser(description="Phase 2 - Feature builder")
    parser.add_argument("--emt-scores",   default="data/processed/rna_seq/emt_scores.csv")
    parser.add_argument("--ews-scores",   default="data/processed/ews/patient_ews.csv")
    parser.add_argument("--cnv-features", default="data/processed/temporal/cnv_features.csv",
                        help="Optional CNV features CSV (from tcga_downloader_cnv.py)")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/temporal")
    parser.add_argument("--no-scale",     action="store_true",
                        help="Skip StandardScaler (saves raw feature values)")
    args = parser.parse_args()

    out_dir      = Path(args.out_dir)
    manifest_dir = Path(args.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 — Step 4: Feature Builder  [ΔΨ(t) state vector]")
    print("=" * 60)

    # Load inputs
    print("\n  Loading Phase 2 outputs...")
    emt_scores = pd.read_csv(args.emt_scores, index_col=0)
    ews_scores = pd.read_csv(args.ews_scores, index_col=0)
    manifest   = pd.read_csv(manifest_dir / "cohort_labeled.csv")

    print(f"    EMT scores : {emt_scores.shape}")
    print(f"    EWS scores : {ews_scores.shape}")
    print(f"    Manifest   : {manifest.shape}")

    # Load optional CNV features
    cnv_path = Path(args.cnv_features)
    cnv_features = None
    if cnv_path.exists():
        cnv_features = pd.read_csv(cnv_path)
        print(f"    CNV features : {cnv_features.shape}")
    else:
        print(f"    CNV features : not found at {cnv_path}, skipping")

    # Load cohort trajectory features (pseudo-time EWS projection)
    traj_path = Path("data/processed/ews/cohort_ews_trajectory.csv")
    trajectory = None
    if traj_path.exists():
        trajectory = pd.read_csv(traj_path)
        print(f"    Trajectory  : {trajectory.shape}")
    else:
        print(f"    Trajectory  : not found at {traj_path}, skipping")

    # Encode clinical variables
    clinical = encode_clinical(manifest)

    # Build merged matrix
    print("\n  Merging feature tables...")
    merged = build_feature_matrix(emt_scores, ews_scores, clinical, cnv_features, trajectory)
    print(f"  Merged shape: {merged.shape}")

    # Label columns to separate out
    LABEL_COLS = ["metastasis_label", "ajcc_stage", "ajcc_m"]
    label_cols_present = [c for c in LABEL_COLS if c in merged.columns]

    # Impute + scale
    features, labels, scaler_df = impute_and_scale(
        merged,
        label_cols=label_cols_present,
        scale=not args.no_scale,
    )

    # Summary
    feature_summary(features, labels, out_dir)

    # Save outputs
    features_path = out_dir / "feature_matrix.csv"
    labels_path   = out_dir / "labels.csv"
    scaler_path   = out_dir / "scaler_params.csv"
    full_path     = out_dir / "full_dataset.csv"

    features.to_csv(features_path)
    labels.to_csv(labels_path)
    if not scaler_df.empty:
        scaler_df.to_csv(scaler_path, index=False)

    # Convenient single file with features + labels
    full = features.join(labels)
    full.to_csv(full_path)

    print(f"\n  Saved:")
    print(f"    Feature matrix  → {features_path}  ← Phase 4 model input")
    print(f"    Labels          → {labels_path}")
    print(f"    Scaler params   → {scaler_path}")
    print(f"    Full dataset    → {full_path}")

    print(f"\n  Phase 2 complete ✓")
    print(f"  Next: python src/ode/emt_ode.py   (Phase 3 — dynamical model)")
    print("=" * 60)


if __name__ == "__main__":
    main()

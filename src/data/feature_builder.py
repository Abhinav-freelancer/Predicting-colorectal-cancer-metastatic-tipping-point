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
    clin = manifest[["submitter_id", "metastasis_label",
                     "ajcc_stage", "ajcc_m", "ajcc_t", "ajcc_n",
                     "gender", "vital_status",
                     "age_at_index", "days_to_last_fu"]].copy()

    clin = clin.set_index("submitter_id")

    clin["stage_order"]          = clin["ajcc_stage"].map(STAGE_ORDER).fillna(1)
    clin["ajcc_t_encoded"]       = clin["ajcc_t"].map(T_ORDER).fillna(1)
    clin["ajcc_n_encoded"]       = clin["ajcc_n"].map(N_ORDER).fillna(0)
    clin["gender_encoded"]       = (clin["gender"].str.lower() == "male").astype(int)
    clin["vital_status_encoded"] = (clin["vital_status"].str.lower() == "dead").astype(int)

    # Normalise continuous clinical features
    for col in ["age_at_index", "days_to_last_fu"]:
        clin[col] = pd.to_numeric(clin[col], errors="coerce").fillna(clin[col].median())

    keep = ["metastasis_label", "ajcc_stage", "ajcc_m",
            "stage_order", "ajcc_t_encoded", "ajcc_n_encoded",
            "gender_encoded", "vital_status_encoded",
            "age_at_index", "days_to_last_fu"]

    return clin[keep]


def build_feature_matrix(
    emt_scores:   pd.DataFrame,
    ews_scores:   pd.DataFrame,
    clinical:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join all three feature tables on patient ID.
    Patients without all three are dropped with a warning.
    """
    # Align indices
    emt_scores.index.name  = "patient_id"
    ews_scores.index.name  = "patient_id"
    clinical.index.name    = "patient_id"

    merged = emt_scores.join(ews_scores,  how="inner", rsuffix="_ews")
    merged = merged.join(clinical,        how="inner", rsuffix="_clin")

    # Rename duplicate columns
    dup_cols = [c for c in merged.columns if c.endswith("_ews") or c.endswith("_clin")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

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
                          and c not in ["stage_order","age_at_index","days_to_last_fu",
                                        "gender_encoded","vital_status_encoded",
                                        "ajcc_t_encoded","ajcc_n_encoded"]],
        "EWS signals":   [c for c in features.columns if c.startswith("ews_")],
        "Clinical":      ["stage_order","age_at_index","days_to_last_fu",
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

    # Encode clinical variables
    clinical = encode_clinical(manifest)

    # Build merged matrix
    print("\n  Merging feature tables...")
    merged = build_feature_matrix(emt_scores, ews_scores, clinical)
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

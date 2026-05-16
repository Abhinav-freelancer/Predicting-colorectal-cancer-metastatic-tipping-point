"""
Phase 1 — Step 4: Cohort validation and summary statistics.
Run after tcga_downloader.py to verify data quality before Phase 2.

Usage:
    python src/data/validate_cohort.py
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "case_id", "file_id", "file_name",
    "metastasis_label", "ajcc_m", "ajcc_stage",
    "vital_status", "gender",
]


def validate_manifest(manifest_path: Path) -> pd.DataFrame:
    print(f"\nLoading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # ── Column check ──────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠ Missing columns: {missing_cols}")
    else:
        print(f"  ✓ All required columns present")

    return df


def cohort_summary(df: pd.DataFrame) -> dict:
    """Print and return a summary dict of the cohort."""
    n = len(df)
    labeled = df[df["metastasis_label"] != -1]
    meta    = df[df["metastasis_label"] == 1]
    nonmet  = df[df["metastasis_label"] == 0]
    unkn    = df[df["metastasis_label"] == -1]

    print("\n" + "=" * 50)
    print("  COHORT SUMMARY")
    print("=" * 50)

    print(f"\n  Patients total        : {n}")
    print(f"  ├─ Metastatic  (M1)   : {len(meta)}  ({100*len(meta)/n:.1f}%)")
    print(f"  ├─ Localized   (M0)   : {len(nonmet)}  ({100*len(nonmet)/n:.1f}%)")
    print(f"  └─ Unknown            : {len(unkn)}  ({100*len(unkn)/n:.1f}%)")

    # AJCC stage distribution
    if "ajcc_stage" in df.columns:
        print("\n  AJCC stage distribution (labeled only):")
        stage_counts = labeled["ajcc_stage"].value_counts().head(10)
        for stage, cnt in stage_counts.items():
            bar = "█" * int(cnt / max(stage_counts) * 20)
            print(f"    {str(stage):<25} {cnt:>4}  {bar}")

    # Gender
    if "gender" in df.columns:
        print("\n  Gender:")
        for g, cnt in df["gender"].value_counts().items():
            print(f"    {str(g):<15} {cnt}")

    # Survival
    if "days_to_death" in df.columns:
        alive  = df["vital_status"].str.lower().eq("alive").sum()
        dead   = df["vital_status"].str.lower().eq("dead").sum()
        print(f"\n  Vital status:")
        print(f"    Alive : {alive}")
        print(f"    Dead  : {dead}")

    # Class balance warning
    if len(labeled) > 0:
        ratio = len(meta) / len(labeled)
        print(f"\n  Class imbalance ratio (meta/labeled): {ratio:.2f}")
        if ratio < 0.2 or ratio > 0.8:
            print("  ⚠ Imbalanced classes — plan to use:")
            print("      • Class-weighted loss in training")
            print("      • Stratified k-fold cross-validation")
            print("      • SMOTE or oversampling for minority class")

    summary = {
        "n_total":          n,
        "n_metastatic":     len(meta),
        "n_nonmetastatic":  len(nonmet),
        "n_unknown":        len(unkn),
        "n_labeled":        len(labeled),
        "imbalance_ratio":  round(len(meta) / max(len(labeled), 1), 3),
    }

    return summary


def check_expression_matrix(out_dir: Path) -> None:
    raw_path = out_dir / "raw_counts.parquet"
    if not raw_path.exists():
        print("\n  ℹ Expression matrix not yet built.")
        print("    Run build_expression_matrix.py after downloading files.")
        return

    matrix = pd.read_parquet(raw_path)
    print(f"\n  Expression matrix: {matrix.shape[0]:,} genes × {matrix.shape[1]} patients")
    print(f"  Zero fraction    : {(matrix == 0).mean().mean():.2%}")
    print(f"  Max count        : {int(matrix.max().max()):,}")


def save_phase1_report(summary: dict, out_dir: Path) -> None:
    report_path = out_dir / "phase1_report.json"
    report = {
        "phase":   "Phase 1 — Data Acquisition",
        "status":  "complete",
        "cohort":  summary,
        "next_steps": [
            "Run Phase 2 preprocessor: python src/data/preprocessor.py",
            "DESeq2 normalisation of RNA-seq counts",
            "EMT gene signature scoring",
            "Early warning signal computation",
        ]
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Phase 1 report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 cohort")
    parser.add_argument("--manifest-dir",    default="data/manifests")
    parser.add_argument("--expression-dir",  default="data/processed/rna_seq")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    expr_dir     = Path(args.expression_dir)

    labeled_path   = manifest_dir / "cohort_labeled.csv"
    full_path      = manifest_dir / "cohort_manifest.csv"

    path = labeled_path if labeled_path.exists() else full_path
    if not path.exists():
        print(f"No manifest found at {path}.")
        print("Run: python src/data/tcga_downloader.py")
        return

    df      = validate_manifest(path)
    summary = cohort_summary(df)
    check_expression_matrix(expr_dir)
    save_phase1_report(summary, manifest_dir)

    print("\n" + "=" * 50)
    print("  Phase 1 validation complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""
Phase 2 — Step 1: Preprocessor
================================
Bridges real TCGA data into the Phase 2 feature pipeline.

Ensures all intermediate files are computed from real expression data
(not synthetic fallback), then sequences the full pipeline:
    ews_computer.py -> feature_builder.py -> physics_exporter.py

Usage:
    python src/data/preprocessor.py
    python src/data/preprocessor.py --verbose
"""

import sys, json, argparse, logging, subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="  %(message)s")
logger = logging.getLogger("preprocessor")

PROJECT_ROOT = Path(__file__).parents[2]

RNA_SEQ_DIR    = PROJECT_ROOT / "data" / "processed" / "rna_seq"
MANIFEST_DIR   = PROJECT_ROOT / "data" / "manifests"
TEMPORAL_DIR   = PROJECT_ROOT / "data" / "processed" / "temporal"
EWS_DIR        = PROJECT_ROOT / "data" / "processed" / "ews"
OUT_DIR        = PROJECT_ROOT / "data" / "processed"

SYNTHETIC_FLAG = PROJECT_ROOT / "data" / "processed" / "synthetic_flag.txt"
PROVENANCE     = OUT_DIR / "provenance.json"


def remove_synthetic_flag():
    if SYNTHETIC_FLAG.exists():
        SYNTHETIC_FLAG.unlink()
        logger.warning("Removed synthetic_flag.txt — pipeline now uses real TCGA data")


def write_provenance(n_patients, n_features, m1_rate, m1_count, n_dropped):
    prov = {
        "data_source": "TCGA_real",
        "n_patients": n_patients,
        "n_features": n_features,
        "n_metastatic": m1_count,
        "m1_rate": round(float(m1_rate), 4),
        "patients_dropped_no_label": n_dropped,
        "run_timestamp": datetime.now().isoformat(),
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE, "w") as f:
        json.dump(prov, f, indent=2)
    logger.info(f"Provenance written -> {PROVENANCE}")


def step_validate(verbose: bool) -> tuple:
    logger.info("Step 1 — Validate inputs")
    raw_path = RNA_SEQ_DIR / "raw_counts.parquet"
    vst_path = RNA_SEQ_DIR / "vst_counts.csv.gz"
    emt_path = RNA_SEQ_DIR / "emt_scores.csv"
    manifest_path = MANIFEST_DIR / "cohort_labeled.csv"

    # Expression matrix must exist
    assert raw_path.exists(), f"Missing: {raw_path}"
    seq = pd.read_parquet(raw_path)
    n_genes, n_patients = seq.shape
    assert n_genes > 10000, f"Too few genes: {n_genes}"
    assert n_patients > 300, f"Too few patients: {n_patients}"
    # No all-zero patients
    zero_counts = (seq == 0).sum(axis=0)
    all_zero = (zero_counts == n_genes).sum()
    assert all_zero == 0, f"{all_zero} patients have all-zero counts (failed download)"
    if verbose:
        logger.info(f"  raw_counts: {n_genes:,} genes x {n_patients} patients")
        logger.info(f"  zero fraction: {(seq == 0).mean().mean():.2%}")
    expr_ids = set(seq.columns.tolist())
    del seq

    # VST file must exist
    assert vst_path.exists(), f"Missing: {vst_path}"
    vst_cols = pd.read_csv(vst_path, index_col=0, nrows=1).columns.tolist()
    assert len(vst_cols) == n_patients, f"VST patient count mismatch: {len(vst_cols)} vs {n_patients}"

    # EMT scores must exist and have all 11 expected columns
    expected_emt = [
        "epithelial", "mesenchymal", "tgfb_pathway", "wnt_pathway",
        "proliferation", "cytotoxic_t", "immune_suppression", "hypoxia",
        "emt_index", "immune_balance", "invasion_potential",
    ]
    emt = pd.read_csv(emt_path, index_col=0)
    missing_emt = [c for c in expected_emt if c not in emt.columns]
    assert emt.shape[1] >= len(expected_emt), f"EMT scores: {emt.shape[1]} cols, expected >= {len(expected_emt)}"
    if missing_emt:
        logger.warning(f"  EMT scores missing: {missing_emt} — will re-run")
        emt_rerun = True
    else:
        emt_rerun = False
    if verbose:
        logger.info(f"  EMT scores: {emt.shape}, all signature columns present")

    # Manifest with labels must exist
    assert manifest_path.exists(), f"Missing: {manifest_path}"
    manifest = pd.read_csv(manifest_path)
    assert "metastasis_label" in manifest.columns, "Missing metastasis_label in manifest"
    assert "case_id" in manifest.columns, "Missing case_id in manifest"

    n_labeled = len(manifest)
    n_m1 = (manifest.metastasis_label == 1).sum()
    n_m0 = (manifest.metastasis_label == 0).sum()
    m1_rate = n_m1 / max(n_labeled, 1)

    assert n_m1 > 0, "No M1 patients found — label alignment error"
    if verbose:
        overlap = len(set(manifest.case_id) & expr_ids)
        logger.info(f"  Manifest-expression overlap: {overlap}/{n_labeled}")
        logger.info(f"  M1 rate: {m1_rate:.1%} ({n_m1}/{n_labeled})")

    if m1_rate < 0.05 or m1_rate > 0.40:
        logger.warning(f"  Label balance outside expected range: {m1_rate:.1%}")
    if m1_rate < 0.01:
        logger.error("  M1 rate < 1% — likely label alignment bug, aborting")
        sys.exit(1)

    manifest_ids = set(manifest.case_id)
    n_dropped = n_patients - len(manifest_ids & expr_ids)

    return emt_rerun, n_patients, vst_path, emt_path, manifest_path


def step_run_ews(verbose: bool):
    logger.info("Step 4 — Run EWS computer on real expression data")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "ews_computer.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error(f"EWS computer failed with code {result.returncode}")
        sys.exit(1)


def step_run_feature_builder(verbose: bool):
    logger.info("Step 5 — Run feature builder (real EMT + EWS + clinical)")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "feature_builder.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error(f"Feature builder failed with code {result.returncode}")
        sys.exit(1)


def step_run_physics_exporter(verbose: bool):
    logger.info("Step 6 — Run physics exporter on real feature matrix")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "ode" / "physics_exporter.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error(f"Physics exporter failed with code {result.returncode}")
        sys.exit(1)


def step_assemble_final(verbose: bool) -> tuple:
    logger.info("Step 7 — Assemble final feature matrix")
    out_dir = TEMPORAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # physics_exporter writes phase4_input.csv to temporal dir already
    phase4_path = out_dir / "phase4_input.csv"
    if not phase4_path.exists():
        logger.warning("phase4_input.csv not found after physics_exporter")
        # Try to assemble from feature_matrix + physics_features
        feat = pd.read_csv(out_dir / "feature_matrix.csv", index_col=0)
        phys = pd.read_csv(out_dir / "physics_features.csv", index_col=0)
        common = feat.index.intersection(phys.index)
        full = feat.loc[common].join(phys.loc[common])
        manifest = pd.read_csv(MANIFEST_DIR / "cohort_labeled.csv")
        labels = manifest.set_index("case_id").reindex(full.index)
        full["metastasis_label"] = labels["metastasis_label"]
        full.to_csv(phase4_path)
        if verbose:
            logger.info(f"  Assembled phase4_input.csv: {full.shape}")

    df = pd.read_csv(phase4_path, index_col=0)
    n_patients = len(df)
    label_col = [c for c in df.columns if "label" in c.lower()]
    if label_col:
        m1_rate = df[label_col[0]].mean()
        m1_count = int(df[label_col[0]].sum())
    else:
        m1_rate = 0.0
        m1_count = 0

    n_feat = df.shape[1] - len(label_col) - 2
    n_missing = df.isna().sum().sum()

    logger.info(f"  Final matrix: {n_patients} patients, {n_feat} features")
    logger.info(f"  M1 rate: {m1_rate:.1%} ({m1_count} patients)")
    logger.info(f"  Missing values: {n_missing}")

    return n_patients, n_feat, m1_rate, m1_count


def print_summary(n_patients, n_features, m1_rate, m1_count, n_dropped):
    print()
    logger.info("=" * 50)
    logger.info("  Preprocessor complete")
    logger.info("=" * 50)
    logger.info(f"  Patients:     {n_patients}")
    if n_dropped > 0:
        logger.info(f"  (dropped {n_dropped} — missing M label)")
    logger.info(f"  Features:     {n_features}")
    logger.info(f"  M1 rate:      {m1_rate:.1%} ({m1_count} patients)")
    logger.info(f"  Data source:  TCGA_real")
    logger.info(f"  Provenance:   {PROVENANCE}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="TCGA preprocessor — bridge real data into Phase 2 pipeline")
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")
    args = parser.parse_args()

    verbose = args.verbose

    print()
    logger.info("=" * 60)
    logger.info("  Phase 2 — Step 1: Preprocessor [TCGA real data bridge]")
    logger.info("=" * 60)

    # Guard: remove synthetic flag if present
    remove_synthetic_flag()

    # Step 1: Validate inputs
    emt_rerun, n_patients, vst_path, emt_path, manifest_path = step_validate(verbose)

    # Step 2: Re-run normaliser if needed (VST already exists, skip)

    # Step 3: EMT scorer check
    logger.info("Step 3 — Re-run EMT scorer if needed")
    if emt_rerun:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "data" / "emt_scorer.py")],
            capture_output=False,
        )
        if result.returncode != 0:
            logger.error(f"EMT scorer failed with code {result.returncode}")
            sys.exit(1)
    else:
        logger.info("  EMT scores valid — skipping re-run")

    # Step 4: Run EWS computer
    step_run_ews(verbose)

    # Step 5: Run feature builder
    step_run_feature_builder(verbose)

    # Step 5b: Remove any stale ODE fitting results (proxy is more reliable)
    ode_path = PROJECT_ROOT / "data" / "processed" / "temporal" / "ode_patient_params.csv"
    if ode_path.exists():
        ode_path.unlink()
        logger.info("  Removed stale ODE fitting results — will use EMT proxy physics")

    # Step 6: Run physics exporter (uses EMT proxy — robust and deterministic)
    step_run_physics_exporter(verbose)

    # Step 7: Assemble final matrix
    n_patients_out, n_features, m1_rate, m1_count = step_assemble_final(verbose)

    # Step 8: Write provenance
    n_dropped = n_patients - n_patients_out
    write_provenance(n_patients_out, n_features, m1_rate, m1_count, max(n_dropped, 0))

    print_summary(n_patients_out, n_features, m1_rate, m1_count, max(n_dropped, 0))

    logger.info("")
    logger.info("  Next: run baseline_models.py --n-trials 30")


if __name__ == "__main__":
    main()

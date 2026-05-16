"""
Phase 1 — Synthetic data generator for local testing.

Generates a realistic TCGA-COAD-like cohort so you can develop and test
the full pipeline without internet access. Run this INSTEAD of
tcga_downloader.py when you don't yet have real data.

Produces:
  - data/manifests/cohort_manifest.csv
  - data/manifests/cohort_labeled.csv
  - data/processed/rna_seq/raw_counts.csv.gz    (patients × genes)
  - data/processed/rna_seq/log1p_counts.csv.gz

Biological realism:
  - EMT genes (CDH1, VIM, ZEB1, SNAI1...) have expression patterns
    that differ between metastatic and non-metastatic tumors
  - Stage IV patients have higher mesenchymal gene scores
  - Noise level matches typical bulk RNA-seq variance

Usage:
    python src/data/generate_synthetic_cohort.py
    python src/data/generate_synthetic_cohort.py --n-patients 200 --n-genes 2000
"""

import argparse
import json
import uuid
import numpy as np
import pandas as pd
from pathlib import Path


# ── EMT gene sets (used to inject biologically realistic signal) ──────────────
EMT_EPITHELIAL = ["CDH1", "EPCAM", "KRT18", "KRT8", "MUC1", "DSP", "OCLN"]
EMT_MESENCHYMAL = ["VIM", "FN1", "CDH2", "SNAI1", "SNAI2",
                   "ZEB1", "ZEB2", "TWIST1", "ACTA2", "MMP2"]
TGFB_PATHWAY   = ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4"]
KNOWN_GENES     = EMT_EPITHELIAL + EMT_MESENCHYMAL + TGFB_PATHWAY

# AJCC stage distributions (approximate, based on TCGA-COAD literature)
STAGE_PROBS     = [0.15, 0.30, 0.25, 0.30]   # I, II, III, IV
STAGES          = ["Stage I", "Stage II", "Stage III", "Stage IV"]
STAGE_TO_M      = {"Stage I": "M0", "Stage II": "M0", "Stage III": "M0", "Stage IV": "M1"}
STAGE_TO_LABEL  = {"Stage I": 0,  "Stage II": 0, "Stage III": 0, "Stage IV": 1}


def _tcga_barcode(patient_idx: int) -> str:
    return f"TCGA-AA-{3000 + patient_idx:04d}"


def _generate_gene_names(n_genes: int) -> list:
    """Mix known biological genes with synthetic ENSG-style IDs."""
    known = KNOWN_GENES[:]
    synthetic = [f"ENSG{i:011d}" for i in range(n_genes - len(known))]
    return known + synthetic


def generate_count_matrix(
    n_patients:  int,
    n_genes:     int,
    labels:      np.ndarray,
    rng:         np.random.Generator,
) -> pd.DataFrame:
    """
    Generate a realistic gene × patient raw count matrix.

    Signal injected:
      - Epithelial genes (CDH1, EPCAM...) are DOWN-regulated in metastatic (M1)
      - Mesenchymal genes (VIM, SNAI1...) are UP-regulated in metastatic (M1)
      - TGF-β pathway genes are moderately up in M1
      - Background genes follow negative-binomial distribution (bulk RNA-seq)
    """
    gene_names = _generate_gene_names(n_genes)
    matrix     = np.zeros((n_genes, n_patients), dtype=np.float64)

    # ── Background genes: negative-binomial approximation ─────────────────
    mean_expr  = rng.lognormal(mean=5, sigma=1.5, size=n_genes)   # gene means
    dispersion = 0.1 + rng.exponential(0.2, size=n_genes)         # overdispersion

    for g in range(n_genes):
        mu  = mean_expr[g]
        var = mu + dispersion[g] * mu ** 2
        p   = mu / var
        r   = mu ** 2 / (var - mu)
        r   = max(r, 0.1)
        matrix[g, :] = rng.negative_binomial(r, p, size=n_patients).astype(float)

    gene_idx = {g: i for i, g in enumerate(gene_names)}

    # ── Inject EMT signal ─────────────────────────────────────────────────
    meta_patients = np.where(labels == 1)[0]

    for gene in EMT_EPITHELIAL:
        g = gene_idx.get(gene)
        if g is None:
            continue
        # Down-regulate in metastatic: multiply by factor 0.15–0.4
        factors = rng.uniform(0.35, 0.85, size=len(meta_patients))
        matrix[g, meta_patients] = np.maximum(
            matrix[g, meta_patients] * factors, 0
        )

    for gene in EMT_MESENCHYMAL:
        g = gene_idx.get(gene)
        if g is None:
            continue
        # Up-regulate in metastatic: multiply by factor 3–8
        factors = rng.uniform(1.8, 3.5, size=len(meta_patients))
        matrix[g, meta_patients] = matrix[g, meta_patients] * factors

    for gene in TGFB_PATHWAY:
        g = gene_idx.get(gene)
        if g is None:
            continue
        # Moderate up-regulation: factor 1.5–3
        factors = rng.uniform(1.5, 3.0, size=len(meta_patients))
        matrix[g, meta_patients] = matrix[g, meta_patients] * factors

    # Round to integers and clip negatives
    matrix = np.round(np.clip(matrix, 0, None)).astype(int)

    df = pd.DataFrame(
        matrix,
        index=gene_names,
        columns=[_tcga_barcode(i) for i in range(n_patients)],
    )
    return df


def generate_clinical_manifest(
    n_patients: int,
    rng:        np.random.Generator,
) -> pd.DataFrame:
    """Generate a synthetic cohort manifest with realistic clinical variables."""
    rows = []
    for i in range(n_patients):
        stage  = rng.choice(STAGES, p=STAGE_PROBS)
        m_code = STAGE_TO_M[stage]
        label  = STAGE_TO_LABEL[stage]
        is_m1  = (label == 1)

        submitter   = _tcga_barcode(i)
        case_id     = str(uuid.uuid4())
        file_id     = str(uuid.uuid4())

        # Survival: metastatic patients have shorter OS on average
        if is_m1:
            days_to_death = int(rng.exponential(365))       # ~1 year median
        else:
            days_to_death = int(rng.exponential(365 * 4))   # ~4 year median

        vital_status = "Dead" if rng.random() < (0.65 if is_m1 else 0.25) else "Alive"
        days_death   = days_to_death if vital_status == "Dead" else None
        days_last_fu = days_to_death + int(rng.uniform(0, 200))

        rows.append({
            "case_id":          case_id,
            "submitter_id":     submitter,
            "file_id":          file_id,
            "file_name":        f"{submitter}.star_counts.tsv",
            "md5sum":           "",
            "file_size_bytes":  int(rng.uniform(1e5, 5e5)),
            "gender":           rng.choice(["male", "female"]),
            "race":             rng.choice(["white", "black or african american",
                                            "asian", "not reported"], p=[0.7, 0.15, 0.05, 0.1]),
            "age_at_index":     int(rng.normal(65, 11)),
            "tumor_stage":      stage,
            "ajcc_stage":       stage,
            "ajcc_m":           m_code,
            "ajcc_t":           rng.choice(["T1", "T2", "T3", "T4"]),
            "ajcc_n":           rng.choice(["N0", "N1", "N2"]),
            "vital_status":     vital_status,
            "days_to_death":    days_death,
            "days_to_last_fu":  days_last_fu,
            "days_to_dx":       int(rng.uniform(0, 30)),
            "days_to_recur":    int(rng.uniform(200, 800)) if not is_m1 else None,
            "metastasis_label": label,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TCGA-COAD cohort")
    parser.add_argument("--n-patients",   type=int, default=450,
                        help="Number of simulated patients (default 450)")
    parser.add_argument("--n-genes",      type=int, default=5000,
                        help="Number of genes in expression matrix (default 5000)")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/rna_seq")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    manifest_dir = Path(args.manifest_dir)
    out_dir      = Path(args.out_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Synthetic TCGA-COAD Cohort Generator")
    print("=" * 60)
    print(f"\n  Patients : {args.n_patients}")
    print(f"  Genes    : {args.n_genes}")
    print(f"  Seed     : {args.seed}")

    # ── Clinical manifest ─────────────────────────────────────────────────
    print("\n  Generating clinical manifest...")
    manifest = generate_clinical_manifest(args.n_patients, rng)
    labels   = manifest["metastasis_label"].values

    n_m1   = (labels == 1).sum()
    n_m0   = (labels == 0).sum()
    print(f"  Metastatic (M1)   : {n_m1}  ({100*n_m1/len(labels):.1f}%)")
    print(f"  Non-metastatic    : {n_m0}  ({100*n_m0/len(labels):.1f}%)")

    manifest.to_csv(manifest_dir / "cohort_manifest.csv", index=False)
    manifest.to_csv(manifest_dir / "cohort_labeled.csv",  index=False)
    print(f"  Saved → {manifest_dir}/cohort_manifest.csv")
    print(f"  Saved → {manifest_dir}/cohort_labeled.csv")

    # ── Expression matrix ─────────────────────────────────────────────────
    print(f"\n  Generating {args.n_genes:,} × {args.n_patients} expression matrix...")
    matrix = generate_count_matrix(args.n_genes, args.n_patients, labels, rng)

    print(f"  Matrix shape : {matrix.shape[0]:,} genes × {matrix.shape[1]} patients")
    print(f"  Zero fraction: {(matrix == 0).mean().mean():.2%}")
    print(f"  Max count    : {int(matrix.max().max()):,}")

    raw_path  = out_dir / "raw_counts.csv.gz"
    log_path  = out_dir / "log1p_counts.csv.gz"
    matrix.to_csv(str(raw_path).replace(".parquet",".csv.gz"), compression="gzip")
    log_matrix = np.log1p(matrix.astype(float))
    pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns).to_csv(str(log_path).replace(".parquet",".csv.gz"), compression="gzip")

    print(f"\n  Saved raw counts → {raw_path}")
    print(f"  Saved log1p      → {log_path}")

    # ── Spot-check EMT signal ─────────────────────────────────────────────
    print("\n  EMT signal check (mean expression by label):")
    print(f"  {'Gene':<12} {'Non-meta (M0)':>16} {'Metastatic (M1)':>16}  Direction")
    print(f"  {'─'*12} {'─'*16} {'─'*16}  {'─'*10}")

    m0_idx  = manifest[manifest["metastasis_label"] == 0]["submitter_id"].tolist()
    m1_idx  = manifest[manifest["metastasis_label"] == 1]["submitter_id"].tolist()

    # matrix columns are TCGA barcodes (submitter_ids)
    m0_cols = [c for c in matrix.columns if c in m0_idx]
    m1_cols = [c for c in matrix.columns if c in m1_idx]

    spot_genes = ["CDH1", "VIM", "ZEB1", "SNAI1", "TGFB1", "EPCAM"]
    for gene in spot_genes:
        if gene not in matrix.index:
            continue
        m0_mean = matrix.loc[gene, m0_cols].mean() if m0_cols else 0
        m1_mean = matrix.loc[gene, m1_cols].mean() if m1_cols else 0
        direction = "↑ in M1" if m1_mean > m0_mean else "↓ in M1"
        print(f"  {gene:<12} {m0_mean:>16.1f} {m1_mean:>16.1f}  {direction}")

    print("\n  Synthetic cohort ready. Run the full pipeline as if this were real data.")
    print("  When you have real data, replace manifests/ and rna_seq/ with actual downloads.")
    print("\n  Next: python src/data/validate_cohort.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

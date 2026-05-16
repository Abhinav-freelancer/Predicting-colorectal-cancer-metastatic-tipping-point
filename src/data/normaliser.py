"""
Phase 2 - Step 1: RNA-seq normalisation
========================================
Implements DESeq2-style median-of-ratios normalisation in pure NumPy/pandas
(no R required). Also produces TPM as an alternative.

DESeq2 median-of-ratios algorithm:
  1. Compute log of every count (skip zeros)
  2. Average log counts across patients per gene  → log geometric mean
  3. Subtract log geometric mean from each sample → log ratio
  4. Median of log ratios per sample              → size factor (log)
  5. Divide raw counts by exp(size factor)        → normalised counts

Usage:
    python src/data/normaliser.py
    python src/data/normaliser.py --method tpm --gene-lengths configs/gene_lengths.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_counts(data_dir: Path) -> pd.DataFrame:
    """Load raw count matrix (genes × patients)."""
    path = data_dir / "raw_counts.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Raw counts not found at {path}. Run Phase 1 first.")
    print(f"  Loading raw counts from {path}...")
    df = pd.read_csv(path, index_col=0, compression="gzip")
    print(f"  Shape: {df.shape[0]:,} genes × {df.shape[1]} patients")
    return df


# ── DESeq2 median-of-ratios ───────────────────────────────────────────────────

def deseq2_normalise(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    DESeq2 median-of-ratios normalisation.

    Returns:
        normalised  : pd.DataFrame  same shape as counts, float
        size_factors: pd.Series     one value per patient (sample)
    """
    print("\n  Running DESeq2 median-of-ratios normalisation...")

    mat = counts.values.astype(float)   # genes × patients

    # Step 1: log of counts — set zeros to NaN to exclude from geometric mean
    with np.errstate(divide="ignore"):
        log_counts = np.where(mat > 0, np.log(mat), np.nan)

    # Step 2: per-gene log geometric mean (mean of log across patients)
    log_geo_mean = np.nanmean(log_counts, axis=1)   # shape: (n_genes,)

    # Step 3: keep only genes where geometric mean is finite (not all-zero)
    valid = np.isfinite(log_geo_mean)
    log_counts_valid  = log_counts[valid, :]
    log_geo_mean_valid = log_geo_mean[valid]

    # Step 4: log ratio of each sample vs geometric mean
    log_ratios = log_counts_valid - log_geo_mean_valid[:, np.newaxis]  # (genes, patients)

    # Step 5: median log ratio per patient = log size factor
    log_size_factors = np.nanmedian(log_ratios, axis=0)   # shape: (n_patients,)
    size_factors      = np.exp(log_size_factors)

    # Step 6: divide raw counts by size factor
    normalised_mat = mat / size_factors[np.newaxis, :]

    size_factors_series = pd.Series(size_factors, index=counts.columns, name="size_factor")
    normalised_df       = pd.DataFrame(normalised_mat,
                                       index=counts.index,
                                       columns=counts.columns)

    print(f"  Size factors — min: {size_factors.min():.3f}  "
          f"max: {size_factors.max():.3f}  "
          f"median: {np.median(size_factors):.3f}")

    # Warn if any size factor is extreme (suggests outlier sample)
    extreme = ((size_factors < 0.1) | (size_factors > 10)).sum()
    if extreme:
        print(f"  ⚠ {extreme} samples have extreme size factors — check for outliers")

    return normalised_df, size_factors_series


# ── TPM normalisation (alternative) ──────────────────────────────────────────

def tpm_normalise(counts: pd.DataFrame,
                  gene_lengths: pd.Series) -> pd.DataFrame:
    """
    TPM (Transcripts Per Million) normalisation.
    Requires gene lengths in base pairs.

    gene_lengths: pd.Series indexed by gene_id, values = gene length (bp)
    """
    print("\n  Running TPM normalisation...")

    # Align gene lengths to count matrix rows
    common = counts.index.intersection(gene_lengths.index)
    counts_aligned  = counts.loc[common]
    lengths_aligned = gene_lengths.loc[common].values.astype(float)

    # RPK: reads per kilobase
    rpk = counts_aligned.values / (lengths_aligned[:, np.newaxis] / 1000.0)

    # TPM: RPK / sum(RPK) * 1e6
    rpk_sum = rpk.sum(axis=0, keepdims=True)
    tpm_mat = rpk / rpk_sum * 1e6

    tpm_df = pd.DataFrame(tpm_mat, index=common, columns=counts.columns)
    print(f"  TPM matrix: {tpm_df.shape[0]:,} genes × {tpm_df.shape[1]} patients")
    return tpm_df


# ── Post-normalisation QC ─────────────────────────────────────────────────────

def qc_report(raw: pd.DataFrame,
              normalised: pd.DataFrame,
              size_factors: pd.Series,
              out_dir: Path) -> pd.DataFrame:
    """
    Compute per-sample QC metrics and flag outliers.
    Returns a QC DataFrame and saves to CSV.
    """
    qc = pd.DataFrame(index=raw.columns)

    qc["total_raw_counts"]    = raw.sum(axis=0)
    qc["total_norm_counts"]   = normalised.sum(axis=0)
    qc["size_factor"]         = size_factors
    qc["n_genes_detected"]    = (raw > 0).sum(axis=0)
    qc["pct_zeros"]           = (raw == 0).mean(axis=0) * 100
    qc["median_raw_count"]    = raw.median(axis=0)

    # Flag samples where size factor deviates > 2 SD from mean
    sf_mean = qc["size_factor"].mean()
    sf_std  = qc["size_factor"].std()
    qc["outlier_flag"] = (
        (qc["size_factor"] < sf_mean - 2 * sf_std) |
        (qc["size_factor"] > sf_mean + 2 * sf_std)
    ).astype(int)

    n_outliers = qc["outlier_flag"].sum()
    print(f"\n  QC report:")
    print(f"    Samples        : {len(qc)}")
    print(f"    Outlier flags  : {n_outliers}")
    print(f"    Median genes detected per sample: {qc['n_genes_detected'].median():.0f}")
    print(f"    Median zero %  : {qc['pct_zeros'].median():.1f}%")

    qc.to_csv(out_dir / "sample_qc.csv")
    return qc


# ── Variance-stabilising transform ───────────────────────────────────────────

def vst(normalised: pd.DataFrame) -> pd.DataFrame:
    """
    Variance-stabilising transform: log2(normalised + 1).
    Compresses dynamic range for downstream ML features.
    Used as input to the GNN and transformer in Phase 4.
    """
    return np.log2(normalised + 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 - RNA-seq normalisation")
    parser.add_argument("--data-dir",  default="data/processed/rna_seq")
    parser.add_argument("--out-dir",   default="data/processed/rna_seq")
    parser.add_argument("--method",    default="deseq2", choices=["deseq2", "tpm"])
    parser.add_argument("--gene-lengths", default=None,
                        help="CSV with columns gene_id, length_bp (required for TPM)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 — Step 1: RNA-seq Normalisation")
    print("=" * 60)

    raw = load_counts(data_dir)

    if args.method == "deseq2":
        normalised, size_factors = deseq2_normalise(raw)
        method_tag = "deseq2"

    elif args.method == "tpm":
        if not args.gene_lengths:
            raise ValueError("--gene-lengths required for TPM normalisation")
        gl = pd.read_csv(args.gene_lengths, index_col="gene_id")["length_bp"]
        normalised = tpm_normalise(raw, gl)
        size_factors = pd.Series(1.0, index=raw.columns)   # placeholder
        method_tag = "tpm"

    # Variance-stabilising transform for ML input
    vst_mat = vst(normalised)

    # Save outputs
    norm_path = out_dir / f"normalised_{method_tag}.csv.gz"
    vst_path  = out_dir / "vst_counts.csv.gz"
    sf_path   = out_dir / "size_factors.csv"

    normalised.to_csv(norm_path, compression="gzip")
    vst_mat.to_csv(vst_path,    compression="gzip")
    size_factors.to_csv(sf_path)

    print(f"\n  Saved:")
    print(f"    Normalised counts : {norm_path}")
    print(f"    VST counts        : {vst_path}  ← used in Phase 4 model")
    print(f"    Size factors      : {sf_path}")

    # QC report
    qc_report(raw, normalised, size_factors, out_dir)

    print(f"\n  Next: python src/data/emt_scorer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

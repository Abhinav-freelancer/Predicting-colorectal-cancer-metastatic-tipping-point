"""
Phase 1 — Step 3: Build expression matrix from downloaded count files.

After downloading RNA-seq files with tcga_downloader.py, this script:
  1. Reads all HTSeq / STAR count TSV files
  2. Merges them into a single gene × patient matrix
  3. Filters low-expression genes
  4. Saves as Parquet for fast downstream loading

Usage:
    python src/data/build_expression_matrix.py
    python src/data/build_expression_matrix.py --data-dir data/raw/tcga_coad/rnaseq
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# Columns emitted by STAR count files that are NOT genes
STAR_SPECIAL_ROWS = {
    "N_unmapped", "N_multimapping",
    "N_noFeature", "N_ambiguous",
}

# Minimum mean count across patients to keep a gene
MIN_MEAN_COUNT = 1.0

# Minimum fraction of patients with non-zero expression
MIN_NONZERO_FRAC = 0.1


def parse_star_counts(filepath: Path) -> pd.Series:
    """
    Parse a single STAR counts TSV. Returns a Series indexed by gene_id.
    STAR files have 4 columns: gene_id, unstranded, fwd_stranded, rev_stranded.
    We use unstranded (column index 1) as per TCGA convention.
    """
    df = pd.read_csv(filepath, sep="\t", comment="#", header=0)

    # Column names vary slightly; normalise
    df.columns = [c.strip() for c in df.columns]

    # STAR output: first col = gene_id/feature, second = unstranded count
    gene_col  = df.columns[0]
    count_col = df.columns[1]   # unstranded

    df = df[[gene_col, count_col]].copy()
    df.columns = ["gene_id", "count"]

    # Drop special summary rows
    df = df[~df["gene_id"].isin(STAR_SPECIAL_ROWS)]
    df = df[df["gene_id"].notna()]
    df = df.set_index("gene_id")["count"]

    return df


def extract_case_id(filepath: Path, manifest: pd.DataFrame) -> str:
    """
    Resolve a file path to a TCGA case_id via the manifest.
    Falls back to the filename stem if not found.
    """
    fname = filepath.name
    if not manifest.empty:
        row = manifest[manifest["file_name"] == fname]
        if not row.empty:
            return row.iloc[0]["case_id"]
    # Fallback: use TCGA barcode from filename (TCGA-XX-XXXX)
    m = re.search(r"(TCGA-\w{2}-\w{4})", fname)
    return m.group(1) if m else filepath.stem


def build_matrix(data_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all count files into a genes × patients DataFrame.
    """
    tsv_files = sorted(data_dir.glob("*.tsv")) + sorted(data_dir.glob("*.txt"))

    if not tsv_files:
        raise FileNotFoundError(
            f"No TSV/TXT count files found in {data_dir}.\n"
            "Run tcga_downloader.py --download first."
        )

    print(f"  Found {len(tsv_files)} count files. Parsing...")

    series_dict = {}
    failed      = []

    for i, fp in enumerate(tsv_files):
        cid = extract_case_id(fp, manifest)
        try:
            s = parse_star_counts(fp)
            series_dict[cid] = s
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tsv_files)}] parsed")
        except Exception as e:
            print(f"  ⚠ Failed {fp.name}: {e}")
            failed.append(fp.name)

    if not series_dict:
        raise RuntimeError("No files parsed successfully.")

    print(f"\n  Parsed {len(series_dict)} files  |  failed {len(failed)}")

    matrix = pd.DataFrame(series_dict)   # genes × patients
    matrix = matrix.fillna(0).astype(int)

    print(f"  Raw matrix shape: {matrix.shape[0]} genes × {matrix.shape[1]} patients")

    return matrix


def filter_genes(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Remove genes with very low expression (likely noise).
    Keeps genes that pass BOTH thresholds:
        - mean raw count ≥ MIN_MEAN_COUNT
        - fraction of patients with count > 0 ≥ MIN_NONZERO_FRAC
    """
    mean_expr      = matrix.mean(axis=1)
    nonzero_frac   = (matrix > 0).mean(axis=1)

    keep = (mean_expr >= MIN_MEAN_COUNT) & (nonzero_frac >= MIN_NONZERO_FRAC)
    filtered = matrix.loc[keep]

    print(f"\n  Gene filtering:")
    print(f"    Before : {matrix.shape[0]:,} genes")
    print(f"    After  : {filtered.shape[0]:,} genes  "
          f"(removed {matrix.shape[0]-filtered.shape[0]:,})")

    return filtered


def log_normalise(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    log1p normalisation: log(count + 1).
    Reduces dynamic range and makes distributions more Gaussian.
    Full DESeq2 normalisation is done in Phase 2 preprocessor.
    """
    return np.log1p(matrix.astype(float))


def main():
    parser = argparse.ArgumentParser(description="Build TCGA-COAD expression matrix")
    parser.add_argument("--data-dir",     default="data/raw/tcga_coad/rnaseq")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/rna_seq")
    parser.add_argument("--no-filter",    action="store_true",
                        help="Skip low-expression gene filtering")
    args = parser.parse_args()

    data_dir     = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Building expression matrix")
    print("=" * 60)

    # Load manifest for case_id mapping
    manifest_path = manifest_dir / "tcga_rnaseq_files.csv"
    manifest      = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    # Build raw count matrix
    matrix = build_matrix(data_dir, manifest)

    # Filter genes
    if not args.no_filter:
        matrix = filter_genes(matrix)

    # Save raw counts (integer)
    raw_path = out_dir / "raw_counts.parquet"
    matrix.to_parquet(raw_path)
    print(f"\n  Saved raw counts → {raw_path}")

    # log1p version for quick exploration
    lognorm = log_normalise(matrix)
    log_path = out_dir / "log1p_counts.parquet"
    lognorm.to_parquet(log_path)
    print(f"  Saved log1p      → {log_path}")

    # Summary statistics
    summary = pd.DataFrame({
        "gene_id":     matrix.index,
        "mean_count":  matrix.mean(axis=1).values,
        "std_count":   matrix.std(axis=1).values,
        "nonzero_frac":(matrix > 0).mean(axis=1).values,
    })
    summary.to_csv(out_dir / "gene_summary.csv", index=False)

    print(f"\n  Final matrix: {matrix.shape[0]:,} genes × {matrix.shape[1]} patients")
    print(f"  Memory usage: {matrix.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("\n  Next: python src/data/tcga_downloader.py to add clinical merge,")
    print("        then Phase 2 preprocessor for DESeq2 normalisation.")
    print("=" * 60)


if __name__ == "__main__":
    main()

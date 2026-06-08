"""
GSE39582 Downloader + Feature Builder
=======================================
Downloads the GSE39582 CRC cohort from GEO, rebuilds the same
feature set as TCGA (EMT scores, EWS, physics proxy), and
prepares it for external validation of the trained MPS model.

GSE39582: 566 CRC patients with Affymetrix GPL570 expression
and M-stage annotation (metastasis vs non-metastatic).

Usage:
    python src/data/geo_downloader.py          # download + build features
    python src/data/geo_downloader.py --validate  # run inference with trained model
"""

import sys, os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.emt_scorer   import score_all_patients, compute_derived_scores, SIGNATURES
from src.data.ews_computer import compute_cross_sectional_ews


GEO_DIR = "data/raw/geo"
OUT_DIR = "data/processed/geo"


def download_gse39582(destdir: str = GEO_DIR) -> tuple:
    """
    Download GSE39582 via GEOparse.
    Returns (expression_matrix, metadata).

    Expression matrix: genes x samples (raw Affymetrix intensities)
    Metadata: sample_id -> AJCC M-stage, other clinical vars
    """
    try:
        import GEOparse
    except ImportError:
        print("  GEOparse not installed. Run: pip install GEOparse")
        sys.exit(1)

    dest = Path(destdir)
    dest.mkdir(parents=True, exist_ok=True)

    print("  Downloading GSE39582 from GEO...")
    gse = GEOparse.get_GEO("GSE39582", destdir=str(dest))

    # Extract expression matrix (first platform)
    platform = list(gse.gpls.values())[0]
    probe_ids = list(platform.table["ID"])
    title_key = "title" if "title" in platform.metadata else "platform_title"
    print(f"  Platform: {platform.metadata[title_key][0]}")

    # Build expression matrix: probes x samples
    sample_ids = sorted(gse.gsms.keys())
    expr_dict = {}
    for sid in sample_ids:
        gsm = gse.gsms[sid]
        table = gsm.table.set_index("ID_REF")
        expr_dict[sid] = table["VALUE"].astype(float)
    expr_df = pd.DataFrame(expr_dict)
    print(f"  Expression: {expr_df.shape[0]} probes x {expr_df.shape[1]} samples")

    # Extract M-stage from metadata
    meta_data = []
    for sid in sample_ids:
        gsm = gse.gsms[sid]
        chars = gsm.metadata.get("characteristics_ch1", [])
        meta_row = {"sample_id": sid}
        for c in chars:
            if ":" in c:
                k, v = c.split(":", 1)
                meta_row[k.strip().lower()] = v.strip()
        meta_data.append(meta_row)
    meta_df = pd.DataFrame(meta_data).set_index("sample_id")

    # Determine metastasis status
    label_cols = [c for c in meta_df.columns
                  if any(x in c for x in ["stage", "m stage", "ajcc", "tnm", "metasta"])]
    print(f"  Phenotype columns with stage info: {label_cols}")

    return expr_df, meta_df, platform


def probeset_to_gene(expr: pd.DataFrame, platform,
                     gene_symbol_col: str = "Gene Symbol") -> pd.DataFrame:
    """
    Map probesets to genes using platform annotation.
    Strategy:
        1. For probes that map to a single gene symbol, use that
        2. For probes mapping to multiple / no symbol, flag and drop
        3. If multiple probes map to the same gene, keep the one
           with highest mean expression (max-mean summarisation)
    """
    gpl_table = platform.table.copy()
    if gene_symbol_col not in gpl_table.columns:
        print(f"  Column '{gene_symbol_col}' not found in platform. "
              f"Available: {list(gpl_table.columns)}")
        return expr

    id_col = gpl_table.columns[0]
    id_to_gene = gpl_table.set_index(id_col)[gene_symbol_col].str.upper()

    # Drop probes without gene symbol
    valid_probes = id_to_gene.dropna().index
    valid_probes = valid_probes[~id_to_gene.dropna().str.contains("///|---", na=False)]
    id_to_gene = id_to_gene[valid_probes]
    id_to_gene = id_to_gene[id_to_gene != ""]
    id_to_gene = id_to_gene[id_to_gene != "---"]

    # Filter expression to valid probes
    common = expr.index.intersection(id_to_gene.index)
    expr_filt = expr.loc[common]
    gene_map = id_to_gene[common]

    # Max-mean summarisation
    gene_map_series = pd.Series(gene_map.values, index=gene_map.index)
    expr_filt = expr_filt.set_index(gene_map_series.values)

    # For each gene, keep the probe with highest mean expression
    gene_means = expr_filt.groupby(level=0).mean()
    keep_probes = []
    for gene in gene_means.index:
        subset = expr_filt.loc[gene]
        if subset.ndim == 1:
            keep_probes.append(subset.name)
        else:
            keep_probes.append(subset.mean(axis=1).idxmax())

    expr_gene = expr_filt.loc[gene_means.index].copy()
    # Take the max probe per gene
    expr_gene = expr_gene.groupby(level=0).max()

    print(f"  Probeset -> gene summarisation: {expr.shape[0]} -> {expr_gene.shape[0]} genes")
    return expr_gene


def rescale_to_tcda(geo_expr: pd.DataFrame,
                    tcga_vst_path: str = "data/processed/rna_seq/vst_counts.csv.gz") -> pd.DataFrame:
    """
    Quantile-normalise GEO expression to match TCGA VST distribution.
    Uses common gene intersection only.

    This is critical: GEO is Affymetrix (raw intensities, log2 scale),
    TCGA is RNA-seq VST (variance-stabilised counts). Distributions
    differ substantially — we match gene-level quantiles to align.
    """
    print("  Loading TCGA VST reference for distribution rescaling...")
    tcga_vst = pd.read_csv(tcga_vst_path, index_col=0, compression="gzip")

    # Find common genes
    common_genes = geo_expr.index.intersection(tcga_vst.index)
    print(f"  Common genes: {len(common_genes)}")

    geo_common = geo_expr.loc[common_genes].copy()
    tcga_ref = tcga_vst.loc[common_genes]

    # Per-gene quantile matching
    geo_rescaled = geo_common.copy()
    for gene in common_genes:
        geo_vals = geo_common.loc[gene].values
        tcga_vals = tcga_ref.loc[gene].values
        # Rank GEO values, then map to TCGA quantiles (arrays differ in length)
        geo_valid = ~np.isnan(geo_vals)
        tcga_valid = ~np.isnan(tcga_vals)
        if geo_valid.sum() < 5 or tcga_valid.sum() < 5:
            continue
        ranked = stats.rankdata(geo_vals[geo_valid]) / (geo_valid.sum() + 1)
        tcga_quantiles = np.percentile(tcga_vals[tcga_valid], ranked * 100)
        geo_rescaled.loc[gene, geo_valid] = tcga_quantiles

    print(f"  Rescaled {len(common_genes)} genes to TCGA distribution")
    return geo_rescaled, common_genes.tolist()


def build_geo_features(geo_expr: pd.DataFrame,
                       meta_df: pd.DataFrame,
                       common_genes: list) -> pd.DataFrame:
    """
    Build the same feature set as TCGA for GSE39582:
    1. EMT signature scores
    2. Early warning signals
    3. Physics proxy features
    4. Metastasis label

    Returns a DataFrame compatible with CRCMetastasisDataset.
    """
    print("  Computing EMT signature scores...")
    emt_scores = score_all_patients(geo_expr, signatures=SIGNATURES)
    emt_scores = compute_derived_scores(emt_scores)
    print(f"    EMT scores: {emt_scores.shape}")

    # Early warning signals (cross-sectional)
    epi_genes = SIGNATURES.get("epithelial", [])
    mes_genes = SIGNATURES.get("mesenchymal", [])
    print(f"  Computing EWS (epithelial={len(epi_genes)}, mesenchymal={len(mes_genes)})...")
    ews_df = compute_cross_sectional_ews(geo_expr, epi_genes, mes_genes)
    print(f"    EWS: {ews_df.shape}")

    # Physics proxy features (approximate from EMT scores)
    if "emt_index" in emt_scores.columns:
        physics_df = pd.DataFrame(index=emt_scores.index)
        physics_df["epi_dist"] = emt_scores["epithelial"]
        physics_df["mes_dist"] = emt_scores["mesenchymal"]
        physics_df["attractor_proximity"] = np.abs(physics_df["epi_dist"] - physics_df["mes_dist"])
        physics_df["bifurcation_score"] = 1.0 / (1.0 + physics_df["attractor_proximity"])
        physics_df["physics_score"] = (physics_df["bifurcation_score"] - 0.5) * 2
        physics_df["fitted_T_ext"] = 0.5
        physics_df["in_tipping_zone"] = (physics_df["attractor_proximity"] < 0.5).astype(float)
    else:
        physics_df = pd.DataFrame(index=emt_scores.index)

    # Merge all features
    feature_df = emt_scores.join(ews_df, how="left")
    feature_df = feature_df.join(physics_df, how="left")

    # Rename any duplicate columns
    dup_cols = [c for c in feature_df.columns if c.endswith("_x") or c.endswith("_y")]
    if dup_cols:
        feature_df = feature_df.drop(columns=dup_cols)

    # Add metastasis label from metadata
    meta_clean = meta_df.copy()
    label_col = None
    for c in meta_clean.columns:
        if "tnm.m" in c or "tnm_m" in c or "m stage" in c or "ajcc_m" in c or "metasta" in c:
            label_col = c
            break

    if label_col:
        print(f"  Using '{label_col}' for metastasis label")
        def parse_m(s):
            s = str(s).upper().strip()
            if s.startswith("M1") or s in ["YES", "1"]:
                return 1
            return 0
        meta_clean["metastasis_label"] = meta_clean[label_col].apply(parse_m)
    else:
        print("  WARNING: No M-stage column found. Defaulting label to 0.")
        meta_clean["metastasis_label"] = 0

    common_patients = feature_df.index.intersection(meta_clean.index)
    feature_df = feature_df.loc[common_patients]
    meta_clean = meta_clean.loc[common_patients]

    feature_df["metastasis_label"] = meta_clean["metastasis_label"].astype(int)
    n_pos = int(feature_df["metastasis_label"].sum())
    print(f"  GEO feature matrix: {len(feature_df)} patients, "
          f"{feature_df.shape[1]} features, {n_pos} metastatic")

    return feature_df


def parse_args():
    parser = argparse.ArgumentParser(description="GSE39582 GEO downloader")
    parser.add_argument("--geo-dir", default=GEO_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--tcga-vst", default="data/processed/rna_seq/vst_counts.csv.gz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  GSE39582 Downloader + Feature Builder")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download
    expr_df, meta_df, platform = download_gse39582(args.geo_dir)

    # Probeset -> gene summarisation (gene symbol index)
    expr_gene = probeset_to_gene(expr_df, platform)

    # Map GEO gene symbols to Ensembl IDs using the TCGA gene name map
    gmap_path = "data/processed/rna_seq/gene_name_map.csv"
    ens_to_sym = {}
    if os.path.exists(gmap_path):
        gmap = pd.read_csv(gmap_path)
        sym_to_ens = gmap.set_index("gene_name")["gene_id"].to_dict()
        # Create reverse map keyed by unversioned Ensembl ID
        ens_to_sym = {}
        for _, row in gmap.iterrows():
            eid = row["gene_id"].split(".")[0]
            ens_to_sym[eid] = row["gene_name"]
        # Convert index from symbol -> Ensembl ID for rescaling
        new_idx = []
        kept = []
        for g in expr_gene.index:
            ens = sym_to_ens.get(g.upper() if isinstance(g, str) else g)
            if ens is not None:
                new_idx.append(ens)
                kept.append(True)
            else:
                kept.append(False)
        expr_gene_ens = expr_gene.loc[kept].copy()
        expr_gene_ens.index = new_idx
        print(f"  Mapped to Ensembl IDs: {len(expr_gene_ens)} genes")
    else:
        print(f"  WARNING: {gmap_path} not found, skipping Ensembl mapping")
        expr_gene_ens = expr_gene

    # Rescale to TCGA distribution (works on Ensembl IDs)
    expr_rescaled_ens, common_genes = rescale_to_tcda(expr_gene_ens, args.tcga_vst)

    # Map back to gene symbols for feature building
    if ens_to_sym:
        sym_idx = [ens_to_sym.get(e.split(".")[0] if "." in str(e) else e, e) for e in expr_rescaled_ens.index]
        expr_rescaled = expr_rescaled_ens.copy()
        expr_rescaled.index = sym_idx
        # Drop duplicate symbols
        expr_rescaled = expr_rescaled[~expr_rescaled.index.duplicated(keep="first")]
        print(f"  Mapped back to symbols: {len(expr_rescaled)} genes")
    else:
        expr_rescaled = expr_rescaled_ens

    # Build features (uses gene symbol index)
    feature_df = build_geo_features(expr_rescaled, meta_df, common_genes)

    # Save rescaled expression (with original Ensembl IDs for validation script)
    vst_path = out_dir / "gse39582_rescaled.csv"
    expr_rescaled_ens.to_csv(vst_path)
    print(f"\n  Rescaled expression saved -> {vst_path}")

    # Save
    feat_path = out_dir / "gse39582_features.csv"
    feature_df.to_csv(feat_path)
    print(f"\n  Features saved -> {feat_path}")
    print(f"  Columns: {list(feature_df.columns[:10])}...")
    print("  Done.")

"""
Phase 2 - Step 2: EMT gene signature scoring
==============================================
Computes per-patient scores for:
  1. Epithelial score       (high = epithelial phenotype)
  2. Mesenchymal score      (high = mesenchymal / invasive phenotype)
  3. EMT index              (mesenchymal - epithelial, continuous scale)
  4. TGF-β pathway activity
  5. Proliferation index    (MKI67, PCNA, TOP2A)
  6. TME immune scores      (cytotoxic T, Treg, M2 macrophage proxies)
  7. Wnt/β-catenin score    (frequently dysregulated in CRC)

Scoring method: single-sample Gene Set Enrichment (ssGSEA-lite)
  - Ranks genes within each patient
  - Sums ranks of signature genes → normalised enrichment score
  - More robust than simple mean expression

Usage:
    python src/data/emt_scorer.py
    python src/data/emt_scorer.py --input data/processed/rna_seq/vst_counts.csv.gz
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


# ── Gene signatures ───────────────────────────────────────────────────────────
# Core hallmark EMT genes (validated in CRC literature)

SIGNATURES = {
    # ── Epithelial programme ──────────────────────────────────────────────
    "epithelial": [
        "CDH1",   # E-cadherin — master epithelial marker
        "EPCAM",  # epithelial cell adhesion molecule
        "KRT18",  # keratin 18
        "KRT8",   # keratin 8
        "MUC1",   # mucin 1
        "DSP",    # desmoplakin — cell junction
        "OCLN",   # occludin — tight junction
        "TJP1",   # ZO-1 tight junction
        "CLDN3",  # claudin 3
        "CLDN4",  # claudin 4
    ],

    # ── Mesenchymal programme ─────────────────────────────────────────────
    "mesenchymal": [
        "VIM",    # vimentin — canonical mesenchymal marker
        "FN1",    # fibronectin 1
        "CDH2",   # N-cadherin (cadherin switch)
        "SNAI1",  # Snail — EMT transcription factor
        "SNAI2",  # Slug — EMT transcription factor
        "ZEB1",   # zinc finger E-box binding homeobox 1
        "ZEB2",   # zinc finger E-box binding homeobox 2
        "TWIST1", # twist family bHLH transcription factor 1
        "ACTA2",  # alpha smooth muscle actin
        "MMP2",   # matrix metalloproteinase 2 — ECM degradation
        "MMP9",   # matrix metalloproteinase 9
        "S100A4", # metastasis-associated protein
    ],

    # ── TGF-β / SMAD pathway ─────────────────────────────────────────────
    "tgfb_pathway": [
        "TGFB1",  # TGF-beta 1 ligand
        "TGFB2",  # TGF-beta 2 ligand
        "TGFB3",  # TGF-beta 3 ligand
        "SMAD2",  # SMAD2 — TGF-β signal transducer
        "SMAD3",  # SMAD3
        "SMAD4",  # SMAD4 — common mediator
        "TGFBR1", # TGF-β receptor type 1
        "TGFBR2", # TGF-β receptor type 2
    ],

    # ── Wnt / β-catenin (CRC driver) ─────────────────────────────────────
    "wnt_pathway": [
        "CTNNB1", # β-catenin — Wnt effector
        "APC",    # adenomatous polyposis coli — tumour suppressor
        "AXIN2",  # Wnt target gene / feedback inhibitor
        "MYC",    # c-Myc — Wnt transcriptional target
        "CCND1",  # Cyclin D1 — Wnt target
        "LEF1",   # lymphoid enhancer binding factor 1
        "TCF7L2", # TCF4 — Wnt transcription factor
        "LGR5",   # leucine-rich repeat-containing GPCR 5 — CRC stem cell marker
    ],

    # ── Proliferation ─────────────────────────────────────────────────────
    "proliferation": [
        "MKI67",  # Ki-67
        "PCNA",   # proliferating cell nuclear antigen
        "TOP2A",  # topoisomerase IIa
        "MCM2",   # minichromosome maintenance complex component 2
        "CCNE1",  # cyclin E1
    ],

    # ── Cytotoxic T cell activity (immune surveillance) ───────────────────
    "cytotoxic_t": [
        "CD8A",   # CD8 alpha chain
        "CD8B",   # CD8 beta chain
        "PRF1",   # perforin
        "GZMB",   # granzyme B
        "IFNG",   # interferon gamma
        "CXCL9",  # chemokine attracting T cells
        "CXCL10", # chemokine attracting T cells
    ],

    # ── Immune suppression (Treg + M2 macrophage) ────────────────────────
    "immune_suppression": [
        "FOXP3",  # Treg master transcription factor
        "IL10",   # interleukin 10 — immunosuppressive
        "TGFB1",  # also immunosuppressive in TME context
        "CD163",  # M2 macrophage marker
        "MRC1",   # mannose receptor / M2 marker
        "ARG1",   # arginase 1 — M2 / immunosuppressive
        "PDCD1",  # PD-1
        "CD274",  # PD-L1
    ],

    # ── Hypoxia / angiogenesis ────────────────────────────────────────────
    "hypoxia": [
        "HIF1A",  # hypoxia-inducible factor 1-alpha
        "VEGFA",  # vascular endothelial growth factor A
        "LDHA",   # lactate dehydrogenase A — Warburg effect
        "SLC2A1", # GLUT1 glucose transporter
        "CA9",    # carbonic anhydrase IX — hypoxia marker
    ],
}


# ── ssGSEA-lite scoring ───────────────────────────────────────────────────────

def ssgsea_score(expression: pd.Series, gene_set: list) -> float:
    """
    Single-sample GSEA score for one patient and one gene set.

    Algorithm:
      1. Rank all genes by expression (ascending)
      2. Compute running sum: +rank_fraction when gene in set, -constant otherwise
      3. Score = max deviation of running sum from zero
      (Simplified: we use rank-sum which correlates >0.95 with full ssGSEA)
    """
    # Only use genes present in the expression vector
    genes_present = [g for g in gene_set if g in expression.index]
    if not genes_present:
        return np.nan

    # Rank genes (higher expression = higher rank)
    ranks = expression.rank(method="average", ascending=True)
    n_total = len(ranks)
    n_set   = len(genes_present)

    # Normalised rank sum of signature genes
    rank_sum   = ranks[genes_present].sum()
    # Expected rank sum under null (random set of same size)
    expected   = n_set * (n_total + 1) / 2
    # Normalise by max possible deviation
    max_dev    = n_set * (n_total - n_set) / 2 + expected

    score = (rank_sum - expected) / max_dev if max_dev != 0 else 0.0
    return float(score)


def score_all_patients(vst_matrix: pd.DataFrame,
                       signatures: dict = SIGNATURES,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Score every patient (column) on every signature (rows of output).

    Returns DataFrame: patients × signatures
    """
    n_patients = vst_matrix.shape[1]
    results    = {}

    for sig_name, gene_set in signatures.items():
        if verbose:
            n_found = sum(1 for g in gene_set if g in vst_matrix.index)
            print(f"  Scoring '{sig_name}': {n_found}/{len(gene_set)} genes found")

        scores = vst_matrix.apply(
            lambda col: ssgsea_score(col, gene_set), axis=0
        )
        results[sig_name] = scores

    score_df = pd.DataFrame(results)   # patients × signatures
    return score_df


# ── Derived composite scores ──────────────────────────────────────────────────

def compute_derived_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute biologically meaningful composite indices from raw scores.
    """
    df = score_df.copy()

    # EMT index: continuous scale from epithelial (negative) to mesenchymal (positive)
    if "mesenchymal" in df.columns and "epithelial" in df.columns:
        df["emt_index"] = df["mesenchymal"] - df["epithelial"]

    # Immune balance: cytotoxic surveillance vs suppression
    if "cytotoxic_t" in df.columns and "immune_suppression" in df.columns:
        df["immune_balance"] = df["cytotoxic_t"] - df["immune_suppression"]

    # Invasion potential: EMT + hypoxia + TGF-β composite
    cols = [c for c in ["emt_index", "hypoxia", "tgfb_pathway"] if c in df.columns]
    if cols:
        df["invasion_potential"] = df[cols].mean(axis=1)

    return df


# ── Differential scoring summary ─────────────────────────────────────────────

def differential_summary(scores: pd.DataFrame,
                          manifest: pd.DataFrame,
                          out_dir: Path) -> pd.DataFrame:
    """
    Show mean score per signature broken down by metastasis label.
    Saves a summary CSV and prints a compact table.
    """
    manifest_indexed = manifest.set_index("submitter_id")

    # Align labels to score_df index
    labels = manifest_indexed.reindex(scores.index)["metastasis_label"]
    scores_with_label = scores.copy()
    scores_with_label["metastasis_label"] = labels.values

    labeled = scores_with_label[scores_with_label["metastasis_label"] != -1]

    summary_rows = []
    print(f"\n  {'Signature':<22} {'M0 mean':>10} {'M1 mean':>10} {'Δ':>8}  Direction")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}  {'─'*12}")

    for col in scores.columns:
        m0 = labeled[labeled["metastasis_label"] == 0][col].mean()
        m1 = labeled[labeled["metastasis_label"] == 1][col].mean()
        delta = m1 - m0
        direction = "↑ in M1" if delta > 0.01 else ("↓ in M1" if delta < -0.01 else "~")

        # Mann-Whitney U test for significance
        grp0 = labeled[labeled["metastasis_label"] == 0][col].dropna()
        grp1 = labeled[labeled["metastasis_label"] == 1][col].dropna()
        if len(grp0) > 1 and len(grp1) > 1:
            _, pval = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
        else:
            pval = np.nan

        sig_flag = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"  {col:<22} {m0:>10.4f} {m1:>10.4f} {delta:>+8.4f}  {direction} {sig_flag}")

        summary_rows.append({
            "signature":  col,
            "m0_mean":    round(m0, 5),
            "m1_mean":    round(m1, 5),
            "delta":      round(delta, 5),
            "pvalue":     round(pval, 5) if not np.isnan(pval) else np.nan,
            "direction":  direction,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "signature_differential.csv", index=False)
    return summary_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 - EMT signature scoring")
    parser.add_argument("--input",        default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/rna_seq")
    args = parser.parse_args()

    input_path   = Path(args.input)
    manifest_dir = Path(args.manifest_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 — Step 2: EMT Signature Scoring")
    print("=" * 60)

    # Load VST expression matrix (genes × patients)
    print(f"\n  Loading VST matrix: {input_path}")
    vst = pd.read_csv(input_path, index_col=0, compression="gzip")
    print(f"  Shape: {vst.shape[0]:,} genes × {vst.shape[1]} patients")

    # Transpose so columns = patients for apply()
    print(f"\n  Computing ssGSEA scores for {len(SIGNATURES)} signatures...")
    scores = score_all_patients(vst)          # patients × signatures

    # Derived composite scores
    scores = compute_derived_scores(scores)

    # Differential summary vs metastasis label
    manifest = pd.read_csv(manifest_dir / "cohort_labeled.csv")
    differential_summary(scores, manifest, out_dir)

    # Save
    scores_path = out_dir / "emt_scores.csv"
    scores.to_csv(scores_path)
    print(f"\n  Saved EMT scores → {scores_path}")
    print(f"  Shape: {scores.shape}  (patients × {scores.shape[1]} features)")

    print(f"\n  Next: python src/data/ews_computer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

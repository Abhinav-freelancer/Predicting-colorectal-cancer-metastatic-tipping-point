"""
Phase 2 - Step 3: Early Warning Signal (EWS) computation
==========================================================
Implements dynamical systems theory indicators that detect approach
to a tipping point (bifurcation) BEFORE it happens.

The three canonical EWS for critical transitions:
  1. Critical slowing down (CSD)
      - System recovers slower from perturbations near a bifurcation
      - Proxy: rising lag-1 autocorrelation (AC1) in a rolling window
      - Proxy: rising variance in a rolling window

  2. Flickering
      - System alternates rapidly between attractor states
      - Proxy: rising skewness and kurtosis of expression distribution

  3. Spatial synchrony
      - Cells become more correlated near a transition
      - Proxy: rising mean pairwise correlation across EMT gene pairs

Applied to: EMT index, individual EMT gene expression, and
            signature scores — all treated as pseudo-time series
            (ordered by disease stage/progression proxy).

Usage:
    python src/data/ews_computer.py
    python src/data/ews_computer.py --window 15 --lag 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Optional
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler


# ── Rolling window EWS functions ─────────────────────────────────────────────

def rolling_ac1(series: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    """
    Lag-1 autocorrelation in a rolling window.
    Rising AC1 = critical slowing down signal.

    Returns array of same length as input, NaN for first (window-1) positions.
    """
    n      = len(series)
    ac1    = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = series[i - window + 1 : i + 1]
        # Detrend to remove linear trend before computing autocorrelation
        detrended = detrend(window_data)
        if np.std(detrended) < 1e-10:
            ac1[i] = np.nan
            continue
        # Pearson correlation between series and lagged series
        if len(detrended) > lag:
            r, _ = stats.pearsonr(detrended[:-lag], detrended[lag:])
            ac1[i] = r

    return ac1


def rolling_variance(series: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling variance. Rising variance = critical slowing down signal.
    """
    n   = len(series)
    var = np.full(n, np.nan)
    for i in range(window - 1, n):
        var[i] = np.var(series[i - window + 1 : i + 1], ddof=1)
    return var


def rolling_skewness(series: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling skewness. Rising skewness = flickering signal.
    """
    n    = len(series)
    skew = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = series[i - window + 1 : i + 1]
        skew[i] = stats.skew(w)
    return skew


def rolling_kurtosis(series: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling kurtosis. Heavy tails indicate bistability near tipping.
    """
    n    = len(series)
    kurt = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = series[i - window + 1 : i + 1]
        kurt[i] = stats.kurtosis(w)
    return kurt


def rolling_cv(series: np.ndarray, window: int) -> np.ndarray:
    """Coefficient of variation (std/mean) in a rolling window."""
    n  = len(series)
    cv = np.full(n, np.nan)
    for i in range(window - 1, n):
        w    = series[i - window + 1 : i + 1]
        mean = np.mean(w)
        if abs(mean) > 1e-10:
            cv[i] = np.std(w, ddof=1) / abs(mean)
    return cv


def kendall_tau_trend(signal: np.ndarray) -> tuple[float, float]:
    """
    Kendall's tau trend statistic for a 1D signal (ignoring NaNs).
    Returns (tau, p_value). Positive tau = rising trend (EWS positive).
    """
    valid = ~np.isnan(signal)
    if valid.sum() < 4:
        return np.nan, np.nan
    x = np.arange(len(signal))[valid]
    y = signal[valid]
    tau, pval = stats.kendalltau(x, y)
    return float(tau), float(pval)


# ── Spatial synchrony ─────────────────────────────────────────────────────────

def spatial_synchrony(vst_matrix: pd.DataFrame,
                      gene_set: list,
                      window: int) -> np.ndarray:
    """
    Mean pairwise Pearson correlation across a set of genes,
    computed in a rolling window across patients (ordered by stage).

    Patients must be pre-sorted by disease progression proxy.
    Rising synchrony → system approaching bifurcation.

    Returns array of length n_patients with NaN for first (window-1).
    """
    genes_present = [g for g in gene_set if g in vst_matrix.index]
    if len(genes_present) < 2:
        return np.full(vst_matrix.shape[1], np.nan)

    sub = vst_matrix.loc[genes_present].values.T   # patients × genes
    n   = sub.shape[0]
    sync = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = sub[i - window + 1 : i + 1, :]   # window × genes
        # Pairwise correlation matrix
        if window_data.shape[0] < 3:
            continue
        corr_mat = np.corrcoef(window_data.T)           # genes × genes
        # Mean of upper triangle (excluding diagonal)
        upper_idx = np.triu_indices_from(corr_mat, k=1)
        vals = corr_mat[upper_idx]
        valid_vals = vals[np.isfinite(vals)]
        if len(valid_vals) > 0:
            sync[i] = np.mean(valid_vals)

    return sync


# ── Per-patient cross-sectional EWS ──────────────────────────────────────────

def compute_cross_sectional_ews(vst_matrix: pd.DataFrame,
                                emt_genes_epithelial: list,
                                emt_genes_mesenchymal: list) -> pd.DataFrame:
    """
    For each patient, compute EWS indicators from their gene expression profile.

    In cross-sectional data (no true time series), we treat the distribution
    of EMT-related gene expression WITHIN a patient as a proxy state space,
    and compute statistics that reflect bifurcation proximity.

    Features computed per patient:
      - variance of epithelial gene expression
      - variance of mesenchymal gene expression
      - skewness of EMT gene expression (bimodality indicator)
      - kurtosis of EMT gene expression
      - coefficient of variation of mesenchymal genes
      - correlation between E and M gene modules (should decrease at transition)
    """
    epi_genes  = [g for g in emt_genes_epithelial  if g in vst_matrix.index]
    mes_genes  = [g for g in emt_genes_mesenchymal if g in vst_matrix.index]
    all_emt    = epi_genes + mes_genes

    rows = []
    for patient_id in vst_matrix.columns:
        expr = vst_matrix[patient_id]

        epi_expr = expr[epi_genes].values  if epi_genes  else np.array([np.nan])
        mes_expr = expr[mes_genes].values  if mes_genes  else np.array([np.nan])
        all_expr = expr[all_emt].values    if all_emt    else np.array([np.nan])

        # Variance signals
        var_epi = float(np.nanvar(epi_expr, ddof=1)) if len(epi_expr) > 1 else np.nan
        var_mes = float(np.nanvar(mes_expr, ddof=1)) if len(mes_expr) > 1 else np.nan

        # Distribution shape of overall EMT genes
        skew_emt = float(stats.skew(all_expr[~np.isnan(all_expr)])) if len(all_expr) > 2 else np.nan
        kurt_emt = float(stats.kurtosis(all_expr[~np.isnan(all_expr)])) if len(all_expr) > 2 else np.nan

        # CV of mesenchymal genes
        mes_mean = np.nanmean(mes_expr)
        cv_mes   = (np.nanstd(mes_expr) / abs(mes_mean)) if abs(mes_mean) > 1e-6 else np.nan

        # E-M correlation (should collapse to ~0 or reverse at transition)
        if len(epi_genes) >= 2 and len(mes_genes) >= 2:
            # Mean expression per module → single-patient E and M scalars
            e_scalar = np.nanmean(epi_expr)
            m_scalar = np.nanmean(mes_expr)
            em_ratio = m_scalar / (e_scalar + 1e-6)   # M/E ratio (rises with EMT)
        else:
            em_ratio = np.nan

        rows.append({
            "patient_id":          patient_id,
            "ews_var_epithelial":  var_epi,
            "ews_var_mesenchymal": var_mes,
            "ews_skew_emt":        skew_emt,
            "ews_kurt_emt":        kurt_emt,
            "ews_cv_mesenchymal":  cv_mes,
            "ews_em_ratio":        em_ratio,
        })

    df = pd.DataFrame(rows).set_index("patient_id")
    return df


def compute_spatial_synchrony(vst:           pd.DataFrame,
                               ordered_ids:   list,
                               gene_ids:      list,
                               window:        int) -> np.ndarray:
    """
    Rolling pairwise gene correlation along a pseudo-time trajectory.

    For each window position, computes the mean pairwise Pearson correlation
    (Fisher z-transformed) among the given gene set. Rising correlation
    (spatial synchrony) is a canonical EWS for approaching a tipping point.

    Args:
        vst:         VST expression matrix (index=ENSG IDs, columns=patient IDs)
        ordered_ids: Patient IDs in trajectory (sorted) order
        gene_ids:    ENSG IDs of the genes to correlate (e.g., 7 ODE genes)
        window:      Rolling window size

    Returns:
        Array of spatial synchrony values, same length as ordered_ids,
        NaN for first (window-1) positions.
    """
    n_patients = len(ordered_ids)
    sync = np.full(n_patients, np.nan, dtype=float)

    # Filter to available genes
    avail_ids = [gid for gid in gene_ids if gid in vst.index]
    if len(avail_ids) < 3:
        return sync  # need at least 3 genes for meaningful correlation

    for i in range(n_patients - window + 1):
        win_ids = ordered_ids[i:i + window]
        # Submatrix: genes x patients in window
        sub = vst.loc[avail_ids, [pid for pid in win_ids if pid in vst.columns]]
        if sub.shape[1] < 3:
            continue
        # Transpose to patients x genes for corrcoef
        corr = np.corrcoef(sub.values.astype(float))
        # Fisher z-transform: arctanh(r)
        triu = corr[np.triu_indices_from(corr, k=1)]
        triu = np.clip(triu, -0.999, 0.999)  # avoid inf from r=±1
        zvals = np.arctanh(triu)
        sync[i + window - 1] = float(np.nanmean(zvals))

    return sync


def compute_cohort_ews_trajectory(scores: pd.DataFrame,
                                   manifest: pd.DataFrame,
                                   vst: Optional[pd.DataFrame] = None,
                                   window: int = 10,
                                   lag: int = 1,
                                   sort_by: str = "emt_index") -> pd.DataFrame:
    """
    Treat the cohort as a pseudo-time series ordered by disease progression.
    Default sorting by EMT index (continuous) — captures within-stage variance.
    Falls back to AJCC stage ordering if sort_by is "stage_order".

    Computes rolling EWS indicators on the cohort-level EMT index trajectory.
    Also projects per-patient trajectory position and EWS slope features.

    Returns DataFrame with one row per patient with trajectory EWS + projection features.
    """
    if "emt_index" not in scores.columns:
        return pd.DataFrame()

    manifest_s = manifest.copy()

    if sort_by == "stage_order":
        stage_order = {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3}
        manifest_s["sort_key"] = manifest_s["ajcc_stage"].map(stage_order).fillna(2)
    else:
        # Sort by EMT index (continuous) — captures within-stage variance
        emt_map = scores["emt_index"].to_dict()
        manifest_s["sort_key"] = manifest_s["case_id"].map(emt_map).fillna(0)

    manifest_s = manifest_s.sort_values("sort_key")
    ordered_ids = manifest_s["case_id"].values
    valid_ids   = [pid for pid in ordered_ids if pid in scores.index]
    emt_series  = scores.loc[valid_ids, "emt_index"].values.astype(float)
    n           = len(emt_series)

    # Rolling EWS along sorted trajectory
    ac1   = rolling_ac1(emt_series,      window, lag)
    var_  = rolling_variance(emt_series,  window)
    skew_ = rolling_skewness(emt_series,  window)
    kurt_ = rolling_kurtosis(emt_series,  window)
    cv_   = rolling_cv(emt_series,        window)

    # Kendall tau trend (is EWS rising?)
    tau_ac1,  p_ac1  = kendall_tau_trend(ac1)
    tau_var,  p_var  = kendall_tau_trend(var_)
    tau_skew, p_skew = kendall_tau_trend(skew_)

    print(f"\n  Cohort EWS trajectory (pseudo-time by {sort_by}, N={n}):")
    print(f"    AC1  trend  — τ={tau_ac1:+.3f}, p={p_ac1:.4f}  "
          + ("✓ Rising" if tau_ac1 > 0 and p_ac1 < 0.05 else "—"))
    print(f"    Var  trend  — τ={tau_var:+.3f}, p={p_var:.4f}  "
          + ("✓ Rising" if tau_var > 0 and p_var < 0.05 else "—"))
    print(f"    Skew trend  — τ={tau_skew:+.3f}, p={p_skew:.4f}  "
          + ("✓ Rising" if tau_skew > 0 and p_skew < 0.05 else "—"))

    # ── Spatial synchrony (pairwise gene correlation along trajectory) ────
    # Rising correlation across ODE gene pairs = approaching tipping point
    ODE_GENE_IDS = [
        "ENSG00000039068.19",  # CDH1
        "ENSG00000026025.16",  # VIM
        "ENSG00000124216.4",   # SNAI1
        "ENSG00000148516.22",  # ZEB1
        "ENSG00000105329.11",  # TGFB1
        "ENSG00000207730.3",   # MIR200B
        "ENSG00000100644.17",  # HIF1A
    ]
    spatial_sync = None
    tau_sync = p_sync = np.nan
    if vst is not None:
        spatial_sync = compute_spatial_synchrony(vst, valid_ids, ODE_GENE_IDS, window)
        if not np.all(np.isnan(spatial_sync)):
            tau_sync, p_sync = kendall_tau_trend(spatial_sync)
            print(f"    Sync trend  — τ={tau_sync:+.3f}, p={p_sync:.4f}  "
                  + ("✓ Rising" if tau_sync > 0 and p_sync < 0.05 else "—"))

    # ── Per-patient trajectory projection features ────────────────────────
    # Compute local slopes using central differences (fill ends with NaN)
    def local_slope(arr):
        s = np.full_like(arr, np.nan, dtype=float)
        for i in range(1, len(arr) - 1):
            s[i] = (arr[i + 1] - arr[i - 1]) / 2.0
        return s

    ac1_slope   = local_slope(ac1)
    var_slope   = local_slope(var_)
    skew_slope  = local_slope(skew_)
    # Trajectory position as percentile [0, 1]
    positions   = np.linspace(0, 1, n)

    # Anomaly score: how far local EWS deviates from trajectory trend
    # Use rolling average as baseline, compute z-score of deviation
    ac1_baseline   = uniform_filter1d(np.nan_to_num(ac1, nan=np.nanmean(ac1)), size=window, mode="nearest")
    var_baseline   = uniform_filter1d(np.nan_to_num(var_, nan=np.nanmean(var_)), size=window, mode="nearest")
    ac1_anomaly    = np.abs(ac1 - ac1_baseline) / max(np.nanstd(ac1), 1e-6)
    var_anomaly    = np.abs(var_ - var_baseline) / max(np.nanstd(var_), 1e-6)
    ews_anomaly    = (ac1_anomaly + var_anomaly) / 2.0

    trajectory_df = pd.DataFrame({
        "patient_id":          valid_ids,
        "trajectory_position": positions,
        "emt_index":           emt_series,
        "ews_ac1":             ac1,
        "ews_variance":        var_,
        "ews_skewness":        skew_,
        "ews_kurtosis":        kurt_,
        "ews_cv":              cv_,
        "spatial_synchrony":   spatial_sync if spatial_sync is not None else np.full(n, np.nan),
        "ews_ac1_slope":       ac1_slope,
        "ews_var_slope":       var_slope,
        "ews_skew_slope":      skew_slope,
        "ews_anomaly":         ews_anomaly,
    })

    return trajectory_df


# ── Composite EWS score ───────────────────────────────────────────────────────

def composite_ews_score(cross_sec_ews: pd.DataFrame) -> pd.Series:
    """
    Combine per-patient EWS indicators into a single composite score
    using standardised Z-scores then averaging.

    Higher score → patient's gene expression shows more tipping-point
    characteristics → closer to metastatic transition.
    """
    scaler   = StandardScaler()
    features = cross_sec_ews.dropna(axis=1, how="all")
    if features.empty:
        return pd.Series(np.nan, index=cross_sec_ews.index)

    # Fill remaining NaNs with column median before scaling
    features = features.fillna(features.median())
    scaled   = scaler.fit_transform(features)
    composite = pd.Series(
        scaled.mean(axis=1),
        index=cross_sec_ews.index,
        name="ews_composite",
    )
    return composite


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 - EWS computation")
    parser.add_argument("--vst-input",    default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--scores-input", default="data/processed/rna_seq/emt_scores.csv")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/ews")
    parser.add_argument("--window",       type=int, default=10,
                        help="Rolling window size for trajectory EWS (default 10)")
    parser.add_argument("--lag",          type=int, default=1,
                        help="Autocorrelation lag (default 1)")
    args = parser.parse_args()

    out_dir      = Path(args.out_dir)
    manifest_dir = Path(args.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 2 — Step 3: Early Warning Signal Computation")
    print("=" * 60)

    # Load data
    print(f"\n  Loading VST matrix...")
    vst = pd.read_csv(args.vst_input, index_col=0, compression="gzip")

    print(f"  Loading EMT scores...")
    scores = pd.read_csv(args.scores_input, index_col=0)

    manifest = pd.read_csv(manifest_dir / "cohort_labeled.csv")

    # ── Cross-sectional EWS (per patient) ─────────────────────────────────
    from emt_scorer import SIGNATURES   # reuse gene sets

    # Load gene name map (same approach as emt_scorer)
    gene_map_path = manifest_dir.parent / "processed" / "rna_seq" / "gene_name_map.csv"
    if gene_map_path.exists():
        gmap = pd.read_csv(gene_map_path, index_col=0, header=0)
        rev = {}
        for gid, gname in gmap.iloc[:, 0].items():
            gname = str(gname).strip()
            if gname and gname != "nan":
                rev.setdefault(gname, []).append(gid)
        symbol_to_id = {sym: ids[0] for sym, ids in rev.items()}
        print(f"  Loaded {len(symbol_to_id)} gene symbols → Ensembl IDs")
        epi_genes = [symbol_to_id.get(g, g) for g in SIGNATURES["epithelial"]]
        mes_genes = [symbol_to_id.get(g, g) for g in SIGNATURES["mesenchymal"]]
    else:
        epi_genes = SIGNATURES["epithelial"]
        mes_genes = SIGNATURES["mesenchymal"]

    print(f"\n  Computing cross-sectional EWS (per patient)...")
    cross_ews = compute_cross_sectional_ews(
        vst,
        epi_genes,
        mes_genes,
    )

    # Composite score
    cross_ews["ews_composite"] = composite_ews_score(cross_ews)

    cross_ews.to_csv(out_dir / "patient_ews.csv")
    print(f"  Saved per-patient EWS → {out_dir}/patient_ews.csv")
    print(f"  Shape: {cross_ews.shape}")

    # ── Cohort pseudo-time trajectory EWS ─────────────────────────────────
    print(f"\n  Computing cohort trajectory EWS (window={args.window}, lag={args.lag})...")
    trajectory = compute_cohort_ews_trajectory(
        scores, manifest, vst=vst, window=args.window, lag=args.lag, sort_by="emt_index"
    )

    if not trajectory.empty:
        trajectory.to_csv(out_dir / "cohort_ews_trajectory.csv", index=False)
        print(f"  Saved trajectory EWS → {out_dir}/cohort_ews_trajectory.csv")

    # ── Quick validation: do EWS scores differ by metastasis label? ────────
    manifest_idx = manifest.set_index("case_id")
    labels       = manifest_idx.reindex(cross_ews.index)["metastasis_label"]

    print(f"\n  EWS composite score by label:")
    for label, name in [(0, "Non-metastatic"), (1, "Metastatic")]:
        group = cross_ews.loc[labels == label, "ews_composite"]
        print(f"    {name:<20} mean={group.mean():+.4f}  "
              f"std={group.std():.4f}  n={len(group)}")

    grp0 = cross_ews.loc[labels == 0, "ews_composite"].dropna()
    grp1 = cross_ews.loc[labels == 1, "ews_composite"].dropna()
    if len(grp0) > 1 and len(grp1) > 1:
        u, p = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
        print(f"    Mann-Whitney p = {p:.4f}  "
              + ("✓ Significant" if p < 0.05 else "(not significant — expected on synthetic data)"))

    print(f"\n  Next: python src/data/feature_builder.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Phase 3 - Step 4: Physics feature exporter
============================================
Combines all Phase 3 ODE outputs into a single physics feature
vector per patient, ready to be injected into the Phase 4
deep learning model as physics-informed priors.

Physics features per patient:
    attractor_proximity   — how close to mesenchymal attractor [0,1]
    bifurcation_score     — position on bifurcation diagram [0,1]
    physics_score         — combined physics signal [0,1]
    fitted_T_ext          — patient's effective TGF-β level
    in_tipping_zone       — binary: near bifurcation point
    n_attractors          — 1 (monostable) or 2 (bistable)
    is_bistable           — binary bistability flag
    epi_dist / mes_dist   — distances to each attractor
    current_state_encoded — 0=epithelial, 1=mesenchymal

Also merges with Phase 2 feature matrix to produce the FINAL
combined feature matrix for Phase 4 training.

Usage:
    python src/ode/physics_exporter.py
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))


def run_quick_bifurcation(out_dir: Path) -> dict:
    """
    If bifurcation_points.csv doesn't exist yet, run a quick
    bifurcation sweep with default parameters.
    """
    bif_path = out_dir / "bifurcation_points.csv"
    if bif_path.exists():
        bp = pd.read_csv(bif_path).iloc[0].to_dict()
        print(f"  Loaded bifurcation points: {bp}")
        return bp

    print("  Running bifurcation sweep (default params)...")
    from src.ode.bifurcation import compute_bifurcation_diagram, find_bifurcation_points
    from src.ode.emt_ode import EMTParams

    bif_df = compute_bifurcation_diagram(EMTParams(), t_min=0.0, t_max=3.0, n_steps=80)
    bp     = find_bifurcation_points(bif_df)
    pd.DataFrame([bp]).to_csv(bif_path, index=False)
    bif_df.to_csv(out_dir / "bifurcation_diagram.csv", index=False)
    return bp


def compute_quick_patient_physics(manifest:   pd.DataFrame,
                                  vst_matrix: pd.DataFrame,
                                  bif_points: dict) -> pd.DataFrame:
    """
    Fast per-patient physics features WITHOUT full ODE fitting.
    Uses EMT scores for classic proxy and raw expression of the 7 ODE
    state genes (CDH1, VIM, SNAI1, ZEB1, TGFB1, MIR200B, HIF1A) for
    physics-inspired derived features.
    """
    T_lower = bif_points.get("T_lower", 0.8)
    T_upper = bif_points.get("T_upper", 2.0)
    width   = T_upper - T_lower

    # Load EMT scores
    emt_path = Path("data/processed/rna_seq/emt_scores.csv")
    if not emt_path.exists():
        return pd.DataFrame()
    emt = pd.read_csv(emt_path, index_col=0)

    # ODE state gene ENSG IDs (with version suffixes)
    ODE_GENE_IDS = {
        "CDH1":  "ENSG00000039068.19",
        "VIM":   "ENSG00000026025.16",
        "SNAI1": "ENSG00000124216.4",
        "ZEB1":  "ENSG00000148516.22",
        "TGFB1": "ENSG00000105329.11",
        "MIR200B": "ENSG00000207730.3",
        "HIF1A": "ENSG00000100644.17",
    }
    ODE_STATE_VARS = ["E", "M", "S", "Z", "T", "R", "H"]

    rows = []
    for i, pid in enumerate(manifest["case_id"]):
        if pid not in emt.index:
            continue

        # ── Derived features from raw ODE gene expression ─────────────────
        # Extract 7 ODE gene values; use 0 if gene not found
        g = {}
        for sym, gid in ODE_GENE_IDS.items():
            if gid in vst_matrix.index:
                g[sym] = vst_matrix.loc[gid, pid] if pid in vst_matrix.columns else 0.0
            else:
                g[sym] = 0.0
        g = {k: float(v) for k, v in g.items()}
        eps = 1e-8

        # ── T_ext proxy from TGFB1 expression ────────────────────────────
        # TGF-β is the bifurcation parameter; map VST range (6.89-13.71) → [0, 3.5]
        # so the tipping zone (T_ext ~0-0.5) covers the lower ~20% of TGFB1
        tgfb1_raw = g.get("TGFB1", 0.0)
        T_ext_proxy = float(np.clip((tgfb1_raw - 7.0) / 4.0, 0.0, 3.5))

        if T_ext_proxy < T_lower:
            dist = T_lower - T_ext_proxy
            bif_score = max(0.0, 0.5 - 0.5 * (dist / (width + 0.1)))
        elif T_ext_proxy > T_upper:
            dist = T_ext_proxy - T_upper
            bif_score = min(1.0, 0.5 + 0.5 * (dist / (width + 0.1)))
        else:
            frac = (T_ext_proxy - T_lower) / (T_upper - T_lower + 1e-6)
            bif_score = 0.4 + 0.2 * frac

        # ── Attractor proximity from raw 7-gene expression ──────────────
        # Epithelial score = CDH1 + MIR200B; Mesenchymal score = VIM + SNAI1 + ZEB1
        epi_score = g["CDH1"] + g["MIR200B"]
        mes_score = g["VIM"] + g["SNAI1"] + g["ZEB1"]
        att_prox = float(np.clip(mes_score / max(epi_score + mes_score, eps), 0.0, 1.0))
        in_tipping = (T_lower - 0.3 * width <= T_ext_proxy <= T_upper + 0.3 * width)

        cdh1_vim_ratio    = g["CDH1"] / max(g["VIM"], eps)
        snai1_expression  = g["SNAI1"]
        hif1a_expression  = g["HIF1A"]
        tgfb1_expression  = g["TGFB1"]
        zeb1_mir200b_ratio = g["ZEB1"] / max(g["MIR200B"], eps)
        emt_tf_activity    = (g["SNAI1"] + g["ZEB1"]) / 2.0
        epith_program      = (g["CDH1"] + g["MIR200B"]) / 2.0
        emt_switch_index   = (g["VIM"] + g["SNAI1"] + g["ZEB1"]) / max(g["CDH1"] + g["MIR200B"] + g["HIF1A"], eps)
        tgf_hypoxia_synergy = g["TGFB1"] * g["HIF1A"]
        tgfb_snai1_axis    = g["TGFB1"] * g["SNAI1"]
        vim_expression     = g["VIM"]
        mir200b_expression = g["MIR200B"]
        em_balance         = (g["CDH1"] - g["VIM"]) / max(g["CDH1"] + g["VIM"], eps)
        epithelial_integrity = g["CDH1"] / max(g["CDH1"] + g["SNAI1"] + g["ZEB1"], eps)
        tgf_hypoxia_loop   = g["TGFB1"] * g["HIF1A"] / max(g["MIR200B"], eps)
        # ── Network motif features ─────────────────────────────────────
        snai1_zeb1_motif   = g["SNAI1"] * g["ZEB1"]  # S-Z mutual activation product
        tgfb_emt_tf_axis   = g["TGFB1"] * (g["SNAI1"] + g["ZEB1"]) / 2.0  # TGF-b driving EMT-TFs
        cdh1_mir200b_motif = g["CDH1"] * g["MIR200B"]  # epithelial barrier product

        rows.append({
            "patient_id":           pid,
            # Classic proxy
            "fitted_T_ext":         round(T_ext_proxy, 4),
            "attractor_proximity":  round(att_prox, 5),
            "bifurcation_score":    round(bif_score, 5),
            "physics_score":        round((bif_score + att_prox) / 2, 5),
            "in_tipping_zone":      in_tipping,
            "n_attractors":         2 if in_tipping else 1,
            "is_bistable":          in_tipping,
            "epi_dist":             round(1.0 - att_prox, 5),
            "mes_dist":             round(att_prox, 5),
            "phase_portrait_angle": round(float(np.arctan2(att_prox, 1.0 - att_prox)), 5),
            "current_state":        "mesenchymal" if att_prox > 0.5 else "epithelial",
            "current_state_encoded": 1 if att_prox > 0.5 else 0,
            # ODE gene-derived features (raw biology)
            "cdh1_vim_ratio":       round(cdh1_vim_ratio, 4),
            "snai1_expression":     round(snai1_expression, 4),
            "hif1a_expression":     round(hif1a_expression, 4),
            "tgfb1_expression":     round(tgfb1_expression, 4),
            "zeb1_mir200b_ratio":   round(zeb1_mir200b_ratio, 4),
            "emt_tf_activity":      round(emt_tf_activity, 4),
            "epith_program":        round(epith_program, 4),
            "em_balance":           round(em_balance, 4),
            "epithelial_integrity": round(epithelial_integrity, 4),
            "tgf_hypoxia_loop":     round(tgf_hypoxia_loop, 4),
            "snai1_zeb1_motif":     round(snai1_zeb1_motif, 4),
            "tgfb_emt_tf_axis":     round(tgfb_emt_tf_axis, 4),
            "cdh1_mir200b_motif":   round(cdh1_mir200b_motif, 4),
            "emt_switch_index":     round(emt_switch_index, 4),
            "tgf_hypoxia_synergy":  round(tgf_hypoxia_synergy, 4),
            "tgfb_snai1_axis":      round(tgfb_snai1_axis, 4),
            "vim_expression":       round(vim_expression, 4),
            "mir200b_expression":   round(mir200b_expression, 4),
        })

    return pd.DataFrame(rows).set_index("patient_id")


def merge_with_phase2(physics_df:  pd.DataFrame,
                      phase2_path: Path,
                      manifest_dir: Path = None) -> pd.DataFrame:
    """
    Left-join physics features with Phase 2 feature matrix.
    Physics columns are added as additional predictors.
    Also merges metastasis_label, ajcc_stage, ajcc_m from manifest
    so evaluation scripts can find them.
    """
    if not phase2_path.exists():
        print(f"  ⚠ Phase 2 feature matrix not found at {phase2_path}")
        return physics_df

    phase2 = pd.read_csv(phase2_path, index_col=0)
    phase2.index.name = "patient_id"

    # Only use physics feature columns (not labels)
    phys_feat_cols = [
        "fitted_T_ext", "attractor_proximity", "bifurcation_score",
        "physics_score", "in_tipping_zone", "n_attractors", "is_bistable",
        "epi_dist", "mes_dist", "current_state_encoded",
        "phase_portrait_angle",
        "cdh1_vim_ratio", "snai1_expression", "hif1a_expression",
        "tgfb1_expression", "zeb1_mir200b_ratio", "emt_tf_activity",
        "emt_switch_index", "tgf_hypoxia_synergy", "tgfb_snai1_axis",
        "vim_expression", "mir200b_expression",
        "epith_program", "em_balance", "epithelial_integrity",
        "tgf_hypoxia_loop", "snai1_zeb1_motif", "tgfb_emt_tf_axis",
        "cdh1_mir200b_motif",
    ]
    phys_cols_present = [c for c in phys_feat_cols if c in physics_df.columns]
    phys_sub = physics_df[phys_cols_present].copy()
    phys_sub.index.name = "patient_id"

    merged = phase2.join(phys_sub, how="left")

    # Fill missing physics features with neutral value (0.5 = uncertain)
    for col in phys_cols_present:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.5)

    # ── Cross-phase interaction features ─────────────────────────────────
    # TGF-β drives EMT most strongly when the system is in the bistable regime.
    if "tgfb_pathway" in merged.columns and "bifurcation_score" in merged.columns:
        merged["tgfb_x_bifurcation"] = merged["tgfb_pathway"] * merged["bifurcation_score"]

    # EMT index is most dangerous when patient is near the tipping point.
    if "emt_index" in merged.columns and "in_tipping_zone" in merged.columns:
        merged["emt_x_tipping"] = merged["emt_index"] * merged["in_tipping_zone"].astype(float)

    # ── Merge labels from labels.csv (co-located with phase2 feature matrix) ──
    labels_path = phase2_path.parent / "labels.csv"
    if labels_path.exists():
        label_df = pd.read_csv(labels_path, index_col=0)
        label_cols = ["metastasis_label", "ajcc_stage", "ajcc_m"]
        label_df.index.name = "patient_id"
        for c in label_cols:
            if c not in merged.columns and c in label_df.columns:
                merged[c] = label_df[c]
        print(f"    Merged labels from labels.csv: "
              f"{[c for c in label_cols if c in merged.columns]}")
    elif manifest_dir is not None:
        manifest_path = manifest_dir / "cohort_labeled.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            label_cols = ["metastasis_label", "ajcc_stage", "ajcc_m"]
            manifest_dedup = manifest.drop_duplicates(subset="case_id")
            manifest_sub = manifest_dedup.set_index("case_id")[label_cols]
            manifest_sub.index.name = "patient_id"
            for c in label_cols:
                if c not in merged.columns and c in manifest_sub.columns:
                    merged[c] = manifest_sub[c]
            print(f"    Merged labels from manifest (fallback): "
                  f"{[c for c in label_cols if c in merged.columns]}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Physics feature export")
    parser.add_argument("--ode-params",   default="data/processed/temporal/ode_patient_params.csv")
    parser.add_argument("--phase2-features", default="data/processed/temporal/feature_matrix.csv")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--vst-input",    default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--out-dir",      default="data/processed/temporal")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 3 — Step 4: Physics Feature Export")
    print("=" * 60)

    manifest = pd.read_csv(Path(args.manifest_dir) / "cohort_labeled.csv")
    vst      = pd.read_csv(args.vst_input, index_col=0, compression="gzip")

    # Load or compute bifurcation points
    bif_points = run_quick_bifurcation(out_dir)

    # Load ODE fitting results or compute proxy
    ode_path = Path(args.ode_params)
    use_proxy = True
    if ode_path.exists():
        ode_df = pd.read_csv(ode_path)
        # Detect failed ODE fits: if all patients have state=unknown or
        # attractor_proximity is constant, ODE fitting didn't converge
        all_unknown = "current_state" in ode_df.columns and (ode_df["current_state"] == "unknown").all()
        prox_const = "attractor_proximity" in ode_df.columns and ode_df["attractor_proximity"].nunique() <= 1
        if all_unknown or prox_const:
            print(f"\n  ODE fitting results found but FAILED (all state=unknown or constant proxy)")
            print(f"  Falling back to EMT-score-based proxy physics features...")
        else:
            print(f"\n  Loading ODE fitting results from {ode_path}...")
            if "bifurcation_score" not in ode_df.columns:
                from src.ode.bifurcation import compute_patient_bifurcation_scores
                ode_df = compute_patient_bifurcation_scores(ode_df, bif_points)
            ode_df["current_state_encoded"] = (ode_df["current_state"] == "mesenchymal").astype(int)
            physics_df = ode_df.set_index("patient_id")
            use_proxy = False

    if use_proxy:
        print(f"  Computing proxy physics features (EMT-score based)...")
        physics_df = compute_quick_patient_physics(manifest, vst, bif_points)

    print(f"  Physics features shape: {physics_df.shape}")

    # ── Physics feature summary ───────────────────────────────────────────
    print(f"\n  Physics feature summary:")
    key_cols = ["attractor_proximity", "bifurcation_score", "physics_score",
                "fitted_T_ext", "in_tipping_zone"]
    for col in key_cols:
        if col in physics_df.columns:
            vals = physics_df[col].astype(float)
            print(f"    {col:<25}: mean={vals.mean():.4f}  std={vals.std():.4f}")

    # Tipping zone statistics
    if "in_tipping_zone" in physics_df.columns:
        n_tip = physics_df["in_tipping_zone"].sum()
        print(f"\n  Patients in tipping zone: {n_tip}/{len(physics_df)} "
              f"({100*n_tip/len(physics_df):.1f}%)")

    # Save physics features alone
    phys_path = out_dir / "physics_features.csv"
    physics_df.to_csv(phys_path)
    print(f"\n  Saved physics features → {phys_path}")

    # ── Merge with Phase 2 features → final Phase 4 input ─────────────────
    print(f"\n  Merging with Phase 2 feature matrix...")
    final_features = merge_with_phase2(
        physics_df, Path(args.phase2_features), manifest_dir=Path(args.manifest_dir)
    )

    final_path = out_dir / "phase4_input.csv"
    final_features.to_csv(final_path)
    print(f"  Saved Phase 4 input → {final_path}")
    print(f"  Final feature matrix: {final_features.shape[0]} patients �- "
          f"{final_features.shape[1]} features")

    # Feature group count
    p2_cols  = pd.read_csv(args.phase2_features, index_col=0, nrows=0).columns.tolist()
    phys_new = [c for c in final_features.columns
                if c not in p2_cols and not c.startswith("metastasis")]
    print(f"\n  Feature breakdown:")
    print(f"    Phase 2 (EMT + EWS + clinical) : {len(p2_cols)}")
    print(f"    Phase 3 (ODE physics)           : {len(phys_new)}")
    print(f"    Total                           : {final_features.shape[1]}")

    print(f"\n  Phase 3 complete ✓")
    print(f"  Phase 4 input ready at: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

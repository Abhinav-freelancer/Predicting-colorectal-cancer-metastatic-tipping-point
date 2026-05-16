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

    bif_df = compute_bifurcation_diagram(EMTParams(), t_min=0.0, t_max=3.5, n_steps=40)
    bp     = find_bifurcation_points(bif_df)
    pd.DataFrame([bp]).to_csv(bif_path, index=False)
    bif_df.to_csv(out_dir / "bifurcation_diagram.csv", index=False)
    return bp


def compute_quick_patient_physics(manifest:   pd.DataFrame,
                                  vst_matrix: pd.DataFrame,
                                  bif_points: dict) -> pd.DataFrame:
    """
    Fast per-patient physics features WITHOUT full ODE fitting
    (uses EMT scores as proxy for T_ext).
    Used when param_fitter.py hasn't been run yet.
    """
    from src.ode.emt_ode import EMTParams, simulate

    T_lower = bif_points.get("T_lower", 0.8)
    T_upper = bif_points.get("T_upper", 2.0)
    width   = T_upper - T_lower

    # Load EMT scores from Phase 2
    emt_path = Path("data/processed/rna_seq/emt_scores.csv")
    if not emt_path.exists():
        return pd.DataFrame()

    emt = pd.read_csv(emt_path, index_col=0)

    rows = []
    for pid in manifest["submitter_id"]:
        if pid not in emt.index:
            continue

        # Proxy T_ext from invasion_potential score (Phase 2)
        # invasion_potential ranges ~ -0.5 to +0.8 → rescale to [0, 3]
        inv = emt.loc[pid, "invasion_potential"] if "invasion_potential" in emt.columns else 0.0
        T_ext_proxy = float(np.clip((inv + 0.5) * 2.0, 0.0, 3.5))

        # Bifurcation score
        if T_ext_proxy < T_lower:
            dist = T_lower - T_ext_proxy
            bif_score = max(0.0, 0.5 - 0.5 * (dist / (width + 0.1)))
        elif T_ext_proxy > T_upper:
            dist = T_ext_proxy - T_upper
            bif_score = min(1.0, 0.5 + 0.5 * (dist / (width + 0.1)))
        else:
            frac = (T_ext_proxy - T_lower) / (T_upper - T_lower + 1e-6)
            bif_score = 0.4 + 0.2 * frac

        # Attractor proximity from EMT index (if available)
        emt_idx = emt.loc[pid, "emt_index"] if "emt_index" in emt.columns else 0.0
        att_prox = float(np.clip((emt_idx + 1.0) / 2.0, 0.0, 1.0))

        in_tipping = (T_lower - 0.3 * width <= T_ext_proxy <= T_upper + 0.3 * width)

        rows.append({
            "patient_id":           pid,
            "fitted_T_ext":         round(T_ext_proxy, 4),
            "attractor_proximity":  round(att_prox, 5),
            "bifurcation_score":    round(bif_score, 5),
            "physics_score":        round((bif_score + att_prox) / 2, 5),
            "in_tipping_zone":      in_tipping,
            "n_attractors":         2 if in_tipping else 1,
            "is_bistable":          in_tipping,
            "epi_dist":             round(1.0 - att_prox, 5),
            "mes_dist":             round(att_prox, 5),
            "current_state":        "mesenchymal" if emt_idx > 0 else "epithelial",
            "current_state_encoded": 1 if emt_idx > 0 else 0,
        })

    return pd.DataFrame(rows).set_index("patient_id")


def merge_with_phase2(physics_df:  pd.DataFrame,
                      phase2_path: Path) -> pd.DataFrame:
    """
    Left-join physics features with Phase 2 feature matrix.
    Physics columns are added as additional predictors.
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
    ]
    phys_cols_present = [c for c in phys_feat_cols if c in physics_df.columns]
    phys_sub = physics_df[phys_cols_present].copy()
    phys_sub.index.name = "patient_id"

    merged = phase2.join(phys_sub, how="left")

    # Fill missing physics features with neutral value (0.5 = uncertain)
    for col in phys_cols_present:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.5)

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
    if ode_path.exists():
        print(f"\n  Loading ODE fitting results from {ode_path}...")
        ode_df    = pd.read_csv(ode_path)

        # Compute bifurcation scores if not present
        if "bifurcation_score" not in ode_df.columns:
            from src.ode.bifurcation import compute_patient_bifurcation_scores
            ode_df = compute_patient_bifurcation_scores(ode_df, bif_points)

        ode_df["current_state_encoded"] = (ode_df["current_state"] == "mesenchymal").astype(int)
        physics_df = ode_df.set_index("patient_id")
    else:
        print(f"\n  ODE fitting results not found. Computing proxy physics features...")
        print(f"  (Run param_fitter.py for full fitted features)")
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
    final_features = merge_with_phase2(physics_df, Path(args.phase2_features))

    final_path = out_dir / "phase4_input.csv"
    final_features.to_csv(final_path)
    print(f"  Saved Phase 4 input → {final_path}")
    print(f"  Final feature matrix: {final_features.shape[0]} patients × "
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

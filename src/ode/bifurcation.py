"""
Phase 3 - Step 3: Bifurcation diagram computation
===================================================
Sweeps the bifurcation parameter (TGF-β external input, T_ext) from 0 → 3
and traces how the stable steady states (attractors) change.

The bifurcation diagram reveals:
  - The MONOSTABLE EPITHELIAL region  (low T_ext: only E attractor exists)
  - The BISTABLE region               (intermediate T_ext: both E and M exist)
  - The MONOSTABLE MESENCHYMAL region (high T_ext: only M attractor exists)

The two FOLD BIFURCATION POINTS (saddle-node bifurcations) define:
  - T_lower: T_ext at which the epithelial attractor disappears
  - T_upper: T_ext at which the mesenchymal attractor first appears (on forward sweep)

The BISTABLE window [T_lower, T_upper] is the tipping zone.
Patients whose TGF-β level falls in this window are near the tipping point.

Also computes:
  - Per-patient T_ext estimate (from fitted params)
  - Per-patient position on the bifurcation diagram
  - Tipping proximity score based on distance to nearest fold point

Usage:
    python src/ode/bifurcation.py
    python src/ode/bifurcation.py --t-max 4.0 --n-steps 80
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.ode.emt_ode import EMTParams, simulate, find_steady_states


# ── Bifurcation diagram ───────────────────────────────────────────────────────

def compute_bifurcation_diagram(params_template: EMTParams,
                                t_min:  float = 0.0,
                                t_max:  float = 3.5,
                                n_steps: int  = 60) -> pd.DataFrame:
    """
    Sweep T_ext from t_min to t_max and record stable steady states.

    For each T_ext value:
      - Run ODE from epithelial IC  → find epithelial branch (if stable)
      - Run ODE from mesenchymal IC → find mesenchymal branch (if stable)

    Returns DataFrame with columns:
        T_ext, E_epi, M_epi, E_mes, M_mes,
        epi_stable, mes_stable, region
    """
    t_values = np.linspace(t_min, t_max, n_steps)
    rows     = []

    # Initial conditions for each branch
    IC_EPI = [1.5, 0.1, 0.05, 0.05, 0.1]    # near epithelial attractor
    IC_MES = [0.05, 2.0, 1.5, 1.5, 0.5]     # near mesenchymal attractor

    print(f"  Sweeping T_ext: {t_min:.2f} → {t_max:.2f} ({n_steps} steps)...")

    for i, T_ext in enumerate(t_values):
        p         = EMTParams(**vars(params_template))
        p.T_ext   = T_ext

        # ── Epithelial branch ─────────────────────────────────────────
        try:
            res_epi   = simulate(p, y0=IC_EPI, t_span=(0, 400), n_points=200)
            ss_epi    = res_epi["steady_state"]
            epi_E     = ss_epi["E"]
            epi_M     = ss_epi["M"]
            epi_stable = epi_E > epi_M   # still in epithelial attractor
        except Exception:
            epi_E, epi_M, epi_stable = np.nan, np.nan, False

        # ── Mesenchymal branch ────────────────────────────────────────
        try:
            res_mes   = simulate(p, y0=IC_MES, t_span=(0, 400), n_points=200)
            ss_mes    = res_mes["steady_state"]
            mes_E     = ss_mes["E"]
            mes_M     = ss_mes["M"]
            mes_stable = mes_M > mes_E   # still in mesenchymal attractor
        except Exception:
            mes_E, mes_M, mes_stable = np.nan, np.nan, False

        # ── Classify region ───────────────────────────────────────────
        if epi_stable and mes_stable:
            # Check they're distinct attractors (not converged to same point)
            if not np.isnan(epi_E) and not np.isnan(mes_E):
                if abs(epi_E - mes_E) > 0.2 or abs(epi_M - mes_M) > 0.2:
                    region = "bistable"
                else:
                    region = "monostable_epi" if epi_E > mes_E else "monostable_mes"
            else:
                region = "bistable"
        elif epi_stable:
            region = "monostable_epi"
        elif mes_stable:
            region = "monostable_mes"
        else:
            region = "transitioning"

        rows.append({
            "T_ext":       round(T_ext, 4),
            "E_epi":       round(float(epi_E), 5) if not np.isnan(epi_E) else np.nan,
            "M_epi":       round(float(epi_M), 5) if not np.isnan(epi_M) else np.nan,
            "E_mes":       round(float(mes_E), 5) if not np.isnan(mes_E) else np.nan,
            "M_mes":       round(float(mes_M), 5) if not np.isnan(mes_M) else np.nan,
            "epi_stable":  epi_stable,
            "mes_stable":  mes_stable,
            "region":      region,
        })

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_steps}] T_ext={T_ext:.2f}  region={region}")

    return pd.DataFrame(rows)


def find_bifurcation_points(bif_df: pd.DataFrame) -> dict:
    """
    Identify the two fold bifurcation points from the diagram.

    T_lower: transition from monostable_epi to bistable
             (mesenchymal attractor appears)
    T_upper: transition from bistable to monostable_mes
             (epithelial attractor disappears)

    Returns dict with T_lower, T_upper, bistable_width
    """
    bistable_rows = bif_df[bif_df["region"] == "bistable"]

    if bistable_rows.empty:
        print("  ⚠ No bistable region found. Try increasing t_max.")
        return {"T_lower": np.nan, "T_upper": np.nan, "bistable_width": 0}

    T_lower = float(bistable_rows["T_ext"].min())
    T_upper = float(bistable_rows["T_ext"].max())
    width   = T_upper - T_lower

    return {
        "T_lower":        round(T_lower, 4),
        "T_upper":        round(T_upper, 4),
        "bistable_width": round(width, 4),
        "bistable_center": round((T_lower + T_upper) / 2, 4),
    }


# ── Per-patient bifurcation positioning ──────────────────────────────────────

def compute_patient_bifurcation_scores(ode_params_df: pd.DataFrame,
                                       bif_points:    dict) -> pd.DataFrame:
    """
    For each patient (with their fitted T_ext), compute:
      1. Position on bifurcation diagram
      2. Distance to lower fold point (T_lower)
      3. Distance to upper fold point (T_upper)
      4. Bifurcation proximity score:
            0.0 = safely in epithelial monostable region
            0.5 = exactly at a fold bifurcation point (TIPPING POINT)
            1.0 = safely in mesenchymal monostable region

    The tipping zone is defined as patients within
    1 x bistable_width of either fold point.
    """
    T_lower = bif_points["T_lower"]
    T_upper = bif_points["T_upper"]
    width   = bif_points.get("bistable_width", 1.0)

    if np.isnan(T_lower) or np.isnan(T_upper):
        ode_params_df["bifurcation_score"] = 0.5
        return ode_params_df

    df = ode_params_df.copy()

    def bif_score(T_ext: float) -> float:
        """
        Map T_ext to a bifurcation score in [0, 1].
        Uses a sigmoid-like mapping centred on the bistable region.
        """
        if T_ext < T_lower:
            # Epithelial monostable: score decreases with distance from T_lower
            dist = T_lower - T_ext
            return max(0.0, 0.5 - 0.5 * (dist / (width + 0.1)))
        elif T_ext > T_upper:
            # Mesenchymal monostable: score increases with distance from T_upper
            dist = T_ext - T_upper
            return min(1.0, 0.5 + 0.5 * (dist / (width + 0.1)))
        else:
            # Bistable zone: interpolate between 0.4 and 0.6 (near tipping point)
            frac = (T_ext - T_lower) / (T_upper - T_lower + 1e-6)
            return 0.4 + 0.2 * frac

    df["T_ext_fitted"]      = df["fitted_T_ext"]
    df["dist_to_T_lower"]   = (df["fitted_T_ext"] - T_lower).abs()
    df["dist_to_T_upper"]   = (df["fitted_T_ext"] - T_upper).abs()
    df["bifurcation_score"] = df["fitted_T_ext"].apply(bif_score)
    df["in_tipping_zone"]   = (
        (df["fitted_T_ext"] >= T_lower - 0.3 * width) &
        (df["fitted_T_ext"] <= T_upper + 0.3 * width)
    )

    # Combine with attractor_proximity for a unified physics score
    if "attractor_proximity" in df.columns:
        df["physics_score"] = (df["bifurcation_score"] + df["attractor_proximity"]) / 2
    else:
        df["physics_score"] = df["bifurcation_score"]

    return df


# ── ASCII bifurcation diagram ─────────────────────────────────────────────────

def print_bifurcation_ascii(bif_df: pd.DataFrame,
                             bif_points: dict,
                             width: int = 55) -> None:
    """
    Print a compact ASCII representation of the bifurcation diagram.
    Shows E-cadherin steady state vs T_ext for both branches.
    """
    print(f"\n  Bifurcation diagram  (E-cadherin vs TGF-β input)")
    print(f"  {'─' * width}")

    T_lower = bif_points.get("T_lower", np.nan)
    T_upper = bif_points.get("T_upper", np.nan)

    # Determine axis ranges
    E_max    = max(bif_df["E_epi"].dropna().max(), bif_df["E_mes"].dropna().max(), 0.1)
    T_values = bif_df["T_ext"].values
    n_cols   = min(len(T_values), width)
    step     = max(1, len(T_values) // n_cols)

    rows_to_show = bif_df.iloc[::step]
    n_rows   = 12
    E_levels = np.linspace(E_max * 1.1, 0, n_rows)

    print(f"  E  ↑")
    for e_level in E_levels:
        line = f" {e_level:4.1f} │"
        for _, row in rows_to_show.iterrows():
            epi_near = (not np.isnan(row["E_epi"]) and
                        abs(row["E_epi"] - e_level) < E_max / (n_rows * 1.5))
            mes_near = (not np.isnan(row["E_mes"]) and
                        abs(row["E_mes"] - e_level) < E_max / (n_rows * 1.5))
            if epi_near and mes_near:
                line += "◆"
            elif epi_near:
                line += "●"
            elif mes_near:
                line += "○"
            elif row["region"] == "bistable":
                line += "·"
            else:
                line += " "
        print(f"  {line}")

    t_axis = f"  {'':5}└{'─' * len(rows_to_show)}"
    print(t_axis + f"  TGF-β →")
    print(f"  {'':6}{T_values[0]:.1f}{' ' * (len(rows_to_show)-8)}{T_values[-1]:.1f}")
    print(f"\n  ● Epithelial branch (stable)  ○ Mesenchymal branch  ◆ Bistable region")

    if not np.isnan(T_lower):
        print(f"\n  Fold bifurcation points:")
        print(f"    T_lower = {T_lower:.3f}  (mesenchymal attractor appears)")
        print(f"    T_upper = {T_upper:.3f}  (epithelial attractor disappears)")
        print(f"    Bistable window width = {bif_points['bistable_width']:.3f}")
    else:
        print(f"\n  ⚠ Bistable region not detected in this T_ext range.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Bifurcation computation")
    parser.add_argument("--ode-params",   default="data/processed/temporal/ode_patient_params.csv",
                        help="Output from param_fitter.py (optional; uses defaults if absent)")
    parser.add_argument("--out-dir",      default="data/processed/temporal")
    parser.add_argument("--t-min",        type=float, default=0.0)
    parser.add_argument("--t-max",        type=float, default=3.5)
    parser.add_argument("--n-steps",      type=int,   default=60)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 3 — Step 3: Bifurcation Computation")
    print("=" * 60)

    # Use default parameters as template for population-level bifurcation
    params_template = EMTParams()

    # ── Compute bifurcation diagram ───────────────────────────────────────
    bif_df = compute_bifurcation_diagram(
        params_template,
        t_min    = args.t_min,
        t_max    = args.t_max,
        n_steps  = args.n_steps,
    )

    bif_points = find_bifurcation_points(bif_df)

    print(f"\n  Bifurcation points found:")
    for k, v in bif_points.items():
        print(f"    {k:<22}: {v}")

    print_bifurcation_ascii(bif_df, bif_points)

    # Save bifurcation diagram
    bif_path = out_dir / "bifurcation_diagram.csv"
    bif_df.to_csv(bif_path, index=False)
    print(f"\n  Saved bifurcation diagram → {bif_path}")

    # Save bifurcation points
    bp_df = pd.DataFrame([bif_points])
    bp_df.to_csv(out_dir / "bifurcation_points.csv", index=False)

    # ── Per-patient scores (if ODE fitting has been run) ──────────────────
    ode_params_path = Path(args.ode_params)
    if ode_params_path.exists():
        print(f"\n  Computing per-patient bifurcation scores...")
        ode_df   = pd.read_csv(ode_params_path)
        scored   = compute_patient_bifurcation_scores(ode_df, bif_points)
        score_path = out_dir / "patient_bifurcation_scores.csv"
        scored.to_csv(score_path, index=False)
        print(f"  Saved → {score_path}")

        n_tipping = scored["in_tipping_zone"].sum()
        print(f"\n  Patients in tipping zone : {n_tipping} / {len(scored)}"
              f"  ({100*n_tipping/len(scored):.1f}%)")
        print(f"  Mean physics score       : {scored['physics_score'].mean():.4f}")
    else:
        print(f"\n  ℹ No ODE fitting results found at {ode_params_path}.")
        print(f"    Run param_fitter.py to get per-patient scores.")

    print(f"\n  Phase 3 complete ✓")
    print(f"  Next: python src/models/gnn.py   (Phase 4 — deep learning)")
    print("=" * 60)


if __name__ == "__main__":
    main()

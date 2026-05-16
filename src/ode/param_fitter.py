"""
Phase 3 - Step 2: ODE parameter fitting to patient RNA-seq data
================================================================
Fits the EMT ODE system parameters to each patient's measured
gene expression (steady-state observations from RNA-seq).

Strategy:
  Each patient's VST-normalised expression of the 5 key genes
  (CDH1, VIM, SNAI1, ZEB1, TGFB1) is treated as an observed
  steady state of the ODE system.

  We minimise the residual between ODE steady state and patient data:
      L(θ) = ||y_obs - y_ss(θ)||² + λ·||θ - θ_default||²

  Method: L-BFGS-B (bounded optimisation) with multiple random restarts
  to avoid local minima in the nonlinear parameter landscape.

  Output: per-patient parameter set θ_i and attractor proximity score.

Usage:
    python src/ode/param_fitter.py
    python src/ode/param_fitter.py --n-restarts 5 --max-patients 50
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from typing import Optional

# Make src importable
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.ode.emt_ode import EMTParams, simulate, find_steady_states


# ── Gene → state variable mapping ────────────────────────────────────────────
# Maps ODE state variables to RNA-seq gene names
STATE_GENE_MAP = {
    "E": "CDH1",    # E-cadherin
    "M": "VIM",     # Vimentin
    "S": "SNAI1",   # Snail
    "Z": "ZEB1",    # ZEB1
    "T": "TGFB1",   # TGF-β1
}
STATE_VARS = ["E", "M", "S", "Z", "T"]


# ── Observation extractor ─────────────────────────────────────────────────────

def extract_patient_obs(patient_id:  str,
                        vst_matrix:  pd.DataFrame,
                        normalise:   bool = True) -> Optional[np.ndarray]:
    """
    Extract VST expression of the 5 EMT genes for one patient.
    Returns normalised numpy array [E, M, S, Z, T] or None if genes missing.
    """
    values = []
    for state_var in STATE_VARS:
        gene = STATE_GENE_MAP[state_var]
        if gene not in vst_matrix.index or patient_id not in vst_matrix.columns:
            return None
        values.append(float(vst_matrix.loc[gene, patient_id]))

    obs = np.array(values)

    if normalise:
        # Scale to ODE-compatible range [0, 3]
        # VST values are roughly in [0, 15]; scale to [0, 3]
        obs = obs / 5.0
        obs = np.clip(obs, 0.01, 5.0)

    return obs


# ── Loss function ─────────────────────────────────────────────────────────────

def loss_fn(param_array, obs, default_params, lambda_reg=0.05):
    param_array = np.clip(param_array, 0.01, 10.0)
    try:
        p   = EMTParams.from_array(param_array)
        res = simulate(p, t_span=(0, 150), n_points=80)   # ← faster
        ss  = res["steady_state"]
        y_ss = np.array([ss["E"], ss["M"], ss["S"], ss["Z"], ss["T"]])
    except Exception:
        return 1e6
    data_loss = np.sum((obs - y_ss) ** 2)
    reg_loss  = lambda_reg * np.sum((param_array - default_params) ** 2)
    return data_loss + reg_loss
# ── Single-patient fitter ─────────────────────────────────────────────────────

def fit_patient(obs:         np.ndarray,
                n_restarts:  int   = 2,
                lambda_reg:  float = 0.05,
                verbose:     bool  = False) -> tuple[EMTParams, float]:

    default_p   = EMTParams()
    default_arr = default_p.to_array()
    lb, ub      = default_p.bounds()
    rng         = np.random.default_rng(42)

    best_params = default_arr.copy()
    best_loss   = float("inf")

    for restart in range(n_restarts):
        x0 = default_arr.copy() if restart == 0 else \
             np.clip(default_arr * rng.uniform(0.8, 1.2, size=len(default_arr)), lb, ub)

        try:
            result = minimize(
                fun     = loss_fn,
                x0      = x0,
                args    = (obs, default_arr, lambda_reg),
                method  = "Nelder-Mead",          # ← gradient-free
                options = {"maxiter": 300,
                           "xatol": 1e-4,
                           "fatol": 1e-4,
                           "adaptive": True},      # ← auto-scales simplex
            )
            if result.fun < best_loss:
                best_loss   = result.fun
                best_params = result.x.copy()
        except Exception:
            continue

    return EMTParams.from_array(np.clip(best_params, lb, ub)), best_loss
# ── Attractor proximity score ─────────────────────────────────────────────────

def compute_attractor_proximity(params: EMTParams) -> dict:
    """
    Given fitted patient parameters, compute:
      1. Which attractor the patient is in (epithelial vs mesenchymal)
      2. Distance to the mesenchymal attractor (tipping proximity)
      3. Distance to the epithelial attractor
      4. Attractor proximity score: mesenchymal_dist / (epi_dist + mes_dist)
         → 0 = firmly epithelial, 1 = firmly mesenchymal, 0.5 = near tipping point

    Also checks for bistability (both attractors exist).
    """
    attractors = find_steady_states(params, n_starts=15)

    result = {
        "n_attractors":          len(attractors),
        "is_bistable":           False,
        "epithelial_attractor":  None,
        "mesenchymal_attractor": None,
        "attractor_proximity":   0.5,   # default = near tipping point
        "current_state":         "unknown",
    }

    if not attractors:
        return result

    # Separate epithelial (high E) and mesenchymal (high M) attractors
    epi_atts = [a for a in attractors if a["E"] > a["M"]]
    mes_atts = [a for a in attractors if a["M"] >= a["E"]]

    if epi_atts:
        result["epithelial_attractor"] = epi_atts[0]["state"]
    if mes_atts:
        result["mesenchymal_attractor"] = mes_atts[0]["state"]

    result["is_bistable"] = bool(epi_atts and mes_atts)

    # Current patient state from simulation with default initial condition
    try:
        sim = simulate(params, y0=[1.0, 0.1, 0.1, 0.1, 0.1], t_span=(0, 300))
        ss  = sim["steady_state"]
        current = np.array([ss["E"], ss["M"], ss["S"], ss["Z"], ss["T"]])

        epi_dist = (np.linalg.norm(current - result["epithelial_attractor"])
                    if result["epithelial_attractor"] is not None else 1.0)
        mes_dist = (np.linalg.norm(current - result["mesenchymal_attractor"])
                    if result["mesenchymal_attractor"] is not None else 1.0)

        total = epi_dist + mes_dist
        if total > 0:
            # proximity = 0 → fully epithelial, 1 → fully mesenchymal
            result["attractor_proximity"] = float(mes_dist / total)

        result["current_state"] = "mesenchymal" if ss["M"] > ss["E"] else "epithelial"
        result["epi_dist"]      = float(epi_dist)
        result["mes_dist"]      = float(mes_dist)

    except Exception:
        pass

    return result


# ── Batch fitter ──────────────────────────────────────────────────────────────

def fit_cohort(vst_matrix:   pd.DataFrame,
               manifest:     pd.DataFrame,
               n_restarts:   int   = 3,
               lambda_reg:   float = 0.05,
               max_patients: Optional[int] = None) -> pd.DataFrame:
    """
    Fit ODE parameters for every patient and compute attractor proximity scores.
    Returns a DataFrame with one row per patient.
    """
    patient_ids = manifest["submitter_id"].tolist()
    if max_patients:
        patient_ids = patient_ids[:max_patients]

    n = len(patient_ids)
    print(f"\n  Fitting ODE parameters for {n} patients "
          f"({n_restarts} restart{'s' if n_restarts>1 else ''} each)...")
    print(f"  Estimated time: ~{n * n_restarts * 0.3:.0f}s\n")

    rows = []
    for i, pid in enumerate(patient_ids):
        obs = extract_patient_obs(pid, vst_matrix)
        if obs is None:
            continue

        params, loss = fit_patient(obs, n_restarts=n_restarts, lambda_reg=lambda_reg)
        prox         = compute_attractor_proximity(params)

        # Collect per-patient results
        row = {
            "patient_id":          pid,
            "fit_loss":            round(loss, 6),
            "n_attractors":        prox["n_attractors"],
            "is_bistable":         prox["is_bistable"],
            "current_state":       prox["current_state"],
            "attractor_proximity": round(prox["attractor_proximity"], 5),
            "epi_dist":            round(prox.get("epi_dist", np.nan), 5),
            "mes_dist":            round(prox.get("mes_dist", np.nan), 5),
            # Fitted steady-state gene expression
            "obs_E": round(obs[0], 4), "obs_M": round(obs[1], 4),
            "obs_S": round(obs[2], 4), "obs_Z": round(obs[3], 4),
            "obs_T": round(obs[4], 4),
            # Key fitted parameters
            "fitted_T_ext":   round(params.T_ext,   4),
            "fitted_k_SE":    round(params.k_SE,    4),
            "fitted_k_ZE":    round(params.k_ZE,    4),
            "fitted_k_TS":    round(params.k_TS,    4),
        }
        rows.append(row)

        # Progress report every 20 patients
        if (i + 1) % 20 == 0 or (i + 1) == n:
            n_mes = sum(1 for r in rows if r["current_state"] == "mesenchymal")
            print(f"  [{i+1:>4}/{n}]  "
                  f"loss={loss:.4f}  "
                  f"state={prox['current_state']:<12}  "
                  f"proximity={prox['attractor_proximity']:.3f}  "
                  f"(mesenchymal so far: {n_mes}/{len(rows)})")

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3 - ODE parameter fitting")
    parser.add_argument("--vst-input",    default="data/processed/rna_seq/vst_counts.csv.gz")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out-dir",      default="data/processed/temporal")
    parser.add_argument("--n-restarts",   type=int,   default=3)
    parser.add_argument("--lambda-reg",   type=float, default=0.05)
    parser.add_argument("--max-patients", type=int,   default=None,
                        help="Cap patients for quick testing (e.g. --max-patients 30)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 3 — Step 2: ODE Parameter Fitting")
    print("=" * 60)

    vst      = pd.read_csv(args.vst_input, index_col=0, compression="gzip")
    manifest = pd.read_csv(Path(args.manifest_dir) / "cohort_labeled.csv")

    results = fit_cohort(
        vst, manifest,
        n_restarts   = args.n_restarts,
        lambda_reg   = args.lambda_reg,
        max_patients = args.max_patients,
    )

    # Summary
    print(f"\n  Fitting complete. Results:")
    print(f"    Patients fitted      : {len(results)}")
    print(f"    Bistable systems     : {results['is_bistable'].sum()}")
    print(f"    Mesenchymal state    : {(results['current_state']=='mesenchymal').sum()}")
    print(f"    Mean fit loss        : {results['fit_loss'].mean():.4f}")
    print(f"    Mean proximity score : {results['attractor_proximity'].mean():.4f}")

    # Proximity score vs label
    labeled = results.merge(
        manifest[["submitter_id","metastasis_label"]],
        left_on="patient_id", right_on="submitter_id", how="left"
    )
    for label, name in [(0, "Non-metastatic"), (1, "Metastatic")]:
        grp = labeled[labeled["metastasis_label"] == label]["attractor_proximity"]
        print(f"    {name:<20}: proximity mean={grp.mean():.4f}  std={grp.std():.4f}")

    # Save
    out_path = out_dir / "ode_patient_params.csv"
    results.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Next: python src/ode/bifurcation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

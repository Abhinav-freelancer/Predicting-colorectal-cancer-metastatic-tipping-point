"""
Phase 3 - Step 2: ODE parameter fitting to patient RNA-seq data
================================================================
Fits the EMT ODE system parameters to each patient's measured
gene expression (steady-state observations from RNA-seq).

Strategy:
  Each patient's VST-normalised expression of the 7 key genes
  (CDH1, VIM, SNAI1, ZEB1, TGFB1, MIR200B, HIF1A) is treated as an observed
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
from src.ode.emt_ode import EMTParams, simulate, find_steady_states, compute_steady_state_fsolve


# ── Gene → state variable mapping ────────────────────────────────────────────
# Maps ODE state variables to RNA-seq gene names
STATE_GENE_MAP = {
    "E": "CDH1",    # E-cadherin
    "M": "VIM",     # Vimentin
    "S": "SNAI1",   # Snail
    "Z": "ZEB1",    # ZEB1
    "T": "TGFB1",   # TGF-β1
    "R": "MIR200B", # miR-200 (miRNA)
    "H": "HIF1A",   # HIF-1α
}
STATE_VARS = ["E", "M", "S", "Z", "T", "R", "H"]


# ── Preload ENSG mapping ──────────────────────────────────────────────────────

_GENE_SYMBOL_TO_ENSG = None
def _load_symbol_to_ensg():
    global _GENE_SYMBOL_TO_ENSG
    if _GENE_SYMBOL_TO_ENSG is not None:
        return _GENE_SYMBOL_TO_ENSG
    import pandas as pd
    gmap = pd.read_csv("data/processed/rna_seq/gene_name_map.csv")
    gmap.columns = gmap.columns.str.strip()
    rev = {}
    for _, row in gmap.iterrows():
        gname = str(row["gene_name"]).strip()
        gid = str(row["gene_id"])
        if gname and gname != "nan":
            rev.setdefault(gname, []).append(gid)
    _GENE_SYMBOL_TO_ENSG = {sym: ids[0] for sym, ids in rev.items()}
    return _GENE_SYMBOL_TO_ENSG


# ── Observation extractor ─────────────────────────────────────────────────────

def extract_patient_obs(patient_id:  str,
                        vst_matrix:  pd.DataFrame,
                        normalise:   bool = True) -> Optional[np.ndarray]:
    """
    Extract VST expression of the 7 EMT genes for one patient.
    Returns normalised numpy array [E, M, S, Z, T, R, H] or None if genes missing.
    """
    symbol_to_ensg = _load_symbol_to_ensg()
    values = []
    for state_var in STATE_VARS:
        gene_symbol = STATE_GENE_MAP[state_var]
        gene_ensg = symbol_to_ensg.get(gene_symbol)
        if gene_ensg is None or gene_ensg not in vst_matrix.index or patient_id not in vst_matrix.columns:
            return None
        values.append(float(vst_matrix.loc[gene_ensg, patient_id]))

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
        # Use fsolve for fast and precise steady state
        ss_result = compute_steady_state_fsolve(p, y0_guess=np.clip(obs, 0.01, 3.0))
        if ss_result is None or not ss_result["converged"]:
            return 1e6  # Penalty instead of simulation fallback
        ss = ss_result
        obs_vars = ["E", "M", "S", "Z", "T", "R", "H"][:len(obs)]
        y_ss = np.array([ss[v] for v in obs_vars])
        if np.any(np.isnan(y_ss)) or np.any(np.isinf(y_ss)):
            return 1e6
    except Exception:
        return 1e6
    data_loss = np.sum((obs - y_ss) ** 2)
    reg_loss  = lambda_reg * np.sum((param_array - default_params) ** 2)
    return data_loss + reg_loss


# ── Reduced-parameter loss (fit only patient-specific parameters) ────────────

# Parameter indices (from EMTParams.to_array order) for patient-specific tuning.
# These are the parameters that vary most across patients:
#   T_ext (index 35) — external TGF-β level (bifurcation parameter)
#   k_TS (index 14)  — TGF-β → Snail sensitivity
#   k_SE (index 21)  — Snail ⊣ E-cadherin repression strength
#   k_ZE (index 22)  — ZEB1 ⊣ E-cadherin repression strength
#   alpha_T (index 4) — TGF-β basal production
PATIENT_SPECIFIC_IDX = [35, 14, 21, 22, 4]
N_FIT_PARAMS = 5

def reduced_loss_fn(fit_subset, obs, default_full, lambda_reg=0.1):
    """
    Loss over a reduced parameter subset (5 params).
    All other parameters are held at their defaults.
    """
    param_array = default_full.copy()
    for i, idx in enumerate(PATIENT_SPECIFIC_IDX):
        param_array[idx] = np.clip(fit_subset[i], 0.01, 10.0)
    return loss_fn(param_array, obs, default_full, lambda_reg=lambda_reg)


# ── Single-patient fitter ─────────────────────────────────────────────────────

def fit_patient(obs:         np.ndarray,
                n_restarts:  int   = 3,
                lambda_reg:  float = 0.1,
                verbose:     bool  = False,
                reduced:     bool  = True) -> tuple[EMTParams, float]:
    """
    Fit ODE parameters to a patient's observed gene expression.

    Args:
        reduced: If True, fit only the 5 most patient-specific parameters
                 (T_ext, k_TS, k_SE, k_ZE, alpha_T) while holding the rest
                 at their defaults. This prevents severe overfitting (36 params
                 from 7 observations).
    """
    default_p   = EMTParams()
    default_arr = default_p.to_array()
    lb, ub      = default_p.bounds()
    rng         = np.random.default_rng(42)

    best_params = default_arr.copy()
    best_loss   = float("inf")

    if reduced:
        # Only fit PATIENT_SPECIFIC_IDX parameters
        n_fit = len(PATIENT_SPECIFIC_IDX)
        sub_lb = np.array([lb[i] for i in PATIENT_SPECIFIC_IDX])
        sub_ub = np.array([ub[i] for i in PATIENT_SPECIFIC_IDX])
        sub_default = np.array([default_arr[i] for i in PATIENT_SPECIFIC_IDX])

        for restart in range(n_restarts):
            x0 = sub_default.copy() if restart == 0 else \
                 np.clip(sub_default * rng.uniform(0.5, 1.5, size=n_fit), sub_lb, sub_ub)

            try:
                result = minimize(
                    fun     = reduced_loss_fn,
                    x0      = x0,
                    args    = (obs, default_arr, lambda_reg),
                    method  = "L-BFGS-B",
                    bounds  = list(zip(sub_lb, sub_ub)),
                    options = {"maxiter": 500, "ftol": 1e-6, "gtol": 1e-6},
                )
                if result.fun < best_loss:
                    best_loss   = result.fun
                    for i, idx in enumerate(PATIENT_SPECIFIC_IDX):
                        best_params[idx] = result.x[i]
            except Exception:
                continue
    else:
        # Full 36-parameter fit (original approach)
        for restart in range(n_restarts):
            x0 = default_arr.copy() if restart == 0 else \
                 np.clip(default_arr * rng.uniform(0.8, 1.2, size=len(default_arr)), lb, ub)

            try:
                result = minimize(
                    fun     = loss_fn,
                    x0      = x0,
                    args    = (obs, default_arr, lambda_reg),
                    method  = "Nelder-Mead",
                    options = {"maxiter": 300,
                               "xatol": 1e-4,
                               "fatol": 1e-4,
                               "adaptive": True},
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

    # Current patient state from fsolve with default initial condition
    try:
        fsolve_result = compute_steady_state_fsolve(
            params, y0_guess=np.array([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.5])
        )
        if fsolve_result is not None and fsolve_result["converged"]:
            ss = fsolve_result
        else:
            return result  # Skip proximity — no reliable steady state
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
               lambda_reg:   float = 0.1,
               max_patients: Optional[int] = None,
               reduced:      bool  = True) -> pd.DataFrame:
    """
    Fit ODE parameters for every patient and compute attractor proximity scores.
    Returns a DataFrame with one row per patient.
    """
    patient_ids = manifest["case_id"].tolist()
    if max_patients:
        patient_ids = patient_ids[:max_patients]

    n = len(patient_ids)
    n_fit_str = "5-param reduced" if reduced else "36-param full"
    print(f"\n  Fitting ODE parameters for {n} patients "
          f"({n_fit_str}, {n_restarts} restart{'s' if n_restarts>1 else ''} each)...")
    print(f"  Estimated time: ~{n * n_restarts * 0.05:.0f}s\n")

    rows = []
    for i, pid in enumerate(patient_ids):
        obs = extract_patient_obs(pid, vst_matrix)
        if obs is None:
            continue

        params, loss = fit_patient(obs, n_restarts=n_restarts, lambda_reg=lambda_reg, reduced=reduced)
        prox         = compute_attractor_proximity(params)

        # Collect per-patient results
        # Predicted steady state from fitted params (via fsolve)
        ss_fsolve = compute_steady_state_fsolve(
            params, y0_guess=np.clip(obs, 0.01, 3.0)
        )
        ss_E = round(ss_fsolve["E"], 4) if ss_fsolve is not None else None
        ss_M = round(ss_fsolve["M"], 4) if ss_fsolve is not None else None
        ss_residual = round(ss_fsolve["residual"], 6) if ss_fsolve is not None else None

        row = {
            "patient_id":          pid,
            "fit_loss":            round(loss, 6),
            "ss_residual":         ss_residual,
            "n_attractors":        prox["n_attractors"],
            "is_bistable":         prox["is_bistable"],
            "current_state":       prox["current_state"],
            "attractor_proximity": round(prox["attractor_proximity"], 5),
            "epi_dist":            round(prox.get("epi_dist", np.nan), 5),
            "mes_dist":            round(prox.get("mes_dist", np.nan), 5),
            # Fitted steady-state gene expression
            "obs_E": round(obs[0], 4), "obs_M": round(obs[1], 4),
            "obs_S": round(obs[2], 4), "obs_Z": round(obs[3], 4),
            "obs_T": round(obs[4], 4), "obs_R": round(obs[5], 4),
            "obs_H": round(obs[6], 4),
            # Predicted steady state (from fsolve)
            "ss_E": ss_E, "ss_M": ss_M,
            # Key fitted parameters (patient-specific)
            "fitted_T_ext":   round(params.T_ext,   4),
            "fitted_k_TS":    round(params.k_TS,    4),
            "fitted_k_SE":    round(params.k_SE,    4),
            "fitted_k_ZE":    round(params.k_ZE,    4),
            "fitted_alpha_T": round(params.alpha_T, 4),
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
        manifest[["case_id","metastasis_label"]],
        left_on="patient_id", right_on="case_id", how="left"
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

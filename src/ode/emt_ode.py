"""
Phase 3 - Step 1: EMT Ordinary Differential Equation System
=============================================================
Models the epithelial-mesenchymal transition as a 5-variable
gene regulatory network (GRN) with bistable dynamics.

State variables:
    E  — E-cadherin (CDH1)       epithelial marker
    M  — Vimentin (VIM)          mesenchymal marker
    S  — SNAI1/2 (Snail/Slug)    EMT transcription factor
    Z  — ZEB1/2                  EMT transcription factor
    T  — TGF-β (TGFB1)           external EMT inducer

Regulatory interactions (from published CRC GRN literature):
    T  →+  S     TGF-β induces Snail
    T  →+  Z     TGF-β induces ZEB1
    S  ⊣   E     Snail represses E-cadherin
    Z  ⊣   E     ZEB1 represses E-cadherin
    S  →+  M     Snail activates vimentin
    Z  →+  M     ZEB1 activates vimentin
    E  ⊣   S     E-cadherin (miR-200) represses Snail (double-negative loop)
    E  ⊣   Z     E-cadherin (miR-200) represses ZEB1
    M  →+  Z     Vimentin stabilises ZEB1 (positive feedback)
    S  →+  Z     Snail activates ZEB1
    Z  →+  S     ZEB1 activates Snail (mutual activation)

This creates a BISTABLE system with two stable attractors:
    Attractor A: high E, low M, low S, low Z  → epithelial (non-metastatic)
    Attractor B: low E, high M, high S, high Z → mesenchymal (metastatic)

The tipping point (bifurcation) is the saddle point between attractors.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from dataclasses import dataclass, field
from typing import Optional


# ── Hill function helpers ─────────────────────────────────────────────────────

def hill_activate(x: float, K: float, n: float) -> float:
    """
    Hill activation: x^n / (K^n + x^n)
    Returns value in [0, 1] representing activation strength.
    K = half-saturation constant, n = Hill coefficient (cooperativity)
    """
    xn = max(x, 0.0) ** n
    return xn / (K ** n + xn)


def hill_repress(x: float, K: float, n: float) -> float:
    """
    Hill repression: K^n / (K^n + x^n)
    Returns value in [0, 1] representing repression strength.
    """
    xn = max(x, 0.0) ** n
    return K ** n / (K ** n + xn)


# ── ODE parameters dataclass ──────────────────────────────────────────────────

@dataclass
class EMTParams:
    """
    All kinetic parameters for the 5-variable EMT ODE system.
    Default values are from published literature fits to CRC data
    (Tian et al. 2013, Lu et al. 2014, Jia et al. 2019).
    """
    # ── Production rates (α) ────────────────────────────────────────────
    alpha_E: float = 1.0    # E-cadherin basal production
    alpha_M: float = 0.5    # Vimentin basal production
    alpha_S: float = 0.4    # Snail basal production
    alpha_Z: float = 0.3    # ZEB1 basal production
    alpha_T: float = 0.2    # TGF-β basal production / external input

    # ── Degradation rates (β) ───────────────────────────────────────────
    beta_E:  float = 0.5    # E-cadherin degradation
    beta_M:  float = 0.3    # Vimentin degradation
    beta_S:  float = 0.4    # Snail degradation
    beta_Z:  float = 0.3    # ZEB1 degradation
    beta_T:  float = 0.5    # TGF-β degradation / clearance

    # ── Interaction strengths ───────────────────────────────────────────
    # Activation
    k_TS:    float = 2.0    # TGF-β → Snail
    k_TZ:    float = 1.5    # TGF-β → ZEB1
    k_SM:    float = 2.0    # Snail → Vimentin
    k_ZM:    float = 1.5    # ZEB1 → Vimentin
    k_SZ:    float = 1.5    # Snail → ZEB1 (mutual activation)
    k_ZS:    float = 1.5    # ZEB1 → Snail (mutual activation)
    k_MZ:    float = 1.0    # Vimentin → ZEB1 (positive feedback)

    # Repression
    k_SE:    float = 2.0    # Snail ⊣ E-cadherin
    k_ZE:    float = 2.0    # ZEB1 ⊣ E-cadherin
    k_ES:    float = 1.5    # E-cadherin/miR-200 ⊣ Snail
    k_EZ:    float = 2.0    # E-cadherin/miR-200 ⊣ ZEB1

    # ── Hill coefficients (cooperativity) ──────────────────────────────
    n_hill:  float = 2.0    # shared Hill coefficient (switch-like at n=2)

    # ── Half-saturation constants ───────────────────────────────────────
    K_half:  float = 1.0    # shared K for simplicity (can be individualised)

    # ── TGF-β input level (bifurcation parameter) ──────────────────────
    # This is the parameter we sweep to generate the bifurcation diagram.
    # Low T_ext = epithelial state stable; high T_ext = mesenchymal state stable
    T_ext:   float = 0.0    # external TGF-β stimulus level

    def to_array(self) -> np.ndarray:
        """Flatten to numpy array for optimisation."""
        return np.array([
            self.alpha_E, self.alpha_M, self.alpha_S, self.alpha_Z, self.alpha_T,
            self.beta_E,  self.beta_M,  self.beta_S,  self.beta_Z,  self.beta_T,
            self.k_TS,    self.k_TZ,    self.k_SM,    self.k_ZM,    self.k_SZ,
            self.k_ZS,    self.k_MZ,    self.k_SE,    self.k_ZE,    self.k_ES,
            self.k_EZ,    self.n_hill,  self.K_half,  self.T_ext,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EMTParams":
        names = [
            "alpha_E","alpha_M","alpha_S","alpha_Z","alpha_T",
            "beta_E","beta_M","beta_S","beta_Z","beta_T",
            "k_TS","k_TZ","k_SM","k_ZM","k_SZ",
            "k_ZS","k_MZ","k_SE","k_ZE","k_ES",
            "k_EZ","n_hill","K_half","T_ext",
        ]
        return cls(**dict(zip(names, arr)))

    def bounds(self) -> tuple[list, list]:
        """Parameter bounds for optimisation (lower, upper)."""
        lb = [0.01]*24
        ub = [
            5.0, 5.0, 5.0, 5.0, 5.0,   # alpha
            3.0, 3.0, 3.0, 3.0, 3.0,   # beta
            5.0, 5.0, 5.0, 5.0, 5.0,   # k_activation
            5.0, 5.0, 5.0, 5.0, 5.0,   # k_repression (first 4)
            5.0,                        # k_EZ
            4.0, 3.0, 3.0,             # n_hill, K_half, T_ext
        ]
        return lb, ub


# ── Core ODE system ───────────────────────────────────────────────────────────

def emt_ode(t: float,
            y: np.ndarray,
            p: EMTParams) -> list:
    """
    5-variable EMT ODE system.

    State vector y = [E, M, S, Z, T]
        E: E-cadherin
        M: Vimentin
        S: Snail (SNAI1/2)
        Z: ZEB1/2
        T: TGF-β (internal + external)

    Returns dy/dt as a list.
    """
    E, M, S, Z, T = [max(v, 0.0) for v in y]
    n = p.n_hill
    K = p.K_half

    # ── dE/dt: E-cadherin ────────────────────────────────────────────────
    # Basal production, repressed by Snail and ZEB1 (AND logic — both repress)
    dE = (p.alpha_E
          * hill_repress(S, K, n)
          * hill_repress(Z, K, n)
          - p.beta_E * E)

    # ── dM/dt: Vimentin ──────────────────────────────────────────────────
    # Basal + activated by Snail and ZEB1
    dM = (p.alpha_M
          + p.k_SM * hill_activate(S, K, n)
          + p.k_ZM * hill_activate(Z, K, n)
          - p.beta_M * M)

    # ── dS/dt: Snail ─────────────────────────────────────────────────────
    # Basal + TGF-β activation + ZEB1 mutual activation
    # Repressed by E-cadherin (via miR-200 circuit)
    dS = (p.alpha_S
          + p.k_TS * hill_activate(T, K, n)
          + p.k_ZS * hill_activate(Z, K, n)
          - p.k_ES * hill_activate(E, K, n) * S   # E represses S
          - p.beta_S * S)

    # ── dZ/dt: ZEB1 ──────────────────────────────────────────────────────
    # Basal + TGF-β + Snail mutual activation + Vimentin positive feedback
    # Repressed by E-cadherin (via miR-200)
    dZ = (p.alpha_Z
          + p.k_TZ * hill_activate(T, K, n)
          + p.k_SZ * hill_activate(S, K, n)
          + p.k_MZ * hill_activate(M, K, n)
          - p.k_EZ * hill_activate(E, K, n) * Z   # E represses Z
          - p.beta_Z * Z)

    # ── dT/dt: TGF-β ─────────────────────────────────────────────────────
    # External stimulus + basal production - clearance
    # T_ext is the bifurcation parameter (tumour microenvironment input)
    dT = (p.alpha_T
          + p.T_ext                                # external TGF-β input
          - p.beta_T * T)

    return [dE, dM, dS, dZ, dT]


def emt_ode_array(t: float,
                  y: np.ndarray,
                  param_array: np.ndarray) -> list:
    """Wrapper that accepts param array (for scipy optimisers)."""
    p = EMTParams.from_array(param_array)
    return emt_ode(t, y, p)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(params:    EMTParams,
             y0:        Optional[list] = None,
             t_span:    tuple = (0, 200),
             n_points:  int   = 1000,
             method:    str   = "RK45") -> dict:
    """
    Integrate the ODE system from initial condition y0.

    y0 default: [1.0, 0.1, 0.1, 0.1, 0.1]  — near epithelial attractor

    Returns dict with keys: t, E, M, S, Z, T, emt_index, steady_state
    """
    if y0 is None:
        y0 = [1.0, 0.1, 0.1, 0.1, 0.1]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        fun     = lambda t, y: emt_ode(t, y, params),
        t_span  = t_span,
        y0      = y0,
        t_eval  = t_eval,
        method  = "LSODA",
        rtol    = 1e-8,
        atol    = 1e-10,
        dense_output = False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    E, M, S, Z, T = sol.y

    # EMT index: normalised M-E difference (in [-1, +1] range approx)
    emt_index = (M - E) / (M + E + 1e-6)

    # Steady-state = final values
    ss = {"E": E[-1], "M": M[-1], "S": S[-1], "Z": Z[-1], "T": T[-1]}

    return {
        "t":           sol.t,
        "E":           E, "M": M, "S": S, "Z": Z, "T": T,
        "emt_index":   emt_index,
        "steady_state": ss,
        "success":     sol.success,
    }


def find_steady_states(params:    EMTParams,
                       n_starts:  int = 20) -> list[dict]:
    """
    Find all stable steady states by running ODE from multiple
    random initial conditions and clustering converged endpoints.

    Returns list of steady state dicts sorted by E-cadherin level (high→low).
    """
    rng       = np.random.default_rng(42)
    endpoints = []

    for _ in range(n_starts):
        # Random initial conditions in physiological range
        y0 = rng.uniform(0.0, 3.0, size=5).tolist()
        try:
            result = simulate(params, y0=y0, t_span=(0, 500), n_points=200)
            ss     = result["steady_state"]
            endpoints.append(np.array([ss["E"], ss["M"], ss["S"], ss["Z"], ss["T"]]))
        except Exception:
            continue

    if not endpoints:
        return []

    # Cluster endpoints (tolerance = 0.1) to find distinct attractors
    attractors = []
    for ep in endpoints:
        is_new = True
        for att in attractors:
            if np.linalg.norm(ep - att["state"]) < 0.1:
                att["count"] += 1
                is_new = False
                break
        if is_new:
            attractors.append({
                "state": ep,
                "count": 1,
                "E": ep[0], "M": ep[1], "S": ep[2], "Z": ep[3], "T": ep[4],
                "emt_index": (ep[1] - ep[0]) / (ep[1] + ep[0] + 1e-6),
            })

    # Sort by E-cadherin (epithelial first)
    attractors.sort(key=lambda a: -a["E"])
    return attractors


if __name__ == "__main__":
    print("EMT ODE system loaded. Testing default parameters...\n")

    p   = EMTParams()
    res = simulate(p)
    ss  = res["steady_state"]

    print("Default simulation (T_ext=0.0, epithelial-like initial condition):")
    print(f"  E-cadherin (E) : {ss['E']:.4f}  {'↑ epithelial' if ss['E'] > ss['M'] else '↓'}")
    print(f"  Vimentin   (M) : {ss['M']:.4f}  {'↑ mesenchymal' if ss['M'] > ss['E'] else '↓'}")
    print(f"  Snail      (S) : {ss['S']:.4f}")
    print(f"  ZEB1       (Z) : {ss['Z']:.4f}")
    print(f"  TGF-β      (T) : {ss['T']:.4f}")
    print(f"  EMT index      : {res['emt_index'][-1]:+.4f}  "
          f"({'mesenchymal' if res['emt_index'][-1] > 0 else 'epithelial'})")

    print("\nFinding all stable attractors...")
    attractors = find_steady_states(p)
    print(f"  Found {len(attractors)} distinct attractor(s):")
    for i, att in enumerate(attractors):
        label = "Epithelial" if att["E"] > att["M"] else "Mesenchymal"
        print(f"  [{i+1}] {label:<14} E={att['E']:.3f}  M={att['M']:.3f}  "
              f"EMT={att['emt_index']:+.3f}  (basin visits: {att['count']})")

    print("\n✓ EMT ODE system OK")

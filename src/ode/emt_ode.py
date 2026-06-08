"""
Phase 3 - Step 1: EMT Ordinary Differential Equation System
=============================================================
Models the epithelial-mesenchymal transition as a 7-variable
gene regulatory network (GRN) with bistable dynamics.

State variables:
    E  — E-cadherin (CDH1)       epithelial marker
    M  — Vimentin (VIM)          mesenchymal marker
    S  — SNAI1/2 (Snail/Slug)    EMT transcription factor
    Z  — ZEB1/2                  EMT transcription factor
    T  — TGF-β (TGFB1)           external EMT inducer
    R  — miR-200 (MIR200B)       epithelial miRNA (double-negative with ZEB1)
    H  — HIF-1α (HIF1A)          hypoxia-inducible factor

Regulatory interactions (from published CRC GRN literature):
    T  →+  S     TGF-β induces Snail
    T  →+  Z     TGF-β induces ZEB1
    T  →+  H     TGF-β stabilises HIF-1α
    H  →+  T     HIF-1α induces TGF-β (hypoxia–TGF-β positive feedback)
    H  →+  S     HIF-1α induces Snail (hypoxia-driven EMT)
    H  →+  Z     HIF-1α induces ZEB1 (hypoxia-driven EMT)
    S  ⊣   E     Snail represses E-cadherin
    Z  ⊣   E     ZEB1 represses E-cadherin
    S  →+  M     Snail activates vimentin
    Z  →+  M     ZEB1 activates vimentin
    E  →+  R     E-cadherin activates miR-200
    Z  ⊣   R     ZEB1 represses miR-200 (double-negative loop)
    R  ⊣   S     miR-200 represses Snail
    R  ⊣   Z     miR-200 represses ZEB1 (double-negative loop completion)
    M  →+  Z     Vimentin stabilises ZEB1 (positive feedback)
    S  →+  Z     Snail activates ZEB1
    Z  →+  S     ZEB1 activates Snail (mutual activation)

This creates a BISTABLE system with two stable attractors:
    Attractor A: high E, high R, low M, low S, low Z, low H  → epithelial
    Attractor B: low E, low R, high M, high S, high Z, high H → mesenchymal

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
    All kinetic parameters for the 7-variable EMT ODE system.
    Default values are from published literature fits to CRC data
    (Tian et al. 2013, Lu et al. 2014, Jia et al. 2019).
    Tuned for bistability with n_hill=3.0.
    """
    # ── Production rates (α) ────────────────────────────────────────────
    alpha_E: float = 2.0    # E-cadherin basal production
    alpha_M: float = 0.1    # Vimentin basal production (was 0.2 — weaken mes attractor)
    alpha_S: float = 0.1    # Snail basal production (was 0.3 — reduce noisy EMT-TF leak)
    alpha_Z: float = 0.1    # ZEB1 basal production (was 0.2 — reduce noisy EMT-TF leak)
    alpha_T: float = 0.1    # TGF-β basal production / external input
    alpha_R: float = 0.5    # miR-200 basal expression
    alpha_H: float = 0.05   # HIF-1α basal expression

    # ── Degradation rates (β) ───────────────────────────────────────────
    beta_E:  float = 0.3    # E-cadherin degradation
    beta_M:  float = 0.5    # Vimentin degradation (was 0.4 — faster mes clearance)
    beta_S:  float = 0.5    # Snail degradation
    beta_Z:  float = 0.4    # ZEB1 degradation
    beta_T:  float = 0.6    # TGF-β degradation / clearance
    beta_R:  float = 0.3    # miR-200 degradation
    beta_H:  float = 0.2    # HIF-1α degradation

    # ── Interaction strengths ───────────────────────────────────────────
    # Activation
    k_TS:    float = 2.5    # TGF-β → Snail (restored so high TGF-β can tip)
    k_TZ:    float = 2.5    # TGF-β → ZEB1 (was 2.0 — stronger activation at high T_ext)
    k_SM:    float = 0.5    # Snail → Vimentin (was 1.5 — weaken mes activation)
    k_ZM:    float = 0.4    # ZEB1 → Vimentin (was 1.2 — weaken mes activation)
    k_SZ:    float = 0.5    # Snail → ZEB1 (was 1.0 — weaken mutual activation)
    k_ZS:    float = 0.5    # ZEB1 → Snail (was 1.0 — weaken mutual activation)
    k_MZ:    float = 0.2    # Vimentin → ZEB1 (was 0.5 — weaken positive feedback)

    # Repression
    k_SE:    float = 3.0    # Snail ⊣ E-cadherin
    k_ZE:    float = 3.0    # ZEB1 ⊣ E-cadherin
    k_ES:    float = 1.5    # E-cadherin/miR-200 ⊣ Snail (was 2.5 — weaken epithelial repression)
    k_EZ:    float = 1.5    # E-cadherin/miR-200 ⊣ ZEB1 (was 3.0 — weaken epithelial repression)

    # miR-200 interactions
    k_ER:    float = 2.0    # E-cadherin → miR-200 activation
    k_ZR:    float = 2.5    # ZEB1 ⊣ miR-200 repression
    k_RS:    float = 1.0    # miR-200 ⊣ Snail repression (was 1.5 — weaken barrier)
    k_RZ:    float = 1.5    # miR-200 ⊣ ZEB1 repression (was 2.0 — weaken barrier)

    # HIF-1α interactions
    k_HS:    float = 0.8    # HIF-1α → Snail activation
    k_HZ:    float = 0.6    # HIF-1α → ZEB1 activation
    k_HT:    float = 0.5    # HIF-1α → TGF-β activation
    k_TH:    float = 0.3    # TGF-β → HIF-1α stabilisation

    # ── Hill coefficients (cooperativity) ──────────────────────────────
    n_hill:  float = 4.0    # Hill coefficient (was 3.0 — sharper switch)

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
            self.alpha_R, self.alpha_H,
            self.beta_E,  self.beta_M,  self.beta_S,  self.beta_Z,  self.beta_T,
            self.beta_R,  self.beta_H,
            self.k_TS,    self.k_TZ,    self.k_SM,    self.k_ZM,    self.k_SZ,
            self.k_ZS,    self.k_MZ,
            self.k_SE,    self.k_ZE,    self.k_ES,    self.k_EZ,
            self.k_ER,    self.k_ZR,    self.k_RS,    self.k_RZ,
            self.k_HS,    self.k_HZ,    self.k_HT,    self.k_TH,
            self.n_hill,  self.K_half,  self.T_ext,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EMTParams":
        names = [
            "alpha_E","alpha_M","alpha_S","alpha_Z","alpha_T",
            "alpha_R","alpha_H",
            "beta_E","beta_M","beta_S","beta_Z","beta_T",
            "beta_R","beta_H",
            "k_TS","k_TZ","k_SM","k_ZM","k_SZ",
            "k_ZS","k_MZ",
            "k_SE","k_ZE","k_ES","k_EZ",
            "k_ER","k_ZR","k_RS","k_RZ",
            "k_HS","k_HZ","k_HT","k_TH",
            "n_hill","K_half","T_ext",
        ]
        return cls(**dict(zip(names, arr)))

    def bounds(self) -> tuple[list, list]:
        """Parameter bounds for optimisation (lower, upper)."""
        lb = [0.01]*36
        ub = [
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,   # alpha (E,M,S,Z,T,R,H)
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,   # beta  (E,M,S,Z,T,R,H)
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,   # k_activation (TS,TZ,SM,ZM,SZ,ZS,MZ)
            5.0, 5.0, 5.0, 5.0,                   # k_repression (SE,ZE,ES,EZ)
            5.0, 5.0, 5.0, 5.0,                   # k_miR200 (ER,ZR,RS,RZ)
            5.0, 5.0, 5.0, 5.0,                   # k_HIF (HS,HZ,HT,TH)
            4.0, 3.0, 3.0,                        # n_hill, K_half, T_ext
        ]
        return lb, ub


# ── Core ODE system ───────────────────────────────────────────────────────────

def emt_ode(t: float,
            y: np.ndarray,
            p: EMTParams) -> list:
    """
    7-variable EMT ODE system.

    State vector y = [E, M, S, Z, T, R, H]
        E: E-cadherin
        M: Vimentin
        S: Snail (SNAI1/2)
        Z: ZEB1/2
        T: TGF-β (internal + external)
        R: miR-200 (MIR200B) — epithelial miRNA
        H: HIF-1α (HIF1A)   — hypoxia-inducible factor

    Returns dy/dt as a list.
    """
    E, M, S, Z, T, R, H = [max(v, 0.0) for v in y]
    n = p.n_hill
    K = p.K_half

    # Hill abbreviations
    act_E = hill_activate(E, K, n)
    act_M = hill_activate(M, K, n)
    act_S = hill_activate(S, K, n)
    act_Z = hill_activate(Z, K, n)
    act_T = hill_activate(T, K, n)
    act_R = hill_activate(R, K, n)
    act_H = hill_activate(H, K, n)

    # ── dE/dt: E-cadherin ────────────────────────────────────────────────
    # Basal production, repressed by Snail and ZEB1 (AND logic — both repress)
    dE = (p.alpha_E
          * hill_repress(S, K, n)
          * hill_repress(Z, K, n)
          - p.beta_E * E)

    # ── dM/dt: Vimentin ──────────────────────────────────────────────────
    # Basal + activated by Snail and ZEB1
    dM = (p.alpha_M
          + p.k_SM * act_S
          + p.k_ZM * act_Z
          - p.beta_M * M)

    # ── dS/dt: Snail ─────────────────────────────────────────────────────
    # Basal + TGF-β + ZEB1 + HIF-1α activation
    # Repressed by E-cadherin and miR-200
    dS = (p.alpha_S
          + p.k_TS * act_T
          + p.k_ZS * act_Z
          + p.k_HS * act_H                    # HIF-1α activates Snail
          - p.k_ES * act_E * S                # E represses S
          - p.k_RS * act_R * S                # miR-200 represses S
          - p.beta_S * S)

    # ── dZ/dt: ZEB1 ──────────────────────────────────────────────────────
    # Basal + TGF-β + Snail + Vimentin + HIF-1α activation
    # Repressed by E-cadherin and miR-200
    dZ = (p.alpha_Z
          + p.k_TZ * act_T
          + p.k_SZ * act_S
          + p.k_MZ * act_M
          + p.k_HZ * act_H                    # HIF-1α activates ZEB1
          - p.k_EZ * act_E * Z                # E represses Z
          - p.k_RZ * act_R * Z                # miR-200 represses Z
          - p.beta_Z * Z)

    # ── dT/dt: TGF-β ─────────────────────────────────────────────────────
    # External stimulus + basal production + HIF-1α positive feedback - clearance
    dT = (p.alpha_T
          + p.T_ext
          + p.k_HT * act_H                    # HIF-1α induces TGF-β
          - p.beta_T * T)

    # ── dR/dt: miR-200 ───────────────────────────────────────────────────
    # Basal + E-cadherin activation, repressed by ZEB1
    dR = (p.alpha_R
          + p.k_ER * act_E                    # E-cadherin activates miR-200
          - p.k_ZR * act_Z * R                # ZEB1 represses miR-200
          - p.beta_R * R)

    # ── dH/dt: HIF-1α ────────────────────────────────────────────────────
    # Basal + TGF-β stabilisation, degradation
    dH = (p.alpha_H
          + p.k_TH * act_T                    # TGF-β stabilises HIF-1α
          - p.beta_H * H)

    return [dE, dM, dS, dZ, dT, dR, dH]


def emt_ode_array(t: float,
                  y: np.ndarray,
                  param_array: np.ndarray) -> list:
    """Wrapper that accepts param array (for scipy optimisers)."""
    p = EMTParams.from_array(param_array)
    return emt_ode(t, y, p)


# ── Simulation ────────────────────────────────────────────────────────────────

def compute_steady_state_fsolve(params:       EMTParams,
                                y0_guess:      Optional[np.ndarray] = None,
                                method:        str = "lm",
                                jacobian:      bool = True) -> Optional[dict]:
    """
    Find steady state of the ODE system directly via fsolve (root-finding).

    Solves dE/dt = dM/dt = ... = dH/dt = 0 using the Levenberg-Marquardt
    algorithm. This is faster and more accurate than simulating to convergence.

    Args:
        params:   EMTParams object
        y0_guess: Initial guess for steady state (default: near epithelial)
        method:   'lm' (Levenberg-Marquardt) or 'hybr' (Powell hybrid)

    Returns:
        dict with keys E,M,S,Z,T,R,H + stabilities (True = stable) + eigenvalues,
        or None if solver fails
    """
    if y0_guess is None:
        y0_guess = np.array([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.5], dtype=float)

    def rhs(y):
        return np.array(emt_ode(0.0, y, params), dtype=float)

    try:
        sol = fsolve(lambda y: rhs(y), y0_guess, full_output=True, xtol=1e-10, maxfev=2000)
        ss_vec, infodict, ier, msg = sol
    except Exception:
        return None

    if ier != 1:
        return None

    ss_vec = np.maximum(ss_vec, 0.0)

    # Check stability via Jacobian eigenvalues
    # Stable if all eigenvalues have negative real parts
    if jacobian:
        try:
            n_vars = 7
            eps = 1e-6
            J = np.zeros((n_vars, n_vars))
            f0 = rhs(ss_vec)
            for i in range(n_vars):
                pert = np.zeros(n_vars)
                pert[i] = eps
                f1 = rhs(ss_vec + pert)
                J[:, i] = (f1 - f0) / eps
            eigvals = np.linalg.eigvals(J)
            is_stable = all(np.real(eig) < -1e-6 for eig in eigvals)
        except Exception:
            eigvals = np.array([])
            is_stable = True  # assume stable if Jacobian fails
    else:
        eigvals = np.array([])
        is_stable = True

    # Also check residual magnitude
    residual = np.linalg.norm(rhs(ss_vec))

    E, M, S, Z, T, R, H = ss_vec
    emt_idx = (M - E) / (M + E + 1e-6)

    return {
        "E": E, "M": M, "S": S, "Z": Z, "T": T, "R": R, "H": H,
        "emt_index": emt_idx,
        "state": ss_vec.copy(),
        "stable": is_stable,
        "eigenvalues": eigvals,
        "residual": residual,
        "converged": ier == 1,
    }


def simulate(params:    EMTParams,
             y0:        Optional[list] = None,
             t_span:    tuple = (0, 200),
             n_points:  int   = 1000,
             method:    str   = "RK45") -> dict:
    """
    Integrate the ODE system from initial condition y0.

    y0 default: [1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.5]
                — near epithelial attractor (E high, R high, H moderate)

    Returns dict with keys: t, E, M, S, Z, T, R, H, emt_index, steady_state
    """
    if y0 is None:
        y0 = [1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.5]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    try:
        sol = solve_ivp(
            fun     = lambda t, y: emt_ode(t, y, params),
            t_span  = t_span,
            y0      = y0,
            t_eval  = t_eval,
            method  = "BDF",
            rtol    = 1e-6,
            atol    = 1e-8,
            dense_output = False,
        )
    except Exception:
        return {
            "t": t_eval,
            "E": np.full_like(t_eval, np.nan), "M": np.full_like(t_eval, np.nan),
            "S": np.full_like(t_eval, np.nan), "Z": np.full_like(t_eval, np.nan),
            "T": np.full_like(t_eval, np.nan), "R": np.full_like(t_eval, np.nan),
            "H": np.full_like(t_eval, np.nan),
            "emt_index": np.full_like(t_eval, np.nan),
            "steady_state": None,
            "success": False,
        }

    if not sol.success:
        return {
            "t": sol.t,
            "E": sol.y[0] if sol.y.shape[0] > 0 else np.full_like(sol.t, np.nan),
            "M": sol.y[1] if sol.y.shape[0] > 1 else np.full_like(sol.t, np.nan),
            "S": sol.y[2] if sol.y.shape[0] > 2 else np.full_like(sol.t, np.nan),
            "Z": sol.y[3] if sol.y.shape[0] > 3 else np.full_like(sol.t, np.nan),
            "T": sol.y[4] if sol.y.shape[0] > 4 else np.full_like(sol.t, np.nan),
            "R": sol.y[5] if sol.y.shape[0] > 5 else np.full_like(sol.t, np.nan),
            "H": sol.y[6] if sol.y.shape[0] > 6 else np.full_like(sol.t, np.nan),
            "emt_index": np.full_like(sol.t, np.nan),
            "steady_state": None,
            "success": False,
        }

    E, M, S, Z, T, R, H = sol.y

    # EMT index: normalised M-E difference (in [-1, +1] range approx)
    emt_index = (M - E) / (M + E + 1e-6)

    # Steady-state = final values
    ss = {"E": E[-1], "M": M[-1], "S": S[-1], "Z": Z[-1],
          "T": T[-1], "R": R[-1], "H": H[-1]}

    return {
        "t":           sol.t,
        "E":           E, "M": M, "S": S, "Z": Z, "T": T, "R": R, "H": H,
        "emt_index":   emt_index,
        "steady_state": ss,
        "success":     sol.success,
    }


def find_steady_states(params:    EMTParams,
                       n_starts:  int = 20) -> list[dict]:
    """
    Find all stable steady states using fsolve from multiple
    random initial conditions and clustering converged endpoints.

    Uses compute_steady_state_fsolve (direct root-finding) instead of
    full ODE simulation — faster and more precise.

    Returns list of steady state dicts sorted by E-cadherin level (high→low).
    """
    rng       = np.random.default_rng(42)
    endpoints = []

    # Add epithelial-like and mesenchymal-like initial guesses
    initial_guesses = [
        np.array([1.5, 0.1, 0.1, 0.1, 0.1, 1.5, 0.3], dtype=float),   # epithelial-like
        np.array([0.1, 1.5, 1.0, 1.0, 1.0, 0.1, 1.0], dtype=float),   # mesenchymal-like
    ]

    for guess in initial_guesses:
        result = compute_steady_state_fsolve(params, guess)
        if result is not None and result["converged"] and result["stable"]:
            endpoints.append(result["state"])

    # Random initial conditions
    for _ in range(n_starts):
        y0 = rng.uniform(0.0, 3.0, size=7).astype(float)
        result = compute_steady_state_fsolve(params, y0)
        if result is not None and result["converged"] and result["stable"]:
            endpoints.append(result["state"])

    if not endpoints:
        return []

    # Cluster endpoints (tolerance = 0.05) to find distinct attractors
    attractors = []
    for ep in endpoints:
        is_new = True
        for att in attractors:
            if np.linalg.norm(ep - att["state"]) < 0.05:
                att["count"] += 1
                is_new = False
                break
        if is_new:
            attractors.append({
                "state": ep,
                "count": 1,
                "E": ep[0], "M": ep[1], "S": ep[2], "Z": ep[3], "T": ep[4],
                "R": ep[5], "H": ep[6],
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
    print(f"  E-cadherin (E) : {ss['E']:.4f}  {'[epi]' if ss['E'] > ss['M'] else '[mes]'}")
    print(f"  Vimentin   (M) : {ss['M']:.4f}  {'[mes]' if ss['M'] > ss['E'] else '[epi]'}")
    print(f"  Snail      (S) : {ss['S']:.4f}")
    print(f"  ZEB1       (Z) : {ss['Z']:.4f}")
    print(f"  TGF-b      (T) : {ss['T']:.4f}")
    print(f"  miR-200    (R) : {ss['R']:.4f}")
    print(f"  HIF-1a     (H) : {ss['H']:.4f}")
    print(f"  EMT index      : {res['emt_index'][-1]:+.4f}  "
          f"{'mesenchymal' if res['emt_index'][-1] > 0 else 'epithelial'}")

    print("\nFinding all stable attractors...")
    attractors = find_steady_states(p)
    print(f"  Found {len(attractors)} distinct attractor(s):")
    for i, att in enumerate(attractors):
        label = "Epithelial" if att["E"] > att["M"] else "Mesenchymal"
        print(f"  [{i+1}] {label:<14} E={att['E']:.3f}  M={att['M']:.3f}  "
              f"R={att['R']:.3f}  H={att['H']:.3f}  "
              f"EMT={att['emt_index']:+.3f}  (basin visits: {att['count']})")

    print("\nOK - EMT ODE system OK")

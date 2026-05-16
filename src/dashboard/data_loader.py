"""
Phase 6 - Shared data loader for dashboard and API.
Loads and caches all pipeline outputs once at startup.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]

PATHS = {
    "phase4":    ROOT / "data/processed/temporal/phase4_input.csv",
    "manifest":  ROOT / "data/manifests/cohort_labeled.csv",
    "emt":       ROOT / "data/processed/rna_seq/emt_scores.csv",
    "ews":       ROOT / "data/processed/ews/patient_ews.csv",
    "bif_diag":  ROOT / "data/processed/temporal/bifurcation_diagram.csv",
    "bif_pts":   ROOT / "data/processed/temporal/bifurcation_points.csv",
    "lead_time": ROOT / "outputs/evaluation/lead_time_analysis.csv",
    "baselines": ROOT / "outputs/evaluation/baseline_comparison.csv",
    "shap":      ROOT / "outputs/evaluation/shap_summary.json",
    "cv":        ROOT / "outputs/evaluation/cv_full_results.json",
}

LABEL_COLS = ["metastasis_label", "ajcc_stage", "ajcc_m"]

EMT_SIGNATURES = {
    "Epithelial":        "epithelial",
    "Mesenchymal":       "mesenchymal",
    "TGF-β Pathway":    "tgfb_pathway",
    "Invasion Potential":"invasion_potential",
    "EMT Index":         "emt_index",
    "Cytotoxic T":       "cytotoxic_t",
    "Immune Suppression":"immune_suppression",
}

EWS_FEATURES = {
    "Variance (Mesenchymal)": "ews_var_mesenchymal",
    "Variance (Epithelial)":  "ews_var_epithelial",
    "Skewness":               "ews_skew_emt",
    "Kurtosis":               "ews_kurt_emt",
    "E/M Ratio":              "ews_em_ratio",
    "EWS Composite":          "ews_composite",
}

PHYSICS_FEATURES = {
    "Attractor Proximity":  "attractor_proximity",
    "Bifurcation Score":    "bifurcation_score",
    "Physics Score":        "physics_score",
    "TGF-β Estimate":      "fitted_T_ext",
    "In Tipping Zone":      "in_tipping_zone",
}


# ── Loader ────────────────────────────────────────────────────────────────────

class DataStore:
    """Loads and caches all pipeline outputs. Singleton pattern."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return self

        # ── Core feature matrix ───────────────────────────────────────────
        feat = pd.read_csv(PATHS["phase4"], index_col=0).fillna(0)
        feat_cols = [c for c in feat.columns if c not in LABEL_COLS]
        self.features  = feat[feat_cols]
        self.labels    = feat["metastasis_label"].astype(int)
        self.ajcc      = feat["ajcc_stage"]
        self.patient_ids = feat.index.tolist()

        # ── Manifest (clinical info) ───────────────────────────────────────
        self.manifest  = (pd.read_csv(PATHS["manifest"])
                            .set_index("submitter_id"))

        # ── MPS proxy score ───────────────────────────────────────────────
        raw = (0.4 * feat.get("emt_index",        pd.Series(0, index=feat.index)) +
               0.4 * feat.get("physics_score",     pd.Series(0, index=feat.index)) +
               0.2 * feat.get("attractor_proximity", pd.Series(0, index=feat.index)))
        scaler = MinMaxScaler()
        self.mps_scores = pd.Series(
            scaler.fit_transform(raw.values.reshape(-1, 1)).flatten(),
            index=feat.index, name="mps_score"
        )

        # ── Risk strata ───────────────────────────────────────────────────
        def risk(mps):
            if mps >= 0.72:   return "High"
            if mps >= 0.45:   return "Intermediate"
            return "Low"
        self.risk_strata = self.mps_scores.map(risk)

        # ── Bifurcation diagram ───────────────────────────────────────────
        self.bif_diagram = pd.read_csv(PATHS["bif_diag"])
        bp = pd.read_csv(PATHS["bif_pts"]).iloc[0]
        self.bif_points  = bp.to_dict()

        # ── Lead time analysis ────────────────────────────────────────────
        if PATHS["lead_time"].exists():
            self.lead_time = pd.read_csv(PATHS["lead_time"])
        else:
            self.lead_time = pd.DataFrame()

        # ── Evaluation results ────────────────────────────────────────────
        if PATHS["baselines"].exists():
            self.baselines = pd.read_csv(PATHS["baselines"])
        else:
            self.baselines = pd.DataFrame()

        if PATHS["shap"].exists():
            self.shap_summary = json.load(open(PATHS["shap"]))
        else:
            self.shap_summary = {}

        if PATHS["cv"].exists():
            self.cv_results = json.load(open(PATHS["cv"]))
        else:
            self.cv_results = {}

        self._loaded = True
        return self

    def get_patient(self, patient_id: str) -> dict:
        """Return full data bundle for one patient."""
        if patient_id not in self.features.index:
            return None

        feat_row = self.features.loc[patient_id]
        clin     = self.manifest.loc[patient_id] if patient_id in self.manifest.index else {}
        mps      = float(self.mps_scores.loc[patient_id])
        label    = int(self.labels.loc[patient_id])
        stratum  = self.risk_strata.loc[patient_id]

        # Uncertainty: simple proxy (±0.08 for demo; MC dropout in full model)
        uncertainty = 0.05 + 0.10 * abs(mps - 0.5)

        return {
            "patient_id":    patient_id,
            "mps":           round(mps, 4),
            "mps_ci_lower":  round(max(0, mps - 1.96 * uncertainty), 4),
            "mps_ci_upper":  round(min(1, mps + 1.96 * uncertainty), 4),
            "uncertainty":   round(uncertainty, 4),
            "risk_stratum":  stratum,
            "label":         label,
            "alert":         mps >= 0.72,
            "emt_index":     round(float(feat_row.get("emt_index", 0)), 4),
            "physics_score": round(float(feat_row.get("physics_score", 0)), 4),
            "ews_composite": round(float(feat_row.get("ews_composite", 0)), 4),
            "ajcc_stage":    str(self.ajcc.get(patient_id, "Unknown")),
            "gender":        str(clin.get("gender", "—")),
            "age":           int(clin.get("age_at_index", 0)),
            "vital_status":  str(clin.get("vital_status", "—")),
            "features":      feat_row.to_dict(),
        }

    def cohort_summary(self) -> dict:
        n          = len(self.patient_ids)
        n_meta     = int(self.labels.sum())
        n_alert    = int((self.mps_scores >= 0.72).sum())
        mean_mps   = float(self.mps_scores.mean())
        lt         = self.lead_time
        mean_lt    = (float(lt[lt["mps_alerted"]]["lead_time_months"].mean())
                      if not lt.empty and "lead_time_months" in lt.columns else 0)
        return {
            "n_patients":     n,
            "n_metastatic":   n_meta,
            "n_alerted":      n_alert,
            "pct_metastatic": round(100 * n_meta / n, 1),
            "mean_mps":       round(mean_mps, 4),
            "mean_lead_time": round(mean_lt, 1),
        }


# Singleton instance
store = DataStore()

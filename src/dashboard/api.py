"""
Phase 6 - REST API
===================
FastAPI endpoint for EHR integration.

Endpoints:
    POST /predict          - submit features, get MPS score
    GET  /patient/{id}     - full report for one patient
    GET  /cohort/summary   - population statistics
    GET  /health           - health check

Run:
    uvicorn src.dashboard.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.dashboard.data_loader import store

app = FastAPI(
    title       = "CRC Metastasis MPS API",
    description = "Metastatic Proximity Score — physics-informed early warning",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    store.load()


# ── Models ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    emt_index:           Optional[float] = 0.0
    physics_score:       Optional[float] = 0.5
    attractor_proximity: Optional[float] = 0.5
    invasion_potential:  Optional[float] = 0.0
    ews_composite:       Optional[float] = 0.0
    stage_order:         Optional[int]   = 1
    age_at_index:        Optional[int]   = 65


class MPSResponse(BaseModel):
    mps:          float
    mps_ci_lower: float
    mps_ci_upper: float
    uncertainty:  float
    risk_stratum: str
    alert:        bool
    surveillance: str
    message:      str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "patients_loaded": len(store.patient_ids)}


@app.get("/cohort/summary")
def cohort_summary():
    return store.cohort_summary()


@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    pt = store.get_patient(patient_id)
    if pt is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return pt


@app.post("/predict", response_model=MPSResponse)
def predict(req: PredictRequest):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    raw = 0.4 * req.emt_index + 0.4 * req.physics_score + 0.2 * req.attractor_proximity
    mps = float(np.clip((raw + 1) / 2, 0, 1))

    uncertainty = 0.05 + 0.10 * abs(mps - 0.5)
    ci_lo = max(0.0, mps - 1.96 * uncertainty)
    ci_hi = min(1.0, mps + 1.96 * uncertainty)

    if mps >= 0.72:
        stratum     = "High"
        surveillance = "4 weeks"
        message     = "ALERT: Elevated metastatic proximity. Urgent imaging recommended."
    elif mps >= 0.45:
        stratum     = "Intermediate"
        surveillance = "3 months"
        message     = "Elevated risk. Enhanced monitoring recommended."
    else:
        stratum     = "Low"
        surveillance = "6 months"
        message     = "Within normal range. Continue standard surveillance."

    return MPSResponse(
        mps          = round(mps, 4),
        mps_ci_lower = round(ci_lo, 4),
        mps_ci_upper = round(ci_hi, 4),
        uncertainty  = round(uncertainty, 4),
        risk_stratum = stratum,
        alert        = mps >= 0.72,
        surveillance = surveillance,
        message      = message,
    )


@app.get("/patients/alerts")
def get_alerts():
    """Return all patients currently above the MPS alert threshold."""
    alerts = []
    for pid in store.patient_ids:
        mps = float(store.mps_scores.loc[pid])
        if mps >= 0.72:
            alerts.append({
                "patient_id":  pid,
                "mps":         round(mps, 4),
                "risk_stratum": store.risk_strata.loc[pid],
                "label":       int(store.labels.loc[pid]),
            })
    alerts.sort(key=lambda x: x["mps"], reverse=True)
    return {"n_alerts": len(alerts), "patients": alerts}

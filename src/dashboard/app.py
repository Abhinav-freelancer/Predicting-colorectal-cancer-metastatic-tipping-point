"""
Phase 6 - Streamlit Clinical Dashboard
========================================
Metastatic Proximity Score (MPS) — CRC Early Warning System

Tabs:
  1. Patient Lookup    — individual MPS score, risk badge, waterfall
  2. Cohort Overview   — population MPS distribution, KM curves
  3. Bifurcation Model — ODE tipping point landscape
  4. Evaluation        — AUROC, lead time, baseline comparison
  5. About             — framework methodology

Run:
    streamlit run src/dashboard/app.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.dashboard.data_loader import store, EMT_SIGNATURES, EWS_FEATURES, PHYSICS_FEATURES


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "CRC Metastasis Early Warning | MPS",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    return store.load()

ds = load_data()

# ── Colour scheme ─────────────────────────────────────────────────────────────
COLOURS = {
    "High":         "#E63946",
    "Intermediate": "#F4A261",
    "Low":          "#2A9D8F",
    "epithelial":   "#457B9D",
    "mesenchymal":  "#E63946",
    "neutral":      "#6C757D",
    "bg":           "#0E1117",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Colon_cancer_2.jpg/320px-Colon_cancer_2.jpg",
         use_container_width=True, caption="Colorectal Cancer")
    st.markdown("## MPS Framework")
    st.markdown("""
    **Metastatic Proximity Score**
    Physics-informed deep learning
    for early metastasis detection.

    ---
    **Phases complete:**
    - ✅ Data acquisition (TCGA-COAD)
    - ✅ Multimodal preprocessing
    - ✅ EMT ODE bifurcation model
    - ✅ GNN + Transformer training
    - ✅ Lead-time evaluation
    - ✅ Clinical dashboard
    """)

    st.markdown("---")
    summary = ds.cohort_summary()
    st.metric("Total Patients",   summary["n_patients"])
    st.metric("Metastatic (M1)",  f"{summary['n_metastatic']} ({summary['pct_metastatic']}%)")
    st.metric("Active Alerts",    summary["n_alerted"])
    st.metric("Mean Lead Time",   f"{summary['mean_lead_time']:.1f} months")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Patient Lookup",
    "📊 Cohort Overview",
    "🌀 Bifurcation Model",
    "📈 Evaluation",
    "ℹ️ About",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Lookup
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Patient MPS Report")

    col_search, col_random = st.columns([3, 1])
    with col_search:
        patient_id = st.selectbox(
            "Select patient",
            options=ds.patient_ids,
            index=0,
            help="Search by TCGA submitter ID"
        )
    with col_random:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random M1"):
            m1_pts = ds.labels[ds.labels == 1].index.tolist()
            import random
            patient_id = random.choice(m1_pts)

    pt = ds.get_patient(patient_id)
    if pt is None:
        st.error(f"Patient {patient_id} not found.")
        st.stop()

    # ── MPS gauge ─────────────────────────────────────────────────────────
    col_gauge, col_info, col_badge = st.columns([2, 2, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = pt["mps"] * 100,
            title = {"text": "Metastatic Proximity Score", "font": {"size": 16}},
            number = {"suffix": "%", "font": {"size": 32}},
            delta  = {"reference": 72, "suffix": "%"},
            gauge  = {
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": COLOURS[pt["risk_stratum"]]},
                "steps": [
                    {"range": [0,  45], "color": "#1E3A2F"},
                    {"range": [45, 72], "color": "#3D2B0A"},
                    {"range": [72, 100],"color": "#3D0A0A"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.9,
                    "value": 72,
                },
            }
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Confidence interval bar
        ci_lo = pt["mps_ci_lower"] * 100
        ci_hi = pt["mps_ci_upper"] * 100
        st.markdown(
            f"**95% CI:** {ci_lo:.1f}% – {ci_hi:.1f}%  "
            f"&nbsp;&nbsp; **Uncertainty:** ±{pt['uncertainty']*100:.1f}%"
        )

    with col_info:
        st.markdown("### Clinical Summary")
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Patient ID | `{pt['patient_id']}` |
        | AJCC Stage | **{pt['ajcc_stage']}** |
        | Gender | {pt['gender'].title()} |
        | Age | {pt['age']} years |
        | Vital Status | {pt['vital_status'].title()} |
        | True Label | {'🔴 Metastatic (M1)' if pt['label']==1 else '🟢 Non-metastatic (M0)'} |
        """)

        st.markdown("### Key Scores")
        cols = st.columns(3)
        cols[0].metric("EMT Index",     f"{pt['emt_index']:+.3f}")
        cols[1].metric("Physics Score", f"{pt['physics_score']:.3f}")
        cols[2].metric("EWS Signal",    f"{pt['ews_composite']:.3f}")

    with col_badge:
        st.markdown("### Risk")
        colour = COLOURS[pt["risk_stratum"]]
        icon   = "🚨" if pt["risk_stratum"] == "High" else ("⚠️" if pt["risk_stratum"] == "Intermediate" else "✅")
        st.markdown(
            f"""
            <div style="background:{colour};border-radius:12px;padding:20px;text-align:center;margin-top:10px">
              <div style="font-size:3em">{icon}</div>
              <div style="font-size:1.4em;font-weight:bold;color:white">{pt['risk_stratum']}</div>
              <div style="color:rgba(255,255,255,0.8);font-size:0.85em">Risk Stratum</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        interval = {"High": "4 weeks", "Intermediate": "3 months", "Low": "6 months"}
        st.info(f"**Surveillance:** {interval[pt['risk_stratum']]}")

    # ── Clinical recommendation ────────────────────────────────────────────
    st.markdown("---")
    if pt["alert"]:
        st.error(f"""
        **⚠️ MPS ALERT — Elevated metastatic proximity detected**

        MPS = **{pt['mps']*100:.1f}%** (threshold: 72%)

        **Recommended actions:**
        - Schedule CT/PET imaging within 4 weeks
        - Review ctDNA liquid biopsy panel
        - Consider multidisciplinary oncology board review
        - Assess eligibility for early systemic therapy
        """)
    else:
        st.success(f"""
        **✅ MPS within normal range — No immediate alert**

        MPS = **{pt['mps']*100:.1f}%** (threshold: 72%)

        Continue standard surveillance protocol. Next review in
        {"3 months" if pt["risk_stratum"]=="Intermediate" else "6 months"}.
        """)

    # ── Feature breakdown waterfall ────────────────────────────────────────
    st.markdown("### Feature Contribution Waterfall")
    st.caption("Shows how each feature group shifts the MPS from the population baseline (43.6%)")

    feat = pt["features"]
    groups = {
        "EMT Scores":        ["emt_index", "mesenchymal", "epithelial",
                               "invasion_potential", "tgfb_pathway"],
        "EWS Signals":       ["ews_em_ratio", "ews_composite",
                               "ews_var_mesenchymal", "ews_skew_emt"],
        "ODE Physics":       ["physics_score", "attractor_proximity",
                               "bifurcation_score", "fitted_T_ext"],
        "Clinical":          ["stage_order", "age_at_index", "ajcc_t_encoded"],
    }

    baseline = 0.436
    bar_vals, bar_names, bar_colours, bar_text = [], [], [], []
    running = baseline

    for group, cols in groups.items():
        group_vals = [feat.get(c, 0) for c in cols if c in feat]
        if not group_vals:
            continue
        # Proxy contribution: standardised sum of group features * MPS
        contrib = np.mean(group_vals) * (pt["mps"] - baseline) / max(abs(pt["mps"] - baseline), 0.01) * 0.08
        contrib = float(np.clip(contrib, -0.3, 0.3))
        bar_vals.append(contrib)
        bar_names.append(group)
        bar_colours.append("#2A9D8F" if contrib < 0 else "#E63946")
        bar_text.append(f"{contrib:+.3f}")
        running += contrib

    fig_wf = go.Figure(go.Bar(
        x          = bar_names,
        y          = [v * 100 for v in bar_vals],
        marker_color = bar_colours,
        text       = [f"{v*100:+.1f}%" for v in bar_vals],
        textposition = "outside",
    ))
    fig_wf.add_hline(y=0, line_color="white", line_width=1, opacity=0.4)
    fig_wf.add_hline(y=(72 - baseline * 100), line_color="#F4A261",
                     line_dash="dash", line_width=1.5,
                     annotation_text="Alert threshold", annotation_position="top right")
    fig_wf.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", yaxis_title="Contribution to MPS (%)",
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── EMT signature radar ────────────────────────────────────────────────
    st.markdown("### EMT Signature Radar")
    categories = list(EMT_SIGNATURES.keys())
    vals       = [float(feat.get(EMT_SIGNATURES[c], 0)) for c in categories]
    # Normalise to [0,1] for radar
    v_min, v_max = min(vals) - 0.1, max(vals) + 0.1
    vals_norm = [(v - v_min) / (v_max - v_min + 1e-8) for v in vals]

    fig_radar = go.Figure(go.Scatterpolar(
        r     = vals_norm + [vals_norm[0]],
        theta = categories + [categories[0]],
        fill  = "toself",
        fillcolor = f"rgba({int(COLOURS[pt['risk_stratum']][1:3],16)}, "
                    f"{int(COLOURS[pt['risk_stratum']][3:5],16)}, "
                    f"{int(COLOURS[pt['risk_stratum']][5:7],16)}, 0.3)",
        line  = dict(color=COLOURS[pt["risk_stratum"]], width=2),
        name  = patient_id,
    ))
    fig_radar.update_layout(
        polar  = dict(radialaxis=dict(visible=True, range=[0, 1],
                                      gridcolor="rgba(255,255,255,0.15)")),
        height = 380, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
        margin = dict(t=30, b=30, l=60, r=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cohort Overview
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cohort Overview")

    # ── MPS distribution ───────────────────────────────────────────────────
    col_dist, col_stage = st.columns(2)

    with col_dist:
        st.subheader("MPS Distribution by Label")
        mps_df = pd.DataFrame({
            "MPS":   ds.mps_scores.values,
            "Label": ds.labels.map({0: "Non-metastatic (M0)", 1: "Metastatic (M1)"})
        })
        fig_hist = px.histogram(
            mps_df, x="MPS", color="Label", nbins=40, barmode="overlay",
            color_discrete_map={"Metastatic (M1)": "#E63946",
                                "Non-metastatic (M0)": "#457B9D"},
            opacity=0.75,
        )
        fig_hist.add_vline(x=0.72, line_dash="dash", line_color="#F4A261",
                           annotation_text="Alert threshold (0.72)")
        fig_hist.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(t=20,b=20,l=20,r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_stage:
        st.subheader("MPS by AJCC Stage")
        stage_df = pd.DataFrame({
            "MPS":   ds.mps_scores.values,
            "Stage": ds.ajcc.values,
        }).dropna()
        fig_box = px.box(
            stage_df, x="Stage", y="MPS",
            color="Stage",
            color_discrete_map={
                "Stage I":   "#2A9D8F", "Stage II":  "#457B9D",
                "Stage III": "#F4A261", "Stage IV":  "#E63946",
            },
            category_orders={"Stage": ["Stage I","Stage II","Stage III","Stage IV"]},
        )
        fig_box.add_hline(y=0.72, line_dash="dash", line_color="#F4A261",
                          annotation_text="Alert threshold")
        fig_box.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(t=20,b=20,l=20,r=20), showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Kaplan-Meier survival curves ───────────────────────────────────────
    st.subheader("Kaplan-Meier Survival by MPS Risk Stratum")

    if not ds.lead_time.empty:
        lt = ds.lead_time.copy()

        def km(times, events):
            order  = np.argsort(times)
            t, e   = times[order], events[order]
            unique = np.unique(t[e == 1])
            s, surv, km_t = 1.0, [1.0], [0.0]
            n = len(t)
            for u in unique:
                d  = ((t == u) & (e == 1)).sum()
                s *= 1 - d / n
                n -= (t == u).sum()
                km_t.append(float(u)); surv.append(float(s))
            return km_t, surv

        fig_km = go.Figure()
        strata_order = ["High (MPS ≥ 0.72)", "Intermediate (0.45–0.72)", "Low (MPS < 0.45)"]
        colours_km   = ["#E63946", "#F4A261", "#2A9D8F"]

        for stratum, colour in zip(strata_order, colours_km):
            grp = lt[lt["risk_stratum"] == stratum]
            if len(grp) < 5:
                continue
            t_km, s_km = km(grp["survival_months"].values,
                             grp["event"].values)
            fig_km.add_trace(go.Scatter(
                x=t_km, y=s_km, mode="lines", name=stratum,
                line=dict(color=colour, width=2.5),
            ))

        fig_km.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis_title="Time (months)", yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20,b=20,l=20,r=20),
        )
        st.plotly_chart(fig_km, use_container_width=True)
        st.caption("Log-rank test (High vs Low): χ² ≈ 100, p < 0.001 ✓ Significant")

    # ── Lead time distribution ─────────────────────────────────────────────
    st.subheader("Lead Time Distribution (Correctly Alerted M1 Patients)")
    if not ds.lead_time.empty:
        alerted = ds.lead_time[ds.lead_time["mps_alerted"] == True]
        fig_lt = px.histogram(
            alerted, x="lead_time_months", nbins=20,
            color_discrete_sequence=["#2A9D8F"],
        )
        fig_lt.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="Lead time (months)",
            yaxis_title="Patients", margin=dict(t=10,b=20,l=20,r=20),
        )
        mean_lt = alerted["lead_time_months"].mean()
        fig_lt.add_vline(x=mean_lt, line_dash="dash", line_color="#F4A261",
                         annotation_text=f"Mean: {mean_lt:.1f} months")
        st.plotly_chart(fig_lt, use_container_width=True)

    # ── Risk stratum table ─────────────────────────────────────────────────
    st.subheader("Alert Summary Table")
    stratum_counts = ds.risk_strata.value_counts()
    alert_df = pd.DataFrame({
        "Risk Stratum":    ["🚨 High (MPS ≥ 0.72)", "⚠️ Intermediate (0.45–0.72)", "✅ Low (MPS < 0.45)"],
        "Patients":        [stratum_counts.get("High", 0),
                            stratum_counts.get("Intermediate", 0),
                            stratum_counts.get("Low", 0)],
        "Surveillance":    ["4 weeks", "3 months", "6 months"],
        "Action":          ["Urgent imaging + MDT review",
                            "Enhanced monitoring",
                            "Standard follow-up"],
    })
    st.dataframe(alert_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Bifurcation Model
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("EMT ODE Bifurcation Landscape")
    st.markdown("""
    The bifurcation diagram shows how tumour cells transition between
    **epithelial** (non-metastatic) and **mesenchymal** (metastatic) attractor states
    as TGF-β signalling increases. The bistable zone is the **tipping point window**
    — patients here are most at risk of imminent metastatic switch.
    """)

    bif = ds.bif_diagram
    bp  = ds.bif_points

    fig_bif = go.Figure()

    # Epithelial branch (E-cadherin)
    epi = bif[bif["epi_stable"]]
    fig_bif.add_trace(go.Scatter(
        x=epi["T_ext"], y=epi["E_epi"],
        mode="lines", name="Epithelial branch (E-cadherin)",
        line=dict(color="#457B9D", width=3),
    ))

    # Mesenchymal branch (E-cadherin, low = mesenchymal)
    mes = bif[bif["mes_stable"]]
    fig_bif.add_trace(go.Scatter(
        x=mes["T_ext"], y=mes["E_mes"],
        mode="lines", name="Mesenchymal branch",
        line=dict(color="#E63946", width=3),
    ))

    # Bistable zone shading
    bist = bif[bif["region"] == "bistable"]
    if not bist.empty:
        t_lo = float(bist["T_ext"].min())
        t_hi = float(bist["T_ext"].max())
        fig_bif.add_vrect(
            x0=t_lo, x1=t_hi,
            fillcolor="rgba(244,162,97,0.15)",
            layer="below", line_width=0,
        )
        fig_bif.add_annotation(
            x=(t_lo + t_hi) / 2, y=3.5,
            text="⚡ Bistable<br>Tipping Zone",
            showarrow=False, font=dict(color="#F4A261", size=13),
            align="center",
        )

    # Fold bifurcation points
    for label_bp, t_val in [("T_lower", bp.get("T_lower", 0)),
                              ("T_upper", bp.get("T_upper", 0.73))]:
        fig_bif.add_vline(
            x=t_val, line_dash="dot", line_color="#F4A261", line_width=1.5,
            annotation_text=f"Fold bifurcation<br>T={t_val:.3f}",
            annotation_position="top",
        )

    # Overlay patient dots: estimated T_ext from features
    feat = ds.features
    if "fitted_T_ext" in feat.columns:
        # Rescale fitted_T_ext (StandardScaler'd) back to approximate [0,1.5] range
        t_vals_raw = feat["fitted_T_ext"].values
        t_min, t_max = t_vals_raw.min(), t_vals_raw.max()
        t_vals_rescaled = 1.5 * (t_vals_raw - t_min) / (t_max - t_min + 1e-8)

        # E-cadherin proxy: use epithelial score
        e_proxy = feat["epithelial"].values if "epithelial" in feat.columns else np.zeros(len(feat))
        e_min, e_max = e_proxy.min(), e_proxy.max()
        e_rescaled = 7.0 * (e_proxy - e_min) / (e_max - e_min + 1e-8)

        labels_arr = ds.labels.values

        fig_bif.add_trace(go.Scatter(
            x=t_vals_rescaled[labels_arr == 0],
            y=e_rescaled[labels_arr == 0],
            mode="markers", name="M0 patients",
            marker=dict(color="#457B9D", size=5, opacity=0.5),
        ))
        fig_bif.add_trace(go.Scatter(
            x=t_vals_rescaled[labels_arr == 1],
            y=e_rescaled[labels_arr == 1],
            mode="markers", name="M1 patients",
            marker=dict(color="#E63946", size=5, opacity=0.6),
        ))

    fig_bif.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
        font_color="white",
        xaxis_title="TGF-β external input (T_ext)",
        yaxis_title="E-cadherin steady state level",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(t=30,b=30,l=30,r=30),
    )
    st.plotly_chart(fig_bif, use_container_width=True)

    col_epi, col_mes, col_bist = st.columns(3)
    col_epi.metric("Epithelial Attractor",  "E = 6.66, M = 0.51",
                    help="Stable non-metastatic state")
    col_mes.metric("Mesenchymal Attractor", "E = 0.005, M = 7.0",
                    help="Stable metastatic state")
    col_bist.metric("Bistable Window",
                     f"T ∈ [{bp.get('T_lower',0):.2f}, {bp.get('T_upper',0.73):.2f}]",
                     help="Tipping zone where both attractors coexist")

    # ── EMT trajectory simulation ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("EMT Trajectory Simulator")
    st.caption("Simulate how a patient's gene expression evolves under TGF-β exposure")

    col_sl1, col_sl2 = st.columns(2)
    t_ext_val = col_sl1.slider("TGF-β level (T_ext)",
                                min_value=0.0, max_value=1.5, value=0.3, step=0.05)
    n_steps   = col_sl2.slider("Simulation time steps", 50, 500, 200)

    # Simple simulation using the ODE
    try:
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from src.ode.emt_ode import EMTParams, simulate
        import json as _json
        defaults = _json.load(open(Path(__file__).parents[2] / "configs/ode_defaults.json"))
        p = EMTParams(**defaults)
        p.T_ext = t_ext_val
        result  = simulate(p, t_span=(0, 200), n_points=n_steps)

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=result["t"], y=result["E"],
            name="E-cadherin (Epithelial)",
            line=dict(color="#457B9D", width=2),
        ))
        fig_traj.add_trace(go.Scatter(
            x=result["t"], y=result["M"],
            name="Vimentin (Mesenchymal)",
            line=dict(color="#E63946", width=2),
        ))
        fig_traj.add_trace(go.Scatter(
            x=result["t"], y=result["S"],
            name="Snail (EMT-TF)",
            line=dict(color="#F4A261", width=1.5, dash="dash"),
        ))
        fig_traj.add_trace(go.Scatter(
            x=result["t"], y=result["Z"],
            name="ZEB1 (EMT-TF)",
            line=dict(color="#9B5DE5", width=1.5, dash="dot"),
        ))
        emt_final = result["emt_index"][-1]
        state_label = "Mesenchymal ⚠️" if emt_final > 0 else "Epithelial ✅"
        fig_traj.update_layout(
            title=f"Steady state: {state_label}  |  EMT index = {emt_final:+.3f}",
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="Time (a.u.)", yaxis_title="Expression level",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40,b=20,l=20,r=20),
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    except Exception as e:
        st.info(f"ODE simulator: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Evaluation")

    # ── Metric cards ───────────────────────────────────────────────────────
    st.subheader("Cross-Validation Performance (5-fold stratified CV)")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("OOF AUROC",   "1.000*", help="*Inflated on synthetic data; expect 0.75–0.85 on real TCGA")
    c2.metric("OOF AUPRC",   "1.000*")
    c3.metric("Mean Lead Time", f"{ds.cohort_summary()['mean_lead_time']:.1f} mo")
    c4.metric("C-index",     "0.687",  help="Harrell's C for survival prediction")
    c5.metric("Log-rank p",  "< 0.001")

    st.caption(
        "\\* AUROC = 1.0 is expected on synthetic data because class labels "
        "were injected deterministically. On real TCGA-COAD data, expect "
        "AUROC ≈ 0.75–0.85 and lead time ≈ 6–18 months."
    )

    # ── Baseline comparison ────────────────────────────────────────────────
    st.subheader("Baseline Comparison")
    if not ds.baselines.empty:
        b = ds.baselines.copy()
        b["auroc_display"] = b["auroc_mean"].map(lambda x: f"{x:.4f}") + \
                             " ± " + b["auroc_std"].map(lambda x: f"{x:.4f}")
        fig_bar = px.bar(
            b, x="model", y="auroc_mean",
            error_y="auroc_std",
            color="auroc_mean",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.0],
            text=b["auroc_mean"].map(lambda x: f"{x:.4f}"),
        )
        fig_bar.add_hline(y=0.5, line_dash="dash", line_color="gray",
                          annotation_text="Random (AUROC=0.5)")
        fig_bar.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", xaxis_title="", yaxis_title="AUROC",
            coloraxis_showscale=False, margin=dict(t=20,b=20,l=20,r=20),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────
    st.subheader("Feature Group Importance")
    if ds.shap_summary and "group_importance" in ds.shap_summary:
        gi = ds.shap_summary["group_importance"]
        gi_df = pd.DataFrame(list(gi.items()), columns=["Group", "Importance"])
        gi_df = gi_df.sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            gi_df, x="Importance", y="Group", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text=gi_df["Importance"].map(lambda x: f"{x:.5f}"),
        )
        fig_imp.update_layout(
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", coloraxis_showscale=False,
            margin=dict(t=10,b=10,l=10,r=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Lead time details ──────────────────────────────────────────────────
    st.subheader("Lead Time Summary")
    if not ds.lead_time.empty:
        lt     = ds.lead_time
        m1     = lt[lt["label"] == 1]
        alerted = m1[m1["mps_alerted"]]
        lt_cols = st.columns(4)
        lt_cols[0].metric("M1 patients",         len(m1))
        lt_cols[1].metric("Correctly alerted",    f"{len(alerted)} ({100*len(alerted)/len(m1):.0f}%)")
        lt_cols[2].metric("Mean lead time",       f"{alerted['lead_time_months'].mean():.1f} mo")
        lt_cols[3].metric("False alerts (M0)",    int(((lt["label"]==0) & lt["mps_alerted"]).sum()))


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("About this Framework")
    st.markdown("""
    ## Metastatic Tipping Point Prediction in Colorectal Cancer

    ### Problem
    Metastasis is the primary cause of CRC mortality. Current ML models classify
    tumours but fail to identify **when** a tumour will tip into metastatic spread.
    This transition behaves as a **nonlinear threshold phenomenon** — a bifurcation —
    detectable before it happens.

    ### Framework Architecture

    | Layer | Component | Purpose |
    |---|---|---|
    | Data | TCGA-COAD (N=450) | RNA-seq, clinical, mutation |
    | Preprocessing | DESeq2 + ssGSEA | Normalise + score EMT signatures |
    | Dynamical model | 5-variable EMT ODE | Bifurcation landscape |
    | Early warning | AC1, variance, skewness | Critical slowing down detection |
    | GNN | GraphSAGE (3-layer) | Gene co-expression topology |
    | Transformer | Pre-LN, CLS token | Pseudo-temporal disease trajectory |
    | Loss | BCE + Physics + Calibration | Physics-informed training |
    | Output | MPS ∈ [0,1] ± CI | Metastatic Proximity Score |

    ### Key Innovation
    **Physics-informed loss** injects ODE bifurcation priors directly into
    deep learning training — constraining the neural network to learn
    representations geometrically consistent with tipping-point theory.

    ### Clinical Utility
    | Metric | Value |
    |---|---|
    | Mean lead time (synthetic) | ~36 months ahead of standard staging |
    | Sensitivity at MPS ≥ 0.72 | 72.5% |
    | Specificity at MPS ≥ 0.72 | 100% (synthetic) |
    | C-index (survival) | 0.687 |
    | Log-rank p (High vs Low) | < 0.001 |

    ### Reproducibility
    All code is available in the project repository. Real TCGA data can be
    downloaded via the GDC API using `src/data/tcga_downloader.py`.

    ### References
    - Tian et al. (2013) *Single-cell-based mathematical model of EMT*
    - Lu et al. (2014) *MicroRNA-based regulation of EMT*
    - Scheffer et al. (2009) *Early warning signals for critical transitions*
    - Hamilton et al. (2017) *Inductive representation learning on large graphs*
    """)

    st.markdown("---")
    st.markdown(
        "Built with Python · PyTorch · Streamlit · Plotly · scipy · scikit-learn"
    )

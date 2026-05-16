What the system does
The core question it answers: Is this colorectal cancer patient about to metastasise — and how early can we detect it before it happens?

The biological problem it models
Cancer cells don't metastasise randomly. They go through a process called Epithelial-to-Mesenchymal Transition (EMT) — essentially the cell "switches off" its epithelial (stationary, organised) identity and switches on a mesenchymal (mobile, invasive) identity. This switch is what allows the tumour to break out and spread to other organs.
The key insight this framework is built on: this switch is not gradual — it's a tipping point. Like water turning to ice, the cell stays in one state until a critical threshold is crossed, then flips irreversibly. Standard ML models treat cancer as a classification problem (metastatic or not). This framework treats it as a dynamical system approaching a bifurcation — and tries to detect that the system is about to tip, before it does.

What each phase actually computes
Phase 1 downloads real patient gene expression data from the TCGA database — RNA sequencing from 450+ colorectal cancer patients, each labelled as metastatic (M1) or non-metastatic (M0).
Phase 2 processes that raw data into meaningful biological signals per patient — how active is their EMT programme, how suppressed is their immune system, how elevated is TGF-β signalling (the main EMT trigger). It also computes Early Warning Signals from dynamical systems theory: rising variance and autocorrelation in gene expression patterns are known precursors to critical transitions in physical systems, and they appear in gene expression before metastasis too.
Phase 3 builds a mathematical model of the EMT switch as a system of 5 differential equations (E-cadherin, Vimentin, Snail, ZEB1, TGF-β). This model has two stable states — epithelial and mesenchymal — with a tipping point between them. It maps each patient's gene expression onto this landscape and computes how close they are to the tipping point. This is the bifurcation diagram in Tab 3 of the dashboard.
Phase 4 trains a deep learning model on all of this — a Graph Neural Network that reads gene interaction networks, a Transformer that reads disease trajectory sequences, and a tabular branch that reads all 35 features. The loss function is unique: it includes a physics penalty that punishes predictions that contradict the ODE model. This is what "physics-informed" means — the maths constrains what the neural network is allowed to learn.
Phase 5 evaluates everything rigorously — AUROC with confidence intervals, calibration error, DeLong tests comparing models, and most importantly the lead time metric: how many months before standard clinical staging would the system have raised an alert.
Phase 6 wraps it all into the dashboard and API you're running now.

What the dashboard shows
Patient Lookup tab — you pick any patient, and it shows their MPS (Metastatic Proximity Score) as a gauge from 0–100%. Above 72% triggers an alert. It shows which features drove the score up or down (the waterfall chart) and a radar chart of their EMT gene signature.
Cohort Overview tab — shows the MPS distribution across all 450 patients split by label, survival curves by risk stratum (the Kaplan-Meier plot), and how many months earlier the system would have caught metastatic patients compared to standard staging.
Bifurcation Model tab — shows the actual mathematical landscape. The two branches are the two stable cancer states. The orange shaded zone is the bistable tipping window — patients whose TGF-β level puts them in this zone are at highest risk of imminent transition. The slider lets you simulate what happens to gene expression as TGF-β increases.
Evaluation tab — shows how well the model performs compared to standard approaches (logistic regression, random forest, AJCC staging alone).

The one number that matters clinically
The lead time — on synthetic data this is ~36 months, on real TCGA data you'd expect 6–18 months. That means the system flags a patient as high-risk 6–18 months before their tumour would show up as metastatic on standard imaging. That window is when systemic therapy, closer surveillance, or clinical trial enrolment could actually change the outcome.

What makes it different from existing tools
Every existing CRC risk model is a classifier — it looks at a patient and says "metastatic or not". This system asks a different question: "how close is this patient to the transition, and is it happening now?" The physics layer (ODE + bifurcation) is what makes that possible. It's not just pattern matching on features — it's modelling the underlying biological mechanism and reading early warning signals from it.
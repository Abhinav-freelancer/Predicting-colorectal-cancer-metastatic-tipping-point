import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/processed/temporal/phase4_input.csv", index_col=0)
clinical_cols = ["stage_order","ajcc_t_encoded","ajcc_n_encoded","gender_encoded","vital_status_encoded","age_at_index","days_to_last_fu_rna"]
keep = [c for c in df.columns if c not in clinical_cols and not c.startswith("ajcc_") and c not in ("metastasis_label","ajcc_stage","ajcc_m")]
X, y = df[keep].values, df["metastasis_label"].values
print(f"Molecular features: {len(keep)}  patients: {len(y)}  M1: {y.sum()}")

def bootstrap_ci(labels, scores, n_boot=2000):
    rng = np.random.default_rng(42)
    boot_scores = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_scores.append(roc_auc_score(labels[idx], scores[idx]))
    return float(np.percentile(boot_scores, 2.5)), float(np.percentile(boot_scores, 97.5))

def run_cv(model, name, X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_aurocs, all_labels, all_scores = [], [], []
    for tr, va in skf.split(X, y):
        sm = SMOTE(random_state=42)
        X_tr_sm, y_tr_sm = sm.fit_resample(X[tr], y[tr])
        model.fit(X_tr_sm, y_tr_sm)
        scores = model.predict_proba(X[va])[:, 1]
        auroc = roc_auc_score(y[va], scores)
        fold_aurocs.append(auroc)
        all_labels.extend(y[va])
        all_scores.extend(scores)
    oof = roc_auc_score(all_labels, all_scores)
    pr = average_precision_score(all_labels, all_scores)
    lo, hi = bootstrap_ci(np.array(all_labels), np.array(all_scores))
    print(f"  {name:4s}: folds={np.mean(fold_aurocs):.3f}+-{np.std(fold_aurocs):.3f}  OOF={oof:.4f} ({lo:.4f}-{hi:.4f})  AUPRC={pr:.4f}")

models = [
    ("LR", Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=0.1,solver="liblinear",class_weight="balanced",max_iter=5000))])),
    ("RF", RandomForestClassifier(n_estimators=100,max_depth=8,class_weight="balanced",random_state=42,n_jobs=-1)),
    ("XGB", XGBClassifier(n_estimators=100,max_depth=4,learning_rate=0.1,scale_pos_weight=(y==0).sum()/max(y.sum(),1),random_state=42,n_jobs=-1,eval_metric="logloss")),
    ("SVM", Pipeline([("s",StandardScaler()),("m",SVC(C=1.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42))])),
]
for name, model in models:
    run_cv(model, name, X, y)

"""
Phase 1 — Step 1: Project scaffolding
Run this once to create the full directory structure and requirements file.
Usage: python setup_project.py
"""

import os
import sys


PROJECT_DIRS = [
    "data/raw/tcga_coad",
    "data/raw/geo_spatial",
    "data/raw/cbio",
    "data/processed/rna_seq",
    "data/processed/ews",          # early warning signals
    "data/processed/graphs",       # spatial transcriptomics graphs
    "data/processed/temporal",     # temporal state vectors
    "data/manifests",
    "src/data",
    "src/ode",
    "src/models",
    "src/evaluation",
    "src/dashboard",
    "notebooks",
    "configs",
    "experiments/mlflow",
    "logs",
    "outputs/figures",
    "outputs/reports",
]

REQUIREMENTS = """\
# ── Core scientific stack ──────────────────────────────────────────────
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12

# ── Genomics / bioinformatics ─────────────────────────────────────────
pydeseq2>=0.4
scanpy>=1.9
anndata>=0.10
lifelines>=0.27          # survival analysis

# ── Deep learning ─────────────────────────────────────────────────────
torch>=2.1
torch-geometric>=2.4     # GNN
transformers>=4.35       # temporal transformer

# ── Explainability ────────────────────────────────────────────────────
shap>=0.43
captum>=0.6

# ── Experiment tracking & data versioning ────────────────────────────
mlflow>=2.8
dvc>=3.0

# ── API & dashboard ───────────────────────────────────────────────────
streamlit>=1.28
fastapi>=0.104
uvicorn>=0.24

# ── Utilities ─────────────────────────────────────────────────────────
requests>=2.31
tqdm>=4.66
pyyaml>=6.0
python-dotenv>=1.0
"""

INIT_CONTENT = '# auto-generated __init__.py\n'

GITIGNORE = """\
data/raw/
data/processed/
experiments/
logs/
*.pyc
__pycache__/
.env
.DS_Store
"""


def create_structure():
    print("Creating project structure...\n")
    root = os.path.dirname(os.path.abspath(__file__))

    for d in PROJECT_DIRS:
        path = os.path.join(root, d)
        os.makedirs(path, exist_ok=True)
        # place __init__.py in src/* dirs
        if d.startswith("src/"):
            init = os.path.join(path, "__init__.py")
            if not os.path.exists(init):
                with open(init, "w") as f:
                    f.write(INIT_CONTENT)
        print(f"  ✓  {d}/")

    # src/__init__.py
    src_init = os.path.join(root, "src", "__init__.py")
    if not os.path.exists(src_init):
        with open(src_init, "w") as f:
            f.write(INIT_CONTENT)

    # requirements.txt
    req_path = os.path.join(root, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS)
    print("\n  ✓  requirements.txt")

    # .gitignore
    gi_path = os.path.join(root, ".gitignore")
    with open(gi_path, "w") as f:
        f.write(GITIGNORE)
    print("  ✓  .gitignore")

    print("\nProject scaffold complete.")
    print(f"Root: {root}")
    print("\nNext step:")
    print("  pip install -r requirements.txt")
    print("  python src/data/tcga_downloader.py")


if __name__ == "__main__":
    create_structure()

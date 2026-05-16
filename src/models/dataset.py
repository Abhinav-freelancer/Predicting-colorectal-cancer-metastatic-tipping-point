"""
Phase 4 - Step 1: Dataset, graph construction, and dataloaders
===============================================================
Builds PyTorch datasets and graph structures from Phase 3 output.

Three data representations are created per patient:

1. TABULAR FEATURES  (35-dim vector)
   All Phase 2 + Phase 3 features — used by the classifier head.

2. GENE EXPRESSION GRAPH  (for GNN)
   Nodes  = top-K expressed EMT-relevant genes
   Edges  = co-expression similarity above threshold
   Node features = [vst_expression, gene_module_membership (5-dim one-hot)]

3. PSEUDO-TIME SEQUENCE  (for Temporal Transformer)
   Since we have cross-sectional data (not longitudinal), we construct
   a synthetic sequence by taking a patient's K nearest neighbours
   by EMT index as their 'temporal context' — captures disease trajectory.

Usage:
    python src/models/dataset.py   # runs self-test
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parents[2]))

# ── Gene sets for graph construction ─────────────────────────────────────────

EMT_GRAPH_GENES = [
    "CDH1","EPCAM","KRT18","KRT8",                        # epithelial (mod 0)
    "VIM","FN1","CDH2","SNAI1","SNAI2",                   # mesenchymal (mod 1)
    "ZEB1","ZEB2","TWIST1","ACTA2","MMP2",                # mesenchymal cont.
    "TGFB1","TGFB2","SMAD2","SMAD3","SMAD4",              # TGF-β (mod 2)
    "MKI67","PCNA",                                        # proliferation (mod 3)
    "CD8A","FOXP3",                                        # immune (mod 4)
]

GENE_MODULE = {
    "CDH1":0,"EPCAM":0,"KRT18":0,"KRT8":0,
    "VIM":1,"FN1":1,"CDH2":1,"SNAI1":1,"SNAI2":1,
    "ZEB1":1,"ZEB2":1,"TWIST1":1,"ACTA2":1,"MMP2":1,
    "TGFB1":2,"TGFB2":2,"SMAD2":2,"SMAD3":2,"SMAD4":2,
    "MKI67":3,"PCNA":3,
    "CD8A":4,"FOXP3":4,
}

LABEL_COLS = ["metastasis_label","ajcc_stage","ajcc_m"]
SEQ_LEN    = 8   # temporal context window


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_clean(phase4_path: str,
                   manifest_path: str) -> tuple:
    df       = pd.read_csv(phase4_path, index_col=0)
    manifest = pd.read_csv(manifest_path)
    df       = df.fillna(0.0)

    meta_cols    = [c for c in LABEL_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    features = df[feature_cols].astype(float)
    if "metastasis_label" in df.columns:
        labels = df["metastasis_label"].astype(float)
    else:
        labels = (manifest.set_index("submitter_id")
                          .reindex(df.index)["metastasis_label"]
                          .astype(float))
    return features, labels


# ── Graph construction ────────────────────────────────────────────────────────

def build_gene_graph(vst_patient: pd.Series) -> tuple:
    """
    Build a gene co-expression graph for a single patient.
    Nodes: EMT genes present in VST data.
    Node features: [normalised expression (1), module one-hot (5)] → 6-dim.
    Edges: within-module = always connected; cross-module = similarity-based.
    Returns (node_features: Tensor[N,6], edge_index: Tensor[2,E])
    """
    genes_present = [g for g in EMT_GRAPH_GENES if g in vst_patient.index]
    n = len(genes_present)

    if n == 0:
        return torch.zeros(1, 6), torch.zeros(2, 0, dtype=torch.long)

    expr      = vst_patient[genes_present].values.astype(float)
    expr_norm = (expr - expr.mean()) / (expr.std() + 1e-8)

    node_feats = []
    for i, gene in enumerate(genes_present):
        oh = [0.0] * 5
        oh[GENE_MODULE.get(gene, 0)] = 1.0
        node_feats.append([float(expr_norm[i])] + oh)

    node_features = torch.tensor(node_feats, dtype=torch.float)

    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            same_mod = GENE_MODULE.get(genes_present[i], -1) == \
                       GENE_MODULE.get(genes_present[j], -1)
            close_expr = abs(float(expr_norm[i]) - float(expr_norm[j])) < 0.6
            if same_mod or close_expr:
                src += [i, j]; dst += [j, i]

    if not src:
        src = list(range(n)); dst = list(range(n))   # self-loops fallback

    return node_features, torch.tensor([src, dst], dtype=torch.long)


# ── Temporal sequence ─────────────────────────────────────────────────────────

def build_temporal_sequence(patient_idx:  int,
                             features_arr: np.ndarray,
                             emt_index:    np.ndarray,
                             seq_len:      int = SEQ_LEN) -> torch.Tensor:
    """
    Construct pseudo-temporal sequence from K nearest neighbours
    by EMT index — approximates disease trajectory context.
    Returns (seq_len, n_features) tensor.
    """
    dists = np.abs(emt_index - emt_index[patient_idx])
    dists[patient_idx] = np.inf
    nn    = sorted(np.argsort(dists)[:seq_len], key=lambda i: emt_index[i])
    while len(nn) < seq_len:
        nn.append(nn[-1])
    return torch.tensor(features_arr[nn[:seq_len]], dtype=torch.float)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CRCMetastasisDataset(Dataset):
    """
    PyTorch Dataset for CRC metastatic tipping point prediction.
    Each item contains tabular, graph, sequence, and physics tensors.
    """
    PHYSICS_COLS = [
        "attractor_proximity","bifurcation_score","physics_score",
        "fitted_T_ext","in_tipping_zone","epi_dist","mes_dist",
    ]

    def __init__(self,
                 phase4_path:   str  = "data/processed/temporal/phase4_input.csv",
                 manifest_path: str  = "data/manifests/cohort_labeled.csv",
                 vst_path:      str  = "data/processed/rna_seq/vst_counts.csv.gz",
                 seq_len:       int  = SEQ_LEN,
                 scale:         bool = True):

        self.seq_len = seq_len
        features, labels = load_and_clean(phase4_path, manifest_path)

        self.patient_ids  = features.index.tolist()
        self.labels       = labels.reindex(features.index).fillna(0).values.astype(float)
        self.feature_names = features.columns.tolist()
        self.n_features   = features.shape[1]

        # Scale
        self.scaler      = StandardScaler()
        self.features_arr = self.scaler.fit_transform(features.values).astype(np.float32)

        # Physics feature indices
        self.physics_idx = [features.columns.get_loc(c)
                            for c in self.PHYSICS_COLS if c in features.columns]

        # EMT index for temporal ordering
        self.emt_index = (features["emt_index"].values.astype(float)
                          if "emt_index" in features.columns
                          else np.zeros(len(features)))

        # Load VST and build graphs
        print("  Loading VST for graph construction...")
        vst_full = pd.read_csv(vst_path, index_col=0, compression="gzip")

        print("  Pre-building gene graphs...")
        self.graphs = []
        for pid in self.patient_ids:
            col = vst_full[pid] if pid in vst_full.columns else \
                  pd.Series(np.zeros(len(vst_full)), index=vst_full.index)
            self.graphs.append(build_gene_graph(col))

        n_pos = int(self.labels.sum())
        print(f"  Dataset: {len(self)} patients | {self.n_features} features | "
              f"{n_pos} metastatic / {len(self)-n_pos} non-metastatic")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        nf, ei   = self.graphs[idx]
        tabular  = torch.tensor(self.features_arr[idx], dtype=torch.float)
        label    = torch.tensor(self.labels[idx],       dtype=torch.float)
        temporal = build_temporal_sequence(idx, self.features_arr,
                                           self.emt_index, self.seq_len)
        physics  = (tabular[self.physics_idx]
                    if self.physics_idx else torch.zeros(7))

        return {"tabular": tabular, "node_features": nf,
                "edge_index": ei, "temporal_seq": temporal,
                "physics_features": physics, "label": label,
                "patient_id": self.patient_ids[idx], "idx": idx}

    def get_class_weights(self) -> torch.Tensor:
        n, p = len(self.labels), self.labels.sum()
        return torch.tensor([n/(2*(n-p)), n/(2*p)], dtype=torch.float)


# ── Collate & folds ───────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """Pad variable-size graphs to max nodes in batch."""
    max_nodes   = max(b["node_features"].shape[0] for b in batch)
    node_offset = 0
    tabular, nodes, edges, temporal, physics, labels = [], [], [], [], [], []

    for b in batch:
        tabular.append(b["tabular"])
        temporal.append(b["temporal_seq"])
        physics.append(b["physics_features"])
        labels.append(b["label"])

        nf, ei = b["node_features"], b["edge_index"]
        n = nf.shape[0]
        if n < max_nodes:
            nf = torch.cat([nf, torch.zeros(max_nodes - n, nf.shape[1])], dim=0)
        nodes.append(nf)
        edges.append(ei + node_offset)
        node_offset += n

    return {
        "tabular":          torch.stack(tabular),
        "node_features":    torch.stack(nodes),
        "edge_index":       torch.cat(edges, dim=1),
        "temporal_seq":     torch.stack(temporal),
        "physics_features": torch.stack(physics),
        "label":            torch.stack(labels),
    }


def get_stratified_folds(dataset, n_splits=5, seed=42):
    """Yield (fold, train_loader, val_loader, train_idx, val_idx)."""
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    labels  = dataset.labels.astype(int)
    for fold, (tr, va) in enumerate(skf.split(indices, labels)):
        tl = DataLoader(torch.utils.data.Subset(dataset, tr),
                        batch_size=32, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
        vl = DataLoader(torch.utils.data.Subset(dataset, va),
                        batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
        yield fold, tl, vl, tr, va


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 4 — Step 1: Dataset self-test")
    print("=" * 60)
    ds   = CRCMetastasisDataset()
    item = ds[0]
    print(f"\n  Sample item shapes:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:<22}: {tuple(v.shape)}")
    print(f"  Label: {item['label'].item()}")
    print(f"  Class weights: {ds.get_class_weights()}")

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"\n  Batch tensor shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:<22}: {tuple(v.shape)}")
    print("\n  ✓ Dataset OK — ready for model training")

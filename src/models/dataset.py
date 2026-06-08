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
from scipy.stats import rankdata
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
    "CLDN7","GRHL2","ELF3",                                # epithelial extra (mod 0)
    "PDGFRB","COL1A1","LOXL2","SERPINE1","ITGB6",         # mesenchymal extra (mod 1)
    "CTNNB1","APC","MYC","CCND1",                         # Wnt (mod 2)
    "TOP2A","MCM2","CCNE1","AURKA",                       # proliferation (mod 3)
    "CD8B","PRF1","GZMB","IFNG","CD274",                  # immune (mod 4)
    "HIF1A","VEGFA","LDHA","SLC2A1","CA9",                # hypoxia (mod 2)
]

GENE_MODULE = {
    "CDH1":0,"EPCAM":0,"KRT18":0,"KRT8":0,
    "CLDN7":0,"GRHL2":0,"ELF3":0,
    "VIM":1,"FN1":1,"CDH2":1,"SNAI1":1,"SNAI2":1,
    "ZEB1":1,"ZEB2":1,"TWIST1":1,"ACTA2":1,"MMP2":1,
    "PDGFRB":1,"COL1A1":1,"LOXL2":1,"SERPINE1":1,"ITGB6":1,
    "TGFB1":2,"TGFB2":2,"SMAD2":2,"SMAD3":2,"SMAD4":2,
    "CTNNB1":2,"APC":2,"MYC":2,"CCND1":2,
    "HIF1A":2,"VEGFA":2,"LDHA":2,"SLC2A1":2,"CA9":2,
    "MKI67":3,"PCNA":3,"TOP2A":3,"MCM2":3,"CCNE1":3,"AURKA":3,
    "CD8A":4,"FOXP3":4,"CD8B":4,"PRF1":4,"GZMB":4,"IFNG":4,"CD274":4,
}

# Preload ENSG mapping
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
    Nodes: EMT genes present in VST data (resolved via ENSG map).
    Node features: [normalised expression (1), module one-hot (5)] → 6-dim.
    Edges: within-module = always connected; cross-module = similarity-based.
    Returns (node_features: Tensor[N,6], edge_index: Tensor[2,E])
    """
    symbol_to_ensg = _load_symbol_to_ensg()
    # Map gene symbols to ENSG IDs that exist in VST index
    ensg_in_vst = []
    symbol_map = {}  # symbol -> index in ensg_in_vst
    for g in EMT_GRAPH_GENES:
        ensg = symbol_to_ensg.get(g)
        if ensg and ensg in vst_patient.index:
            if g not in symbol_map:
                symbol_map[g] = len(ensg_in_vst)
                ensg_in_vst.append(ensg)

    n = len(ensg_in_vst)
    symbols = list(symbol_map.keys())

    if n == 0:
        return torch.zeros(1, 6), torch.zeros(2, 0, dtype=torch.long)

    expr      = vst_patient[ensg_in_vst].values.astype(float)

    # Within-patient percentile ranks — platform-agnostic
    from scipy.stats import rankdata
    expr_rank = rankdata(expr, method="average") / len(expr)  # [0, 1]

    node_feats = []
    for i, gene in enumerate(symbols):
        oh = [0.0] * 5
        oh[GENE_MODULE.get(gene, 0)] = 1.0
        node_feats.append([float(expr_rank[i])] + oh)

    node_features = torch.tensor(node_feats, dtype=torch.float)

    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            same_mod = GENE_MODULE.get(symbols[i], -1) == \
                       GENE_MODULE.get(symbols[j], -1)
            close_expr = abs(float(expr_rank[i]) - float(expr_rank[j])) < 0.2
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

        # Within-patient percentile ranks — platform-agnostic
        vals = features.values.astype(np.float32)
        for i in range(vals.shape[0]):
            vals[i] = rankdata(vals[i], method="average") / vals.shape[1]
        self.features_arr = vals

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


def get_stratified_folds(dataset, n_splits=3, seed=42, precomputed_splits=None):
    """Yield (fold, train_loader, val_loader, train_idx, val_idx).

    Uses fixed random_state=42 for reproducible 3-fold splits.
    With precomputed_splits, reuses the same fold indices across calls
    (required for ensemble — all seeds must train/validate on identical splits).
    """
    indices = np.arange(len(dataset))
    labels  = dataset.labels.astype(int)

    if precomputed_splits is not None:
        fold_iter = enumerate(precomputed_splits)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_iter = enumerate(skf.split(indices, labels))

    for fold, (tr, va) in fold_iter:
        n_m1_val = labels[va].sum()
        if n_m1_val < 8:
            print(f"  Warning Fold {fold+1}: only {int(n_m1_val)} M1 samples in val set "
                  f"({int(n_m1_val/len(va)*100):.0f}%) — re-splitting...")
            from sklearn.model_selection import train_test_split
            combined_idx = np.concatenate([tr, va])
            combined_labels = labels[combined_idx]
            tr2, va2 = train_test_split(
                np.arange(len(combined_idx)),
                test_size=1.0 / n_splits,
                stratify=combined_labels,
                random_state=seed + fold,
            )
            tr = combined_idx[tr2]
            va = combined_idx[va2]
            print(f"    Adjusted fold {fold+1}: {len(tr)} train, {len(va)} val "
                  f"({int(labels[va].sum())} M1)")

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

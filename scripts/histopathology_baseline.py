"""
Histopathology Baseline — EfficientNet-B0 on NCT-CRC-HE-100K
==============================================================
Fine-tunes a pretrained EfficientNet-B0 on colorectal cancer
histopathology patches. Only needed if tabular AUROC stays below 0.80.

Dataset: NCT-CRC-HE-100K (100,000 patches, 9 tissue classes)
Binary classification: Tumor vs Rest (or customize as needed)

Download:
    wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
    unzip NCT-CRC-HE-100K.zip -d data/nct_crc/

Usage:
    python scripts/histopathology_baseline.py
    python scripts/histopathology_baseline.py --data-dir data/nct_crc/NCT-CRC-HE-100K
"""

import sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NCT-CRC-HE-100K classes
NCT_CLASSES = [
    "ADI",   # adipose tissue
    "BACK",  # background
    "DEB",   # debris
    "LYM",   # lymphocytes
    "MUC",   # mucus
    "MUS",   # smooth muscle
    "NORM",  # normal colon mucosa
    "STR",   # stroma
    "TUM",   # tumor epithelium
]

# For CRC metastasis prediction, define which classes are "positive"
# Default: TUM (tumor epithelium) is positive, everything else is negative.
# Options: can also include MUC (mucinous) if relevant for your cohort.
POSITIVE_CLASSES = {"TUM"}


def get_binary_targets(dataset):
    """Convert multi-class ImageFolder targets to binary."""
    class_to_idx = dataset.class_to_idx
    pos_indices = {class_to_idx[c] for c in POSITIVE_CLASSES if c in class_to_idx}
    targets = np.array([1 if t in pos_indices else 0 for t in dataset.targets])
    return targets


def main():
    parser = argparse.ArgumentParser(description="Histopathology Baseline — EfficientNet-B0")
    parser.add_argument("--data-dir", default="data/nct_crc/NCT-CRC-HE-100K")
    parser.add_argument("--out-dir", default="outputs/histopathology")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"\n  Dataset not found at {data_dir}")
        print(f"  Download from: https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip")
        print(f"  Then unzip to {data_dir.parent}")
        print("\n  Example:")
        print(f"    wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip")
        print(f"    unzip NCT-CRC-HE-100K.zip -d {data_dir.parent}")
        sys.exit(1)

    print("=" * 60)
    print("  Histopathology Baseline — EfficientNet-B0")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Transforms ───────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    # ── Load dataset ─────────────────────────────────────────────────────
    print("\n  Loading NCT-CRC-HE-100K...")
    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    print(f"  Classes: {full_dataset.classes}")
    print(f"  Total images: {len(full_dataset)}")

    # Binary targets
    binary_targets = get_binary_targets(full_dataset)
    full_dataset.targets = binary_targets.tolist()
    full_dataset.samples = [(s[0], int(binary_targets[i]))
                            for i, s in enumerate(full_dataset.samples)]
    full_dataset.classes = ["Negative", "Positive"]
    full_dataset.class_to_idx = {"Negative": 0, "Positive": 1}

    n_pos = binary_targets.sum()
    n_neg = len(binary_targets) - n_pos
    print(f"  Binary: {n_pos} positive ({100*n_pos/len(binary_targets):.1f}%), "
          f"{n_neg} negative ({100*n_neg/len(binary_targets):.1f}%)")

    # ── Train / test split ───────────────────────────────────────────────
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=0,
                               generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} patches")
    print(f"  Test:  {len(test_dataset)} patches")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n  Building EfficientNet-B0 (pretrained)...")
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(DEVICE)

    # Class weights for imbalance
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ─────────────────────────────────────────────────────────
    best_auroc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_preds, all_gt = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = torch.sigmoid(model(images)).cpu().numpy()
                all_preds.extend(outputs.squeeze())
                all_gt.extend(labels.numpy())

        test_auroc = roc_auc_score(all_gt, all_preds)
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1:>2}/{args.epochs}  "
              f"Loss={avg_loss:.4f}  Test AUROC={test_auroc:.4f}",
              end="")
        if test_auroc > best_auroc:
            best_auroc = test_auroc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_dir / "efficientnet_best.pth")
            print("  ← saved")
        else:
            print()

    print(f"\n  Best test AUROC: {best_auroc:.4f} (epoch {best_epoch})")

    # ── Final ROC curve ──────────────────────────────────────────────────
    model.eval()
    all_preds, all_gt = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.extend(outputs.squeeze())
            all_gt.extend(labels.numpy())

    fpr, tpr, _ = roc_curve(all_gt, all_preds)
    final_auroc = roc_auc_score(all_gt, all_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#2ca02c",
            label=f"EfficientNet-B0 (AUROC={final_auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUROC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — NCT-CRC-HE-100K Histopathology", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "histopathology_roc.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve saved -> {path}")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "dataset": "NCT-CRC-HE-100K",
        "model": "EfficientNet-B0",
        "pretrained": True,
        "binary_positive_classes": list(POSITIVE_CLASSES),
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "best_epoch": best_epoch,
        "best_test_auroc": round(float(best_auroc), 4),
        "final_test_auroc": round(float(final_auroc), 4),
    }
    with open(out_dir / "histopathology_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {out_dir / 'histopathology_results.json'}")

    # ── Guidance ─────────────────────────────────────────────────────────
    if final_auroc >= 0.80:
        print(f"\n  ✓ AUROC >= 0.80 — histopathology features are informative!")
        print(f"    Consider extracting patch-level embeddings and")
        print(f"    using them as an additional modality in the MPS model.")
    else:
        print(f"\n  Histopathology AUROC is {final_auroc:.2f}.")
        print(f"    This may be due to: binary label definition, class imbalance,")
        print(f"    or insufficient fine-tuning. Try increasing epochs or")
        print(f"    using class-balanced sampling.")

    print("=" * 60)


if __name__ == "__main__":
    main()

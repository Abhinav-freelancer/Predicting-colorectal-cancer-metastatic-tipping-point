"""
Phase 2 — CNV data downloader for TCGA-COAD
=============================================
Downloads "Masked Copy Number Segment" files from GDC and derives
per-patient copy number variation (CNV) features correlated with
CRC metastasis potential.

CNV features per patient:
    fga                 — fraction of the genome altered (copy number != 2)
    n_segments          — total number of CNV segments
    mean_segment_mean   — average log2 copy ratio across all segments
    n_gains             — number of amplified segments (log2 ratio > 0.3)
    n_losses            — number of deleted segments (log2 ratio < -0.3)
    cnv_burden_score    — composite: (n_gains + n_losses) / n_segments
    amp_8q              — chr8q amplification (MYC locus)
    amp_20q             — chr20q amplification (AURKA, BCAS1)
    amp_13q             — chr13q amplification (CDX2)
    del_18q             — chr18q deletion (SMAD4, DCC)
    del_17p             — chr17p deletion (TP53)
    del_5q              — chr5q deletion (APC)
    del_3p              — chr3p deletion (VHL/PBRM1)

Usage:
    python src/data/tcga_downloader_cnv.py
    python src/data/tcga_downloader_cnv.py --max-files 20
"""

import sys
import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.tcga_downloader import _post, _get, REQUEST_DELAY, GDC_DATA_URL, GDC_FILES_URL, CHUNK_SIZE, _md5

PROJECT_ID = "TCGA-COAD"

CHROMOSOME_ARMS = {
    "amp_8q":    ("8", 40000000, 145000000, 0.3),
    "amp_20q":   ("20", 20000000, 64000000, 0.3),
    "amp_13q":   ("13", 20000000, 115000000, 0.3),
    "del_18q":   ("18", 20000000, 80000000, -0.3),
    "del_17p":   ("17", 0, 25000000, -0.3),
    "del_5q":    ("5", 40000000, 182000000, -0.3),
    "del_3p":    ("3", 0, 90000000, -0.3),
}


def query_cnv_files(limit: int = 500) -> pd.DataFrame:
    print("\nQuerying GDC for TCGA-COAD Masked Copy Number Segment files...")
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT_ID}},
            {"op": "=", "content": {"field": "data_type", "value": "Copy Number Segment"}},
            {"op": "=", "content": {"field": "data_format", "value": "TXT"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "ASCAT2"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "WGS"}},
        ],
    }
    fields = [
        "file_id", "file_name", "md5sum", "file_size",
        "cases.case_id", "cases.submitter_id",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields":  ",".join(fields),
        "format":  "json",
        "size":    str(limit),
    }
    data = _get(GDC_FILES_URL, params)
    hits = data.get("data", {}).get("hits", [])
    if not hits:
        print("  No WGS CNV files found. Trying WXS (exome) CNV files...")
        filters["content"][4] = {"op": "=", "content": {"field": "experimental_strategy", "value": "WXS"}}
        params["filters"] = json.dumps(filters)
        data = _get(GDC_FILES_URL, params)
        hits = data.get("data", {}).get("hits", [])
    if not hits:
        raise RuntimeError("No CNV segment files returned from GDC API.")
    rows = []
    for hit in hits:
        case = hit.get("cases", [{}])[0]
        rows.append({
            "file_id":      hit["file_id"],
            "file_name":    hit["file_name"],
            "md5sum":       hit.get("md5sum", ""),
            "file_size":    hit.get("file_size", 0),
            "case_id":      case.get("case_id", ""),
            "submitter_id": case.get("submitter_id", ""),
        })
    out = pd.DataFrame(rows)
    print(f"    Found {len(out)} CNV segment files ({len(out['case_id'].unique())} unique cases)")
    return out


def download_cnv_files(
    manifest:   pd.DataFrame,
    out_dir:    Path,
    max_files:  Optional[int] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_ids = manifest["file_id"].tolist()
    if max_files:
        file_ids = file_ids[:max_files]
    id_to_md5  = dict(zip(manifest["file_id"], manifest["md5sum"]))
    id_to_name = dict(zip(manifest["file_id"], manifest["file_name"]))
    downloaded = {}
    print(f"\n    Downloading {len(file_ids)} CNV files -> {out_dir}")
    for idx, fid in enumerate(file_ids):
        fname    = id_to_name.get(fid, f"{fid}.txt")
        out_path = out_dir / fname
        expected = id_to_md5.get(fid, "")
        if out_path.exists() and expected:
            actual = _md5(out_path)
            if actual == expected:
                print(f"    [{idx+1}/{len(file_ids)}] SKIP (cached)  {fname}")
                downloaded[fid] = str(out_path)
                continue
        print(f"    [{idx+1}/{len(file_ids)}] Downloading  {fname}", end="", flush=True)
        try:
            resp = _post(GDC_DATA_URL, {"ids": [fid]})
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
            if expected:
                actual = _md5(out_path)
                ok = "OK" if actual == expected else "md5 mismatch"
                print(f"  {ok}")
            else:
                print("  OK")
            downloaded[fid] = str(out_path)
        except Exception as e:
            print(f"  FAILED: {e}")
    return downloaded


def compute_patient_cnv_features(segments_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        fpath = segments_dir / row["file_name"]
        if not fpath.exists():
            continue
        try:
            seg = pd.read_csv(fpath, sep="\t", comment="#")
        except Exception:
            seg = pd.read_csv(fpath, sep="\t", comment="#", header=None,
                               names=["Chromosome","Start","End","Num_Probes","Segment_Mean"])
        case_id = row["case_id"]
        seg = seg.dropna(subset=["Segment_Mean"])
        seg["Segment_Mean"] = seg["Segment_Mean"].astype(float)
        total_bp = (seg["End"] - seg["Start"]).sum()
        arm_features = {}
        for feat_name, (chrom, start, end, threshold) in CHROMOSOME_ARMS.items():
            mask = (seg["Chromosome"].astype(str) == chrom) & (seg["Start"] >= start) & (seg["End"] <= end)
            arm_segs = seg[mask]
            if len(arm_segs) > 0:
                mean_val = arm_segs["Segment_Mean"].mean()
                arm_features[feat_name] = 1.0 if (
                    (threshold > 0 and mean_val > threshold) or
                    (threshold < 0 and mean_val < threshold)
                ) else 0.0
            else:
                arm_features[feat_name] = 0.0
        altered_bp = seg[np.abs(seg["Segment_Mean"]) > 0.2]
        altered_bp_sum = (altered_bp["End"] - altered_bp["Start"]).sum()
        features = {
            "case_id":              case_id,
            "fga":                  round(altered_bp_sum / max(total_bp, 1), 6),
            "n_segments":           len(seg),
            "mean_segment_mean":    round(seg["Segment_Mean"].mean(), 6),
            "n_gains":              int((seg["Segment_Mean"] > 0.3).sum()),
            "n_losses":             int((seg["Segment_Mean"] < -0.3).sum()),
            "cnv_burden_score":     round((seg["Segment_Mean"] > 0.3).sum() / max(len(seg), 1), 6),
            **arm_features,
        }
        rows.append(features)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Download TCGA-COAD CNV data")
    parser.add_argument("--data-dir",     default="data/raw/tcga_coad",
                        help="Root directory for raw TCGA data")
    parser.add_argument("--manifest-dir", default="data/manifests",
                        help="Directory for manifest CSVs")
    parser.add_argument("--out-dir",      default="data/processed/temporal",
                        help="Output directory for CNV features CSV")
    parser.add_argument("--limit",        type=int, default=500)
    parser.add_argument("--download",     action="store_true",
                        help="Actually download files (default: metadata only)")
    parser.add_argument("--max-files",    type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TCGA-COAD CNV Data Acquisition")
    print("=" * 60)

    cnv_manifest = query_cnv_files(limit=args.limit)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cnv_manifest.to_csv(manifest_dir / "tcga_cnv_files.csv", index=False)

    if args.download:
        cnv_dir = data_dir / "cnv"
        downloaded = download_cnv_files(cnv_manifest, cnv_dir, max_files=args.max_files)
        print(f"\n  Downloaded {len(downloaded)} CNV files to {cnv_dir}")
        print("\n  Computing per-patient CNV features...")
        features = compute_patient_cnv_features(cnv_dir, cnv_manifest)
    else:
        print("\n  Metadata-only mode. To download, add --download flag.")
        print("  Using existing files if present...")
        cnv_dir = data_dir / "cnv"
        if cnv_dir.exists():
            features = compute_patient_cnv_features(cnv_dir, cnv_manifest)
        else:
            features = pd.DataFrame()

    if len(features) > 0:
        out_path = out_dir / "cnv_features.csv"
        features.to_csv(out_path, index=False)
        print(f"\n  CNV features saved -> {out_path}")
        print(f"  Shape: {features.shape}")
        print(f"  Features: {[c for c in features.columns if c != 'case_id']}")
        print(f"  Patients with CNV data: {len(features)}")

        labeled = pd.read_csv(manifest_dir / "cohort_labeled.csv")
        merged = labeled.merge(features, on="case_id", how="left")
        n_with_cnv = merged["fga"].notna().sum()
        print(f"  Patients with both labels and CNV: {n_with_cnv}/{len(labeled)}")
    else:
        print("\n  No CNV features computed. Run with --download first.")

    print("\n  Next: Run feature_builder.py to integrate CNV features into Phase 2")
    print("=" * 60)


if __name__ == "__main__":
    main()

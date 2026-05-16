"""
Phase 1 — Step 2: TCGA-COAD data downloader
Uses the NCI GDC (Genomic Data Commons) public API — no login required.

Downloads:
  1. RNA-seq gene expression (HTSeq counts)
  2. Clinical data (survival, stage, metastasis status)
  3. Somatic mutations (MAF files)

Usage:
    python src/data/tcga_downloader.py
    python src/data/tcga_downloader.py --data-dir data/raw/tcga_coad --limit 50
"""

import os
import sys
import json
import time
import argparse
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ── GDC API endpoints ────────────────────────────────────────────────────────
GDC_FILES_URL   = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL    = "https://api.gdc.cancer.gov/data"
GDC_CASES_URL   = "https://api.gdc.cancer.gov/cases"

# ── TCGA-COAD project identifier ─────────────────────────────────────────────
PROJECT_ID = "TCGA-COAD"

# ── Rate limiting: be polite to the API ──────────────────────────────────────
REQUEST_DELAY = 0.5   # seconds between API calls
CHUNK_SIZE    = 1024  # bytes per download chunk (1 KB — conservative)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET with retry logic and polite rate limiting."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{retries} after {wait}s — {e}")
            time.sleep(wait)


def _post(url: str, payload: dict, retries: int = 3) -> requests.Response:
    """POST with retry logic."""
    headers = {"Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{retries} after {wait}s — {e}")
            time.sleep(wait)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Query file manifests from GDC
# ─────────────────────────────────────────────────────────────────────────────

def query_rnaseq_files(limit: int = 500) -> pd.DataFrame:
    """
    Query GDC for TCGA-COAD RNA-seq HTSeq count files.
    Returns a DataFrame with file_id, file_name, case_id, md5sum.
    """
    print(f"\n[1/3] Querying GDC for RNA-seq files (project={PROJECT_ID})...")

    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT_ID}},
            {"op": "=", "content": {"field": "data_type",     "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "data_format",   "value": "TSV"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "experimental_strategy",  "value": "RNA-Seq"}},
        ]
    }

    fields = [
        "file_id", "file_name", "md5sum", "file_size",
        "cases.case_id", "cases.submitter_id",
        "cases.demographic.gender", "cases.demographic.vital_status",
        "cases.diagnoses.tumor_stage", "cases.diagnoses.days_to_death",
        "cases.diagnoses.days_to_last_follow_up",
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
        raise RuntimeError("No RNA-seq files returned. Check GDC API connectivity.")

    rows = []
    for hit in hits:
        case = hit.get("cases", [{}])[0]
        diag = case.get("diagnoses", [{}])[0]
        rows.append({
            "file_id":           hit["file_id"],
            "file_name":         hit["file_name"],
            "md5sum":            hit.get("md5sum", ""),
            "file_size_bytes":   hit.get("file_size", 0),
            "case_id":           case.get("case_id", ""),
            "submitter_id":      case.get("submitter_id", ""),
            "gender":            case.get("demographic", {}).get("gender", ""),
            "vital_status":      case.get("demographic", {}).get("vital_status", ""),
            "tumor_stage":       diag.get("tumor_stage", ""),
            "days_to_death":     diag.get("days_to_death"),
            "days_to_last_fu":   diag.get("days_to_last_follow_up"),
        })

    df = pd.DataFrame(rows)
    print(f"    Found {len(df)} RNA-seq files.")
    return df


def query_clinical_data(case_ids: list) -> pd.DataFrame:
    """
    Query GDC Clinical API for detailed metastasis and staging information.
    """
    print(f"\n[2/3] Querying clinical data for {len(case_ids)} cases...")

    fields = [
        "case_id", "submitter_id",
        "diagnoses.tumor_stage", "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_pathologic_m",    # M0 / M1 = metastasis status
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.days_to_diagnosis",
        "diagnoses.days_to_recurrence",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "demographic.age_at_index",
        "demographic.gender",
        "demographic.race",
    ]

    # GDC supports up to 2000 per query; batch if needed
    batch_size = 200
    all_rows   = []

    for i in range(0, len(case_ids), batch_size):
        batch = case_ids[i : i + batch_size]
        filters = {
            "op": "in",
            "content": {"field": "case_id", "value": batch}
        }
        params = {
            "filters": json.dumps(filters),
            "fields":  ",".join(fields),
            "format":  "json",
            "size":    str(batch_size),
        }
        data = _get(GDC_CASES_URL, params)
        hits = data.get("data", {}).get("hits", [])

        for hit in hits:
            diag  = hit.get("diagnoses", [{}])[0]
            demog = hit.get("demographic", {})
            all_rows.append({
                "case_id":          hit.get("case_id", ""),
                "submitter_id":     hit.get("submitter_id", ""),
                "tumor_stage":      diag.get("tumor_stage", ""),
                "ajcc_stage":       diag.get("ajcc_pathologic_stage", ""),
                "ajcc_m":           diag.get("ajcc_pathologic_m", ""),   # KEY: M0/M1
                "ajcc_t":           diag.get("ajcc_pathologic_t", ""),
                "ajcc_n":           diag.get("ajcc_pathologic_n", ""),
                "days_to_dx":       diag.get("days_to_diagnosis"),
                "days_to_recur":    diag.get("days_to_recurrence"),
                "days_to_last_fu":  diag.get("days_to_last_follow_up"),
                "vital_status":     diag.get("vital_status", ""),
                "days_to_death":    diag.get("days_to_death"),
                "age_at_index":     demog.get("age_at_index"),
                "gender":           demog.get("gender", ""),
                "race":             demog.get("race", ""),
            })

        print(f"    Batch {i//batch_size + 1}: retrieved {len(hits)} records")

    df = pd.DataFrame(all_rows)
    print(f"    Total clinical records: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Download files from GDC
# ─────────────────────────────────────────────────────────────────────────────

def download_files(
    file_ids:  list,
    out_dir:   Path,
    manifest:  pd.DataFrame,
    max_files: Optional[int] = None,
) -> dict:
    """
    Download RNA-seq count files from GDC Data endpoint.
    Skips files already downloaded (checks md5 if available).
    Returns dict mapping file_id → local path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_files:
        file_ids = file_ids[:max_files]

    id_to_md5  = dict(zip(manifest["file_id"], manifest["md5sum"]))
    id_to_name = dict(zip(manifest["file_id"], manifest["file_name"]))
    downloaded = {}

    print(f"\n    Downloading {len(file_ids)} files → {out_dir}")
    print("    (already-downloaded files will be skipped)\n")

    for idx, fid in enumerate(file_ids):
        fname     = id_to_name.get(fid, fid)
        out_path  = out_dir / fname
        expected  = id_to_md5.get(fid, "")

        # Skip if file exists and md5 matches
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

            # Verify md5 if available
            if expected:
                actual = _md5(out_path)
                ok = "✓" if actual == expected else "⚠ md5 mismatch"
                print(f"  {ok}")
            else:
                print("  ✓")

            downloaded[fid] = str(out_path)

        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build cohort manifest with metastasis labels
# ─────────────────────────────────────────────────────────────────────────────

def build_cohort_manifest(
    rna_manifest:  pd.DataFrame,
    clinical_df:   pd.DataFrame,
    out_dir:       Path,
) -> pd.DataFrame:
    """
    Merge RNA-seq file metadata with clinical data.
    Derive binary metastasis label from AJCC M-stage.

    Label encoding:
        0 = non-metastatic (M0, Stage I/II/III)
        1 = metastatic     (M1, Stage IV)
        -1 = unknown       (excluded from supervised training)
    """
    print("\n[3/3] Building cohort manifest...")

    merged = rna_manifest.merge(
        clinical_df[["case_id", "ajcc_m", "ajcc_stage", "ajcc_t", "ajcc_n",
                     "days_to_recur", "days_to_death", "days_to_last_fu",
                     "vital_status", "age_at_index", "gender", "race"]],
        on="case_id",
        how="left",
        suffixes=("_rna", "_clin"),
    )

    # ── Derive metastasis label from AJCC M stage ─────────────────────────
    def label_from_m(m: str) -> int:
        if pd.isna(m) or str(m).strip() in ("", "--", "not reported"):
            return -1
        m = str(m).upper().strip()
        if m.startswith("M1"):
            return 1        # metastatic
        if m.startswith("M0"):
            return 0        # non-metastatic
        return -1           # ambiguous

    merged["metastasis_label"] = merged["ajcc_m"].apply(label_from_m)

    # ── Also derive from stage string as a fallback ───────────────────────
    def label_from_stage(row) -> int:
        if row["metastasis_label"] != -1:
            return row["metastasis_label"]   # already resolved
        stage = str(row.get("ajcc_stage", "")).upper()
        if "IV" in stage:
            return 1
        if any(s in stage for s in ["I ", "II", "III"]):
            return 0
        return -1

    merged["metastasis_label"] = merged.apply(label_from_stage, axis=1)

    # ── Summary stats ─────────────────────────────────────────────────────
    counts   = merged["metastasis_label"].value_counts()
    n_meta   = counts.get(1, 0)
    n_nonmet = counts.get(0, 0)
    n_unkn   = counts.get(-1, 0)
    n_total  = len(merged)

    print(f"\n    Cohort summary")
    print(f"    ├─ Total patients      : {n_total}")
    print(f"    ├─ Metastatic   (1)    : {n_meta}  ({100*n_meta/n_total:.1f}%)")
    print(f"    ├─ Non-metastatic (0)  : {n_nonmet}  ({100*n_nonmet/n_total:.1f}%)")
    print(f"    └─ Unknown / excluded  : {n_unkn}  ({100*n_unkn/n_total:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cohort_manifest.csv"
    labeled_path  = out_dir / "cohort_labeled.csv"

    merged.to_csv(manifest_path, index=False)

    # Labeled-only (excludes unknown) for model training
    labeled = merged[merged["metastasis_label"] != -1].copy()
    labeled.to_csv(labeled_path, index=False)

    print(f"\n    Saved:")
    print(f"    ├─ Full manifest : {manifest_path}")
    print(f"    └─ Labeled only  : {labeled_path}  ({len(labeled)} patients)")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 4. Validate download completeness
# ─────────────────────────────────────────────────────────────────────────────

def validate_downloads(data_dir: Path, manifest: pd.DataFrame) -> None:
    """
    Cross-check downloaded files against the manifest.
    Reports missing files and size anomalies.
    """
    print("\nValidating downloads...")
    missing = []
    ok      = 0

    for _, row in manifest.iterrows():
        fpath = data_dir / row["file_name"]
        if fpath.exists():
            ok += 1
        else:
            missing.append(row["file_name"])

    print(f"  ✓ Present : {ok}")
    print(f"  ✗ Missing : {len(missing)}")
    if missing[:5]:
        for f in missing[:5]:
            print(f"    - {f}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download TCGA-COAD data via GDC API")
    parser.add_argument("--data-dir",    default="data/raw/tcga_coad",
                        help="Root directory for raw TCGA data")
    parser.add_argument("--manifest-dir", default="data/manifests",
                        help="Directory for cohort manifest CSVs")
    parser.add_argument("--limit",       type=int, default=500,
                        help="Max number of RNA-seq files to query (default 500)")
    parser.add_argument("--download",    action="store_true",
                        help="Actually download files (default: metadata only)")
    parser.add_argument("--max-files",   type=int, default=None,
                        help="Cap downloads for testing (e.g. --max-files 10)")
    args = parser.parse_args()

    data_dir     = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)

    print("=" * 60)
    print("  TCGA-COAD Phase 1 Data Acquisition")
    print("=" * 60)

    # Step 1: Query RNA-seq file manifest
    rna_manifest = query_rnaseq_files(limit=args.limit)
    rna_manifest.to_csv(manifest_dir / "tcga_rnaseq_files.csv", index=False)

    # Step 2: Query clinical data
    case_ids    = rna_manifest["case_id"].dropna().unique().tolist()
    clinical_df = query_clinical_data(case_ids)
    clinical_df.to_csv(manifest_dir / "tcga_clinical.csv", index=False)

    # Step 3: Build labeled cohort manifest
    cohort = build_cohort_manifest(rna_manifest, clinical_df, manifest_dir)

    # Step 4: Optionally download files
    if args.download:
        file_ids   = rna_manifest["file_id"].tolist()
        rnaseq_dir = data_dir / "rnaseq"
        downloaded = download_files(
            file_ids,
            rnaseq_dir,
            rna_manifest,
            max_files=args.max_files,
        )
        validate_downloads(rnaseq_dir, rna_manifest)
        print(f"\n  Downloaded {len(downloaded)} files to {rnaseq_dir}")
    else:
        print("\n  Metadata-only mode. To download files, add --download flag.")
        print("  For a test run:  python src/data/tcga_downloader.py --download --max-files 10")
        print("  Full download:   python src/data/tcga_downloader.py --download")

    print("\n  Phase 1 complete. Next: python src/data/preprocessor.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

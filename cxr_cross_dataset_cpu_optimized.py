# =============================================================================
# A Lightweight Multimodal AI for Chest X-ray Diagnosis
# Using Clinical Text with Cross-Dataset Generalization
#
# ✅ FULLY OPTIMIZED FOR:
#    CPU-only training (AMD Ryzen 7 7730U)
#    16 GB RAM
#    496 MB GPU (ignored — CPU mode forced)
#    Windows 11 Home
#
# NOVELTY:
#    1. Domain Adversarial Training (DANN)
#    2. Bidirectional Cross-Modal Attention
#    3. MAML-inspired Meta-Learning
#    4. Dataset-Conditioned Classifier
#    5. Harmonized Label Set (3 datasets → 10 classes)
#
# DATASETS:
#    MIMIC-CXR     → train + in-domain val
#    CheXpert      → cross-dataset validation
#    ChestX-ray14  → cross-dataset test (held-out)
#
# INSTALL:
#    pip install pydicom pillow numpy pandas tqdm torch torchvision
#                transformers scikit-learn
# =============================================================================

import os, re, gc, json, time, warnings, argparse, ast
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.autograd import Function

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# ✅ FORCE CPU — your GPU (496MB) cannot handle this model
# =============================================================================
DEVICE = torch.device("cpu")
torch.set_num_threads(8)          # Use all Ryzen 7 cores
torch.set_num_interop_threads(4)
os.environ["OMP_NUM_THREADS"]  = "8"
os.environ["MKL_NUM_THREADS"]  = "8"

print(f"  Running on : {DEVICE}")
print(f"  CPU threads: {torch.get_num_threads()}")

# =============================================================================
# CONFIG — Edit paths before running
# =============================================================================

CONFIG = {
    # ── Dataset Paths ────────────────────────────────────────────────────────
    "datasets": {
        "mimic": {
            "dicom_dir"   : "C:/Users/turvi/OneDrive/Desktop/dip code/official_data_iccv_final",
            "metadata_csv": "C:/Users/turvi/OneDrive/Desktop/dip code/mimic_cxr_aug_train.csv",
            "validate_csv": "C:/Users/turvi/OneDrive/Desktop/dip code/mimic_cxr_aug_validate.csv",
            "labels_csv"  : "C:/Users/turvi/OneDrive/Desktop/dip code/mimic_cxr_aug_train.csv",
            "reports_dir" : "C:/Users/turvi/OneDrive/Desktop/dip code/official_data_iccv_final/files",
            "output_dir"  : "C:/Users/turvi/Downloads",
            "role"        : "train",
            "domain_id"   : 0,
        },
        "chexpert": {
            "image_dir"   : "",
            "labels_csv"  : "",
            "output_dir"  : "C:/Users/turvi/Downloads/chexpert",
            "role"        : "cross_val",
            "domain_id"   : 1,
        },
        "chestxray14": {
            "image_dir"   : "",
            "labels_csv"  : "",
            "output_dir"  : "C:/Users/turvi/Downloads/chexpert/chestxray14",
            "role"        : "cross_test",
            "domain_id"   : 2,
        },
    },

    # ── Preprocessing ─────────────────────────────────────────────────────
    "target_size"            : (224, 224),
    "image_format"           : "PNG",
    "jpeg_quality"           : 90,
    "views"                  : ["PA", "AP"],

    # ✅ CPU-safe sample sizes — use all available MIMIC images (~12k)
    "max_samples_per_dataset": 12000,  # All available images ≈ 3GB RAM

    # ── Model ─────────────────────────────────────────────────────────────
    "image_backbone"  : "mobilenet_v3_small",   # ✅ Lightest backbone (~2.5M params)
    "text_encoder"    : "emilyalsentzer/Bio_ClinicalBERT",
    "num_classes"     : 10,
    "num_domains"     : 3,
    "embed_dim"       : 128,            # ✅ Reduced from 256 → saves RAM
    "dropout"         : 0.3,

    # ── Cross-Dataset Novelty Weights ──────────────────────────────────
    "lambda_domain"   : 0.1,
    "lambda_meta"     : 0.05,
    "meta_lr"         : 1e-3,
    "meta_steps"      : 2,             # ✅ Reduced inner loop steps (CPU friendly)

    # ── Training ─────────────────────────────────────────────────────────
    "batch_size"      : 4,             # ✅ Very small batch for 16GB RAM
    "epochs"          : 10,            # ✅ 10 epochs — realistic on CPU with early stop
    "lr"              : 1e-4,
    "weight_decay"    : 1e-5,
    "max_text_len"    : 64,            # ✅ Reduced from 128 → halves BERT memory
    "num_workers"     : 0,             # ✅ Must be 0 on Windows (multiprocessing bug)
    "model_save_path" : "best_cxr_model.pth",
    "results_path"    : "cross_dataset_results.json",

    # ── Early Stopping ───────────────────────────────────────────────────
    "early_stop_patience" : 3,         # Stop if no improvement for 3 epochs
    "min_delta"           : 0.001,     # Minimum AUC improvement to count
}

# =============================================================================
# HARMONIZED LABELS (10 classes across all 3 datasets)
# =============================================================================

HARMONIZED_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "No Finding", "Pneumonia", "Pneumothorax",
    "Lung Opacity", "Fracture",
]

LABEL_MAPS = {
    "mimic": {
        "Atelectasis":"Atelectasis","Cardiomegaly":"Cardiomegaly",
        "Consolidation":"Consolidation","Edema":"Edema",
        "Pleural Effusion":"Effusion","No Finding":"No Finding",
        "Pneumonia":"Pneumonia","Pneumothorax":"Pneumothorax",
        "Lung Opacity":"Lung Opacity","Fracture":"Fracture",
    },
    "chexpert": {
        "Atelectasis":"Atelectasis","Cardiomegaly":"Cardiomegaly",
        "Consolidation":"Consolidation","Edema":"Edema",
        "Pleural Effusion":"Effusion","No Finding":"No Finding",
        "Pneumonia":"Pneumonia","Pneumothorax":"Pneumothorax",
        "Lung Opacity":"Lung Opacity","Fracture":"Fracture",
    },
    "chestxray14": {
        "Atelectasis":"Atelectasis","Cardiomegaly":"Cardiomegaly",
        "Consolidation":"Consolidation","Edema":"Edema",
        "Effusion":"Effusion","No Finding":"No Finding",
        "Pneumonia":"Pneumonia","Pneumothorax":"Pneumothorax",
        "Infiltration":"Lung Opacity","Fracture":"Fracture",
    },
}

# =============================================================================
# STEP 1 — PREPROCESSING (DICOM → PNG, resize, label harmonization)
# =============================================================================

def normalize_dicom(pixel_array):
    arr = pixel_array.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    arr *= 255.0
    return arr.astype(np.uint8)


def _parse_list_field(value):
    """Parse a string representation of a Python list (e.g. "['a','b']")."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except Exception:
        return [str(value)]


def _first_list_item(value, default=""):
    items = _parse_list_field(value)
    if not items:
        return default
    return str(items[0])


# =============================================================================
# NLP-BASED LABEL EXTRACTION FROM RADIOLOGY REPORTS
# =============================================================================

# Keyword map: harmonized label -> list of keyword patterns
_LABEL_KEYWORDS = {
    "Atelectasis"  : ["atelectasis", "atelectatic"],
    "Cardiomegaly" : ["cardiomegaly", "enlarged cardiac", "enlarged heart",
                      "cardiac enlargement", "heart is enlarged"],
    "Consolidation": ["consolidation", "consolidative"],
    "Edema"        : ["edema", "pulmonary edema", "vascular congestion",
                      "fluid overload"],
    "Effusion"     : ["effusion", "pleural effusion", "pleural fluid"],
    "No Finding"   : ["no acute cardiopulmonary", "no acute intrathoracic",
                      "no acute process", "no acute finding",
                      "normal study", "unremarkable",
                      "no active disease", "clear lungs"],
    "Pneumonia"    : ["pneumonia"],
    "Pneumothorax" : ["pneumothorax"],
    "Lung Opacity" : ["opacity", "opacit", "infiltrat", "haziness", "haze"],
    "Fracture"     : ["fracture", "fractures"],
}

# Negation prefixes — if any appears right before a keyword, the mention is negated
_NEGATION_PREFIXES = [
    "no ", "no evidence of ", "without ", "negative for ",
    "not ", "unlikely ", "absent ", "no definite ",
    "no definitive ", "no convincing ", "no significant ",
    "rather than ", "resolved ", "no focal ",
]

def _is_negated(text_lower, start_pos):
    """Check if a keyword starting at start_pos in text_lower is preceded by negation."""
    prefix_region = text_lower[max(0, start_pos - 35):start_pos]
    for neg in _NEGATION_PREFIXES:
        if prefix_region.rstrip().endswith(neg.rstrip()):
            return True
    return False


def extract_labels_from_text(report_text):
    """
    Extract binary labels for all 10 harmonized classes from a radiology report
    using keyword matching with negation detection.

    Returns a dict: {label_name: 0 or 1}
    """
    labels = {lbl: 0 for lbl in HARMONIZED_LABELS}
    if not report_text or not isinstance(report_text, str):
        return labels

    text_lower = report_text.lower()

    for lbl, keywords in _LABEL_KEYWORDS.items():
        for kw in keywords:
            pos = 0
            while True:
                idx = text_lower.find(kw, pos)
                if idx == -1:
                    break
                if not _is_negated(text_lower, idx):
                    labels[lbl] = 1
                    break  # Found a positive mention, done for this label
                pos = idx + len(kw)

    # "No Finding" should be mutually exclusive with pathology labels
    # If any pathology is found, "No Finding" should be 0
    pathology_found = any(labels[l] == 1 for l in HARMONIZED_LABELS if l != "No Finding")
    if pathology_found:
        labels["No Finding"] = 0
    # If no pathology and no explicit "No Finding" marker, check common patterns
    if not pathology_found and labels["No Finding"] == 0:
        no_finding_phrases = ["normal", "no acute", "unremarkable", "clear lungs"]
        for phrase in no_finding_phrases:
            idx = text_lower.find(phrase)
            if idx != -1:
                labels["No Finding"] = 1
                break

    return labels


def extract_labels_for_df(df, text_col="report_text"):
    """
    Apply label extraction to an entire DataFrame.
    Returns df with all HARMONIZED_LABELS columns populated.
    """
    all_labels = df[text_col].apply(extract_labels_from_text)
    for lbl in HARMONIZED_LABELS:
        df[lbl] = all_labels.apply(lambda d: d.get(lbl, 0))

    # Print label distribution
    print("\n  Label Distribution (extracted from reports):")
    print(f"    {'Label':<28} {'Positive':>8} {'Negative':>8} {'Rate':>7}")
    print(f"    {'-'*55}")
    for lbl in HARMONIZED_LABELS:
        pos = int(df[lbl].sum())
        neg = len(df) - pos
        rate = pos / len(df) * 100 if len(df) > 0 else 0
        print(f"    {lbl:<28} {pos:>8} {neg:>8} {rate:>6.1f}%")

    return df


def dicom_to_png(dicom_path, output_path, size=(224, 224)):
    try:
        ds    = pydicom.dcmread(str(dicom_path))
        pixel = normalize_dicom(ds.pixel_array)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel = 255 - pixel
        img = Image.fromarray(pixel).convert("L").resize(size, Image.LANCZOS)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), format="PNG", optimize=True)
        return True
    except:
        return False


def resize_and_save(src_path, dst_path, size=(224, 224)):
    try:
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(str(src_path)).convert("L").resize(size, Image.LANCZOS)
        img.save(str(dst_path), format="PNG", optimize=True)
        return True
    except:
        return False


def apply_label_map(df, dataset_name):
    """Map dataset-specific columns to harmonized labels."""
    for src, harm in LABEL_MAPS[dataset_name].items():
        if src in df.columns:
            df[harm] = df[src].fillna(0).replace(-1, 1).astype(int)
        elif harm not in df.columns:
            df[harm] = 0
    for lbl in HARMONIZED_LABELS:
        if lbl not in df.columns:
            df[lbl] = 0
    return df


def preprocess_mimic(cfg):
    print("\n[MIMIC-CXR] Preprocessing DICOM → PNG...")
    dcfg = cfg["datasets"]["mimic"]
    meta_path = Path(dcfg["metadata_csv"])
    if not meta_path.exists():
        print(f"  ⚠ MIMIC metadata not found at '{meta_path}', skipping.")
        return None

    meta = pd.read_csv(meta_path)

    # Try to support both standard MIMIC-CXR metadata format and the custom "aug" dataset format
    if {"study_id", "dicom_id", "subject_id", "ViewPosition"}.issubset(meta.columns):
        labels_path = Path(dcfg["labels_csv"])
        if not labels_path.exists():
            print(f"  ⚠ MIMIC labels not found at '{labels_path}', proceeding with zeros.")
            merged = meta[meta["ViewPosition"].isin(cfg["views"])].copy()
        else:
            labels = pd.read_csv(labels_path)
            merged = meta[meta["ViewPosition"].isin(cfg["views"])].merge(labels, on="study_id", how="inner")
        merged = apply_label_map(merged, "mimic")

        n = min(len(merged), cfg["max_samples_per_dataset"])
        merged = merged.sample(n, random_state=42).reset_index(drop=True)

        paths, rows = [], []
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc="  DICOM→PNG"):
            p, s, d = str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"])
            dcm = Path(dcfg["dicom_dir"]) / f"p{p[:2]}" / f"p{p}" / f"s{s}" / f"{d}.dcm"
            out = Path(dcfg["output_dir"]) / f"p{p[:2]}" / f"p{p}" / f"s{s}" / f"{d}.png"
            if dicom_to_png(dcm, out, cfg["target_size"]):
                paths.append(str(out)); rows.append(row)

        df = pd.DataFrame(rows).reset_index(drop=True)
        df["image_path"]  = paths
        df["domain_id"]   = dcfg["domain_id"]
        df["dataset"]     = "mimic"
        df["report_text"] = df.apply(
            lambda r: _load_report(dcfg["reports_dir"], r["subject_id"], r["study_id"]), axis=1)
        print(f"  ✓ MIMIC: {len(df):,} samples")
        return df[["image_path","domain_id","dataset","report_text"] + HARMONIZED_LABELS]

    # Fallback processing for the custom "aug" dataset format (image paths lists + report text)
    # In this format, each CSV row has:
    #   image: stringified list of ALL image paths for this patient
    #   AP/PA/Lateral: stringified lists of image paths per view
    #   text: stringified list of report texts (one per study)
    #   text_augment: augmented/translated reports

    text_col = "text" if "text" in meta.columns else None
    if text_col is None:
        print("  ⚠ No 'text' column found in MIMIC CSV; skipping.")
        return None

    # Gather frontal (AP + PA) image paths from view-specific columns
    print("  Exploding per-image rows from aug CSV...")
    exploded_rows = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="  Parsing rows"):
        # Collect frontal image paths (AP first, then PA)
        frontal_images = []
        if "AP" in meta.columns:
            frontal_images.extend(_parse_list_field(row.get("AP", "[]")))
        if "PA" in meta.columns:
            frontal_images.extend(_parse_list_field(row.get("PA", "[]")))
        if not frontal_images:
            # Fall back to all images if no AP/PA columns
            frontal_images = _parse_list_field(row.get("image", "[]"))

        # Get report texts (list of strings)
        texts = _parse_list_field(row.get(text_col, "[]"))
        # Combine all texts into one report for this patient
        combined_text = " ".join(str(t) for t in texts if t and str(t).strip())
        if not combined_text.strip():
            combined_text = "Chest radiograph. No report available."

        # Create one row per frontal image
        for img_rel in frontal_images:
            img_rel = str(img_rel).strip()
            if not img_rel:
                continue
            src = Path(dcfg["dicom_dir"]) / img_rel
            if src.exists():
                exploded_rows.append({
                    "image_rel": img_rel,
                    "subject_id": row.get("subject_id", ""),
                    "report_text_raw": combined_text,
                })

    if not exploded_rows:
        print("  ⚠ No valid images found on disk; skipping.")
        return None

    edf = pd.DataFrame(exploded_rows)
    print(f"  Found {len(edf):,} frontal images on disk (from {len(meta):,} CSV rows)")

    # Sample down if needed
    n = min(len(edf), cfg["max_samples_per_dataset"])
    edf = edf.sample(n, random_state=42).reset_index(drop=True)

    # Resize and save each image
    paths, valid_rows = [], []
    for _, row in tqdm(edf.iterrows(), total=len(edf), desc="  Resizing"):
        src = Path(dcfg["dicom_dir"]) / row["image_rel"]
        dst = Path(dcfg["output_dir"]) / row["image_rel"]
        if resize_and_save(src, dst, cfg["target_size"]):
            paths.append(str(dst))
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows).reset_index(drop=True)
    df["image_path"]  = paths
    df["domain_id"]   = dcfg["domain_id"]
    df["dataset"]     = "mimic"
    df["report_text"] = df["report_text_raw"]

    # ✅ Extract labels from radiology report text using NLP
    print("\n  Extracting labels from radiology reports...")
    df = extract_labels_for_df(df, text_col="report_text")

    print(f"\n  ✓ MIMIC (aug): {len(df):,} samples")
    return df[["image_path","domain_id","dataset","report_text"] + HARMONIZED_LABELS]


def preprocess_chexpert(cfg):
    print("\n[CheXpert] Preprocessing...")
    dcfg = cfg["datasets"]["chexpert"]
    if not dcfg["labels_csv"] or not Path(dcfg["labels_csv"]).exists():
        print("  ⚠ CheXpert labels not found, skipping.")
        return None

    df = pd.read_csv(dcfg["labels_csv"])
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"]
    df = apply_label_map(df, "chexpert")
    n  = min(len(df), cfg["max_samples_per_dataset"])
    df = df.sample(n, random_state=42).reset_index(drop=True)

    paths, rows = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Resizing"):
        src = Path(dcfg["image_dir"]) / row["Path"]
        dst = Path(dcfg["output_dir"]) / row["Path"]
        if resize_and_save(src, dst, cfg["target_size"]):
            paths.append(str(dst)); rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["image_path"]  = paths
    out["domain_id"]   = dcfg["domain_id"]
    out["dataset"]     = "chexpert"
    out["report_text"] = "Chest radiograph frontal view."
    print(f"  ✓ CheXpert: {len(out):,} samples")
    return out[["image_path","domain_id","dataset","report_text"] + HARMONIZED_LABELS]


def preprocess_chestxray14(cfg):
    print("\n[ChestX-ray14] Preprocessing...")
    dcfg = cfg["datasets"]["chestxray14"]
    if not dcfg["labels_csv"] or not Path(dcfg["labels_csv"]).exists():
        print("  ⚠ ChestX-ray14 labels not found, skipping.")
        return None

    df = pd.read_csv(dcfg["labels_csv"])
    img_col = "Image Index" if "Image Index" in df.columns else df.columns[0]

    # Parse pipe-separated findings
    for lbl in HARMONIZED_LABELS:
        df[lbl] = 0
    for idx, row in df.iterrows():
        findings = str(row.get("Finding Labels","")).split("|")
        for src, harm in LABEL_MAPS["chestxray14"].items():
            if src in findings:
                df.at[idx, harm] = 1

    n  = min(len(df), cfg["max_samples_per_dataset"])
    df = df.sample(n, random_state=42).reset_index(drop=True)

    paths, rows = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Resizing"):
        src = Path(dcfg["image_dir"]) / row[img_col]
        dst = Path(dcfg["output_dir"]) / row[img_col]
        if resize_and_save(src, dst, cfg["target_size"]):
            paths.append(str(dst)); rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["image_path"]  = paths
    out["domain_id"]   = dcfg["domain_id"]
    out["dataset"]     = "chestxray14"
    out["report_text"] = "PA chest X-ray. Findings noted."
    print(f"  ✓ ChestX-ray14: {len(out):,} samples")
    return out[["image_path","domain_id","dataset","report_text"] + HARMONIZED_LABELS]


def _load_report(reports_dir, subject_id, study_id):
    p = Path(reports_dir) / f"p{str(subject_id)[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"
    if p.exists():
        return re.sub(r'\n+', ' ', p.read_text(errors="ignore")).strip()[:256]
    return "Chest radiograph. No report available."


def preprocess_mimic_validate(cfg):
    """Preprocess the MIMIC validation CSV as a separate held-out set."""
    print("\n[MIMIC-CXR Validation] Preprocessing...")
    dcfg = cfg["datasets"]["mimic"]
    val_path = dcfg.get("validate_csv", "")
    if not val_path or not Path(val_path).exists():
        print("  No validation CSV found, skipping.")
        return None

    meta = pd.read_csv(val_path)
    text_col = "text" if "text" in meta.columns else None
    if text_col is None:
        print("  No 'text' column found; skipping validation set.")
        return None

    # Explode per-image rows, same strategy as train
    exploded_rows = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="  Parsing val rows"):
        frontal_images = []
        if "AP" in meta.columns:
            frontal_images.extend(_parse_list_field(row.get("AP", "[]")))
        if "PA" in meta.columns:
            frontal_images.extend(_parse_list_field(row.get("PA", "[]")))
        if not frontal_images:
            frontal_images = _parse_list_field(row.get("image", "[]"))

        texts = _parse_list_field(row.get(text_col, "[]"))
        combined_text = " ".join(str(t) for t in texts if t and str(t).strip())
        if not combined_text.strip():
            combined_text = "Chest radiograph. No report available."

        for img_rel in frontal_images:
            img_rel = str(img_rel).strip()
            if not img_rel:
                continue
            src = Path(dcfg["dicom_dir"]) / img_rel
            if src.exists():
                exploded_rows.append({
                    "image_rel": img_rel,
                    "report_text_raw": combined_text,
                })

    if not exploded_rows:
        print("  No valid validation images found; skipping.")
        return None

    edf = pd.DataFrame(exploded_rows)
    print(f"  Found {len(edf):,} validation images on disk")

    paths, valid_rows = [], []
    for _, row in tqdm(edf.iterrows(), total=len(edf), desc="  Resizing val"):
        src = Path(dcfg["dicom_dir"]) / row["image_rel"]
        dst = Path(dcfg["output_dir"]) / row["image_rel"]
        if resize_and_save(src, dst, cfg["target_size"]):
            paths.append(str(dst))
            valid_rows.append(row)

    if not valid_rows:
        print("  No valid validation images after resize; skipping.")
        return None

    df = pd.DataFrame(valid_rows).reset_index(drop=True)
    df["image_path"] = paths
    df["domain_id"]  = dcfg["domain_id"]
    df["dataset"]    = "mimic"
    df["report_text"] = df["report_text_raw"]

    print("  Extracting labels from validation reports...")
    df = extract_labels_for_df(df, text_col="report_text")
    print(f"  Validation set: {len(df):,} samples")
    return df[["image_path","domain_id","dataset","report_text"] + HARMONIZED_LABELS]


def preprocess_all(cfg):
    print("\n" + "="*60)
    print("STEP 1: Preprocessing All Datasets")
    print("="*60)
    dfs = {}
    for name, fn in [("mimic", preprocess_mimic),
                     ("chexpert", preprocess_chexpert),
                     ("chestxray14", preprocess_chestxray14)]:
        df = fn(cfg)
        if df is not None and len(df) > 0:
            out = Path(cfg["datasets"][name]["output_dir"]) / "manifest.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            dfs[name] = df
            print(f"  Saved: {out}")

    # Process validation CSV as held-out cross-validation set
    val_df = preprocess_mimic_validate(cfg)
    if val_df is not None and len(val_df) > 0:
        out = Path(cfg["datasets"]["mimic"]["output_dir"]) / "manifest_val.csv"
        val_df.to_csv(out, index=False)
        dfs["mimic_val"] = val_df
        print(f"  Saved: {out}")

    # Size summary
    print("\n  Size Summary:")
    print(f"  {'Dataset':<15} {'Samples':>8} {'Est. Size':>12}")
    print("  " + "-"*38)
    for name, df in dfs.items():
        est_mb = len(df) * 224 * 224 / (1024**2)
        print(f"  {name:<15} {len(df):>8,} {est_mb:>10.1f} MB")
    return dfs


# =============================================================================
# STEP 2 — DATASET CLASS
# =============================================================================

class CXRDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64, transform=None):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        enc = self.tokenizer(
            str(row["report_text"]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return (
            img,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor([float(row[l]) for l in HARMONIZED_LABELS], dtype=torch.float32),
            torch.tensor(int(row["domain_id"]), dtype=torch.long),
        )


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


# =============================================================================
# STEP 3 — LIGHTWEIGHT MODEL (CPU-optimized)
# =============================================================================

class GradRevFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None

class GradReverse(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, alpha=1.0):
        return GradRevFn.apply(x, alpha)


class ImageEncoder(nn.Module):
    """MobileNetV3-Small — only 2.5M params, fast on CPU"""
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Freeze early layers, train only last 3 blocks
        children = list(base.features.children())
        for layer in children[:-3]:
            for p in layer.parameters():
                p.requires_grad = False
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.encoder   = base
        self.projector = nn.Sequential(
            nn.Linear(in_f, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.projector(self.encoder(x))


class TextEncoder(nn.Module):
    """ClinicalBERT — only last 2 transformer layers trainable"""
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze everything
        for p in self.bert.parameters():
            p.requires_grad = False
        # Unfreeze last 2 layers only
        for layer in self.bert.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        bert_dim = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0, :])


class BidirectionalFusion(nn.Module):
    """Bidirectional cross-modal attention (image↔text)"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.img2txt = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.txt2img = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)

    def forward(self, img, txt):
        iq, tq = img.unsqueeze(1), txt.unsqueeze(1)
        if_att, _ = self.img2txt(iq, tq, tq)
        tf_att, _ = self.txt2img(tq, iq, iq)
        return self.norm1(if_att.squeeze(1) + img), self.norm2(tf_att.squeeze(1) + txt)


class DomainAdversary(nn.Module):
    """DANN adversarial domain classifier"""
    def __init__(self, in_dim, num_domains=3):
        super().__init__()
        self.grl = GradReverse()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_domains),
        )
    def forward(self, x, alpha=1.0):
        return self.net(self.grl(x, alpha))


class DatasetConditionedHead(nn.Module):
    """Per-dataset output bias for domain-specific calibration"""
    def __init__(self, in_dim, num_classes=10, num_domains=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head        = nn.Linear(in_dim // 2, num_classes)
        self.domain_bias = nn.Embedding(num_domains, num_classes)
        nn.init.zeros_(self.domain_bias.weight)  # Start neutral

    def forward(self, x, domain_ids):
        return self.head(self.shared(x)) + self.domain_bias(domain_ids)


class CrossDatasetCXRModel(nn.Module):
    """
    Full lightweight multimodal model for cross-dataset CXR diagnosis.
    Total trainable params: ~8–10M (CPU feasible)
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg["embed_dim"]
        self.img_enc  = ImageEncoder(D)
        self.txt_enc  = TextEncoder(cfg["text_encoder"], D)
        self.fusion   = BidirectionalFusion(D)
        self.domain_h = DomainAdversary(D * 2, cfg["num_domains"])
        self.clf      = DatasetConditionedHead(D * 2, cfg["num_classes"], cfg["num_domains"])

    def forward(self, imgs, iids, amask, domain_ids, alpha=1.0):
        img_f = self.img_enc(imgs)
        txt_f = self.txt_enc(iids, amask)
        img_f, txt_f = self.fusion(img_f, txt_f)
        combined     = torch.cat([img_f, txt_f], dim=-1)
        logits       = self.clf(combined, domain_ids)
        dom_logits   = self.domain_h(combined, alpha)
        return logits, dom_logits


# =============================================================================
# STEP 4 — EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path="best_cxr_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.counter    = 0
        self.best_auc   = 0.0
        self.stop       = False

    def __call__(self, val_auc, model, epoch):
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter  = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_auc": val_auc}, self.path)
            print(f"  ✓ Saved best model (AUC={val_auc:.4f})")
        else:
            self.counter += 1
            print(f"  ⏳ No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                print("  🛑 Early stopping triggered.")
                self.stop = True


# =============================================================================
# STEP 5 — TRAINING
# =============================================================================

def get_alpha(epoch, total):
    p = epoch / total
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def train_epoch(model, loaders, optimizer, criterion, dom_crit, cfg, epoch):
    model.train()
    alpha = get_alpha(epoch, cfg["epochs"])
    iters = {n: iter(l) for n, l in loaders.items()}
    steps = max(len(l) for l in loaders.values())
    totals = {"loss": 0, "cls": 0, "dom": 0, "n": 0}

    for step in tqdm(range(steps), desc=f"  Epoch {epoch:02d}", leave=False):
        for dname, it in iters.items():
            try:
                batch = next(it)
            except StopIteration:
                iters[dname] = iter(loaders[dname])
                batch = next(iters[dname])

            imgs, iids, amask, labels, domains = batch

            optimizer.zero_grad()
            logits, dom_log = model(imgs, iids, amask, domains, alpha)
            cls_loss = criterion(logits, labels)
            dom_loss = dom_crit(dom_log, domains)
            loss     = cls_loss + cfg["lambda_domain"] * dom_loss

            # Meta-learning on cross-domain batches every 5 steps
            if step % 5 == 0 and dname != "mimic":
                fast  = deepcopy(model)
                f_opt = torch.optim.SGD(
                    [p for p in fast.parameters() if p.requires_grad],
                    lr=cfg["meta_lr"]
                )
                for _ in range(cfg["meta_steps"]):
                    fl, _ = fast(imgs, iids, amask, domains)
                    fl_loss = criterion(fl, labels)
                    f_opt.zero_grad(); fl_loss.backward(); f_opt.step()
                meta_l, _ = fast(imgs, iids, amask, domains)
                loss = loss + cfg["lambda_meta"] * criterion(meta_l, labels)
                del fast; gc.collect()  # ✅ Free RAM immediately

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["cls"]  += cls_loss.item()
            totals["dom"]  += dom_loss.item()
            totals["n"]    += 1

    n = totals["n"]
    return totals["loss"]/n, totals["cls"]/n, totals["dom"]/n


@torch.no_grad()
def evaluate(model, loader, criterion, tag=""):
    model.eval()
    total_loss, all_p, all_l = 0.0, [], []
    for imgs, iids, amask, labels, domains in tqdm(loader, desc=f"  Eval {tag}", leave=False):
        logits, _ = model(imgs, iids, amask, domains)
        total_loss += criterion(logits, labels).item()
        all_p.append(torch.sigmoid(logits).numpy())
        all_l.append(labels.numpy())

    probs  = np.concatenate(all_p)
    truth  = np.concatenate(all_l)
    per_cls = {}
    for i, lbl in enumerate(HARMONIZED_LABELS):
        try:
            # AUC requires at least one positive and one negative sample
            if truth[:,i].sum() == 0 or truth[:,i].sum() == len(truth):
                per_cls[lbl] = None
            else:
                per_cls[lbl] = round(float(roc_auc_score(truth[:,i], probs[:,i])), 4)
        except Exception:
            per_cls[lbl] = None
    valid = [v for v in per_cls.values() if v is not None]
    return total_loss/len(loader), float(np.mean(valid)) if valid else 0.0, per_cls


def build_loaders(dfs, tokenizer, cfg, split="train"):
    loaders = {}
    tr = get_transforms(split == "train")
    for name, df in dfs.items():
        ds = CXRDataset(df, tokenizer, cfg["max_text_len"], tr)
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["num_workers"],  # 0 on Windows
            pin_memory=False,                # CPU only
        )
    return loaders


def train(cfg, dfs):
    print("\n" + "="*60)
    print("STEP 5: Training")
    print(f"  Device        : CPU (Ryzen 7 7730U)")
    print(f"  Epochs        : {cfg['epochs']} (+ early stopping)")
    print(f"  Batch size    : {cfg['batch_size']}")
    print(f"  Embed dim     : {cfg['embed_dim']}")
    print(f"  Max text len  : {cfg['max_text_len']}")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"])

    # ── Build splits ────────────────────────────────────────────────
    train_dfs, val_dfs, crossval_dfs, crosstest_dfs = {}, {}, {}, {}

    for name, df in dfs.items():
        # mimic_val is a special held-out validation set
        if name == "mimic_val":
            crossval_dfs[name] = df
            continue
        role = cfg["datasets"].get(name, {}).get("role", "train")
        if role == "train":
            tr, va = train_test_split(df, test_size=0.15, random_state=42)
            train_dfs[name] = tr
            val_dfs[name]   = va
        elif role == "cross_val":
            crossval_dfs[name] = df
        elif role == "cross_test":
            crosstest_dfs[name] = df

    print(f"\n  Train  : {sum(len(d) for d in train_dfs.values()):,}")
    print(f"  Val    : {sum(len(d) for d in val_dfs.values()):,}")
    print(f"  C-Val  : {sum(len(d) for d in crossval_dfs.values()):,}")
    print(f"  C-Test : {sum(len(d) for d in crosstest_dfs.values()):,}")

    train_loaders   = build_loaders(train_dfs,    tokenizer, cfg, "train")
    val_loader      = build_loaders(val_dfs,      tokenizer, cfg, "val")
    crossval_loader = build_loaders(crossval_dfs, tokenizer, cfg, "val") if crossval_dfs else None
    crosstest_loader= build_loaders(crosstest_dfs,tokenizer, cfg, "val") if crosstest_dfs else None

    # Flatten val loaders into one combined loader
    all_val_df = pd.concat(list(val_dfs.values())).reset_index(drop=True)
    val_loader = DataLoader(
        CXRDataset(all_val_df, tokenizer, cfg["max_text_len"], get_transforms(False)),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=False
    )

    # ── Model ──────────────────────────────────────────────────────
    model     = CrossDatasetCXRModel(cfg)

    # ✅ Compute pos_weight for balanced BCE — handles label imbalance
    all_train_df = pd.concat(list(train_dfs.values())).reset_index(drop=True)
    pos_counts = all_train_df[HARMONIZED_LABELS].sum().values.astype(np.float32)
    neg_counts = len(all_train_df) - pos_counts
    # Avoid division by zero: if a class has 0 positives, use weight=1
    pos_weights = np.where(pos_counts > 0, neg_counts / (pos_counts + 1e-6), 1.0)
    # Clip extreme weights to avoid instability
    pos_weights = np.clip(pos_weights, 0.5, 20.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"\n  Pos weights (balanced BCE): {[f'{w:.1f}' for w in pos_weights]}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    dom_crit  = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    stopper     = EarlyStopping(cfg["early_stop_patience"], cfg["min_delta"], cfg["model_save_path"])

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n  Total params     : {total_p:.2f}M")
    print(f"  Trainable params : {train_p:.2f}M")
    print(f"\n  ⏱  Estimated time/epoch on Ryzen 7 7730U: ~{int(sum(len(d) for d in train_dfs.values()) * 0.4 / 60)} min")

    # ── Training loop ─────────────────────────────────────────────
    history = []
    print("\n  " + "─"*58)
    print(f"  {'Ep':>3} {'TrLoss':>8} {'ValLoss':>8} {'ValAUC':>8} {'CValAUC':>9} {'Time':>7}")
    print("  " + "─"*58)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, cls_l, dom_l = train_epoch(
            model, train_loaders, optimizer, criterion, dom_crit, cfg, epoch)
        val_loss, val_auc, val_pc = evaluate(model, val_loader, criterion, "val")
        scheduler.step()
        elapsed = int(time.time() - t0)

        cv_auc = None
        if crossval_loader:
            all_cv_df = pd.concat(list(crossval_dfs.values())).reset_index(drop=True)
            cv_loader = DataLoader(
                CXRDataset(all_cv_df, tokenizer, cfg["max_text_len"], get_transforms(False)),
                batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
            _, cv_auc, _ = evaluate(model, cv_loader, criterion, "CheXpert")

        row = {
            "epoch": epoch, "tr_loss": round(tr_loss,4),
            "val_loss": round(val_loss,4), "val_auc": round(val_auc,4),
            "cv_auc": round(cv_auc,4) if cv_auc else None,
            "elapsed_s": elapsed,
        }
        history.append(row)

        cv_str = f"{cv_auc:.4f}" if cv_auc else "  N/A "
        print(f"  {epoch:>3} {tr_loss:>8.4f} {val_loss:>8.4f} {val_auc:>8.4f} {cv_str:>9} {elapsed:>5}s")

        stopper(val_auc, model, epoch)
        if stopper.stop:
            break

        gc.collect()  # Free RAM every epoch

    # ── Final evaluation ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Final Cross-Dataset Evaluation")
    print("="*60)

    if os.path.exists(cfg["model_save_path"]):
        ckpt = torch.load(cfg["model_save_path"], map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print("  Loaded best checkpoint for final evaluation.")
    else:
        print("  ⚠ No checkpoint saved (AUC may have been NaN). Using current model weights.")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_auc": 0.0}, cfg["model_save_path"])

    results = {"history": history, "per_dataset": {}}

    eval_sets = [("MIMIC (in-domain val)", val_loader, None)]
    if crossval_dfs:
        all_cv = pd.concat(list(crossval_dfs.values())).reset_index(drop=True)
        eval_sets.append(("CheXpert (cross-val)", DataLoader(
            CXRDataset(all_cv, tokenizer, cfg["max_text_len"], get_transforms(False)),
            batch_size=cfg["batch_size"], shuffle=False, num_workers=0), None))
    if crosstest_dfs:
        all_ct = pd.concat(list(crosstest_dfs.values())).reset_index(drop=True)
        eval_sets.append(("ChestX-ray14 (cross-test)", DataLoader(
            CXRDataset(all_ct, tokenizer, cfg["max_text_len"], get_transforms(False)),
            batch_size=cfg["batch_size"], shuffle=False, num_workers=0), None))

    for tag, loader, _ in eval_sets:
        loss, auc, per_cls = evaluate(model, loader, criterion, tag)
        safe_auc = auc if not (isinstance(auc, float) and np.isnan(auc)) else 0.0
        results["per_dataset"][tag] = {"loss": round(loss,4), "mean_auc": round(safe_auc,4), "per_class": per_cls}
        print(f"\n  [{tag}]")
        print(f"    Mean AUC : {safe_auc:.4f}")
        print(f"    {'Label':<28} {'AUC':>6}  {'Bar'}")
        print(f"    {'─'*55}")
        for lbl, a in per_cls.items():
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                bar = "█" * int(a * 20)
                print(f"    {lbl:<28} {a:.4f}  {bar}")
            else:
                print(f"    {lbl:<28}    N/A")

    with open(cfg["results_path"], "w") as f:
        json.dump(results, f, indent=2)

    # ── Training summary table ────────────────────────────────────
    print("\n" + "="*60)
    print("  Training Summary")
    print("="*60)
    print(f"  {'Ep':>3} {'TrLoss':>8} {'ValAUC':>8} {'CValAUC':>9}")
    print("  " + "─"*35)
    for r in history:
        cv = f"{r['cv_auc']:.4f}" if r['cv_auc'] else "  N/A "
        print(f"  {r['epoch']:>3} {r['tr_loss']:>8.4f} {r['val_auc']:>8.4f} {cv:>9}")
    print(f"\n  Best Val AUC   : {stopper.best_auc:.4f}")
    print(f"  Results saved  : {cfg['results_path']}")
    print(f"  Model saved    : {cfg['model_save_path']}")
    return model, results


# =============================================================================
# STEP 7 — INFERENCE
# =============================================================================

def predict(model, image_path, report_text, tokenizer, cfg, domain_id=0):
    model.eval()
    img = get_transforms(False)(Image.open(image_path).convert("RGB")).unsqueeze(0)
    enc = tokenizer(report_text, max_length=cfg["max_text_len"],
                    padding="max_length", truncation=True, return_tensors="pt")
    dom = torch.tensor([domain_id], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(img, enc["input_ids"], enc["attention_mask"], dom)
        probs = torch.sigmoid(logits).numpy()[0]
    return dict(sorted(
        {l: round(float(p), 4) for l, p in zip(HARMONIZED_LABELS, probs)}.items(),
        key=lambda x: x[1], reverse=True
    ))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preprocess","train","infer","all"], default="all")
    parser.add_argument("--image",     default="")
    parser.add_argument("--report",    default="")
    parser.add_argument("--domain_id", type=int, default=0,
                        help="0=MIMIC  1=CheXpert  2=ChestX-ray14")
    args = parser.parse_args()

    dfs = {}

    if args.mode in ("preprocess", "all"):
        dfs = preprocess_all(CONFIG)

    if args.mode in ("train", "all"):
        if args.mode == "train":
            for name, dcfg in CONFIG["datasets"].items():
                m = Path(dcfg["output_dir"]) / "manifest.csv"
                if m.exists():
                    df = pd.read_csv(m)
                    for col in HARMONIZED_LABELS:
                        if col in df.columns:
                            df[col] = df[col].fillna(0).astype(int)
                    dfs[name] = df
                    print(f"  Loaded {name}: {len(df):,} rows")
        if dfs:
            model, results = train(CONFIG, dfs)
        else:
            print("No datasets found. Run --mode preprocess first.")

    if args.mode == "infer":
        if not args.image:
            print("Provide --image <path>")
        else:
            tok   = AutoTokenizer.from_pretrained(CONFIG["text_encoder"])
            model = CrossDatasetCXRModel(CONFIG)
            ckpt  = torch.load(CONFIG["model_save_path"], map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            preds = predict(model, args.image,
                            args.report or "Chest radiograph frontal view.",
                            tok, CONFIG, args.domain_id)
            print("\n  Predicted Pathology Probabilities:")
            print("  " + "─"*45)
            for lbl, p in preds.items():
                bar = "█" * int(p * 25)
                print(f"  {lbl:<30} {p:.4f}  {bar}")

# =============================================================================
# QUICK START (Windows CMD / PowerShell):
#
#   pip install pydicom pillow numpy pandas tqdm torch torchvision transformers scikit-learn
#
#   Step 1 - Edit the CONFIG paths at the top of this file
#
#   Step 2 - Preprocess all datasets:
#     python cxr_cross_dataset_cpu_optimized.py --mode preprocess
#
#   Step 3 - Train:
#     python cxr_cross_dataset_cpu_optimized.py --mode train
#
#   Step 4 - Run inference:
#     python cxr_cross_dataset_cpu_optimized.py --mode infer
#       --image C:/path/chest.png
#       --report "Patient has shortness of breath, bilateral infiltrates"
#       --domain_id 0
#
# ─────────────────────────────────────────────────────────────────
# SYSTEM-SPECIFIC OPTIMIZATIONS (HP ENVY x360 / Ryzen 7 7730U):
#   ✅ DEVICE forced to CPU (GPU has only 496MB)
#   ✅ num_workers=0 (Windows multiprocessing fix)
#   ✅ batch_size=4 (fits in 16GB RAM)
#   ✅ embed_dim=128 (half of original 256)
#   ✅ max_text_len=64 (halves BERT memory usage)
#   ✅ MobileNetV3-Small backbone (2.5M params vs 5.3M EfficientNet-B0)
#   ✅ Only last 2 BERT layers trainable
#   ✅ gc.collect() after meta-learning to free RAM
#   ✅ Early stopping (patience=5) to avoid wasted CPU hours
#   ✅ max_samples=3000 per dataset (~2GB RAM total)
#
# EXPECTED TRAINING TIME (Ryzen 7 7730U, CPU-only):
#   Per epoch (3000×3 samples, batch=4) : ~25–40 min
#   Full 20 epochs                      : ~8–12 hours
#   With early stopping (stops at ~10)  : ~4–6 hours
#   💡 TIP: Run overnight or use Google Colab for GPU
# =============================================================================

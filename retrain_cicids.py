"""
retrain_cicids.py
=================
Retrain LogSentinel's Random Forest model using the CICIDS-2017 dataset.

Steps performed:
  1. Extract MachineLearningCSV.zip  (skipped if already extracted)
  2. Load all 8 CSV files
  3. Clean column names and drop bad rows
  4. Select 10 best features common across all files
  5. Balance classes with random undersampling
  6. Train Random Forest (500 trees)
  7. Evaluate and print classification report
  8. Save model to models/http_random_forest.joblib
     (overwrites the old model — your existing pipeline picks it up automatically)

Run:
    python retrain_cicids.py
"""

import glob
import os
import sys
import zipfile
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
ZIP_PATH   = BASE_DIR / "MachineLearningCSV.zip"
EXTRACT_DIR = BASE_DIR / "cicids2017"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "http_random_forest.joblib"

# ─────────────────────────────────────────────
#  Features to use from CICIDS-2017
#  (these 10 columns exist in all 8 CSV files)
# ─────────────────────────────────────────────
CICIDS_FEATURES = [
    " Flow Duration",          # how long the connection lasted (µs)
    " Total Fwd Packets",      # packets sent forward
    " Total Backward Packets", # packets sent back
    " Total Length of Fwd Packets",  # bytes forward
    " Total Length of Bwd Packets",  # bytes backward
    " Flow Bytes/s",           # throughput
    " Flow Packets/s",         # packet rate
    " Average Packet Size",    # avg bytes per packet
    " Avg Fwd Segment Size",   # avg forward payload
    " Avg Bwd Segment Size",   # avg backward payload
]

# Rename to clean names for our pipeline
FEATURE_NAMES = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "avg_packet_size",
    "avg_fwd_segment",
    "avg_bwd_segment",
]

LABEL_COL = " Label"   # CICIDS uses a space prefix

# Attack label normalisation — map CICIDS labels → our standard labels
LABEL_MAP = {
    "BENIGN":                       "BENIGN",
    "DoS Hulk":                     "DOS",
    "DoS GoldenEye":                "DOS",
    "DoS slowloris":                "DOS",
    "DoS Slowhttptest":             "DOS",
    "Heartbleed":                   "DOS",
    "DDoS":                         "DDOS",
    "PortScan":                     "PORT_SCAN",
    "FTP-Patator":                  "BRUTE_FORCE",
    "SSH-Patator":                  "BRUTE_FORCE",
    "Bot":                          "BOTNET",
    "Infiltration":                 "INFILTRATION",
    "Web Attack \x96 Brute Force":  "WEB_BRUTE_FORCE",
    "Web Attack – Brute Force":     "WEB_BRUTE_FORCE",
    "Web Attack \x96 XSS":          "XSS",
    "Web Attack – XSS":             "XSS",
    "Web Attack \x96 Sql Injection":"SQLI",
    "Web Attack – Sql Injection":   "SQLI",
}


def banner(msg: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


# ─────────────────────────────────────────────
#  STEP 1: Extract zip
# ─────────────────────────────────────────────
def extract_zip() -> Path:
    csv_glob = list(EXTRACT_DIR.glob("MachineLearningCVE/*.csv"))
    if csv_glob:
        print(f"[✓] Already extracted — found {len(csv_glob)} CSV files in {EXTRACT_DIR}")
        return EXTRACT_DIR / "MachineLearningCVE"

    if not ZIP_PATH.exists():
        print(f"[✗] Cannot find: {ZIP_PATH}")
        print("    Download MachineLearningCSV.zip from the CIC-IDS-2017 page first.")
        sys.exit(1)

    print(f"[*] Extracting {ZIP_PATH.name}  …  (this may take a minute)")
    EXTRACT_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    csv_glob = list(EXTRACT_DIR.glob("MachineLearningCVE/*.csv"))
    print(f"[✓] Extracted {len(csv_glob)} CSV files to {EXTRACT_DIR}")
    return EXTRACT_DIR / "MachineLearningCVE"


# ─────────────────────────────────────────────
#  STEP 2: Load and merge all CSVs
# ─────────────────────────────────────────────
def load_cicids(csv_dir: Path) -> pd.DataFrame:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        print(f"[✗] No CSV files found in {csv_dir}")
        sys.exit(1)

    frames = []
    for f in files:
        print(f"    Loading {f.name}  …", end="", flush=True)
        try:
            df = pd.read_csv(f, low_memory=False)
            print(f"  {len(df):,} rows")
            frames.append(df)
        except Exception as e:
            print(f"  ERROR: {e} — skipping")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n[✓] Total rows loaded: {len(combined):,}")
    return combined


# ─────────────────────────────────────────────
#  STEP 3: Clean and select features
# ─────────────────────────────────────────────
def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Check required columns exist
    missing_cols = [c for c in CICIDS_FEATURES + [LABEL_COL] if c not in df.columns]
    if missing_cols:
        print(f"[!] Missing columns (will try with available): {missing_cols}")
        available = [c for c in CICIDS_FEATURES if c in df.columns]
    else:
        available = CICIDS_FEATURES

    X = df[available].copy()
    y = df[LABEL_COL].str.strip()

    # Rename to clean names
    rename_map = {old: new for old, new in zip(CICIDS_FEATURES, FEATURE_NAMES) if old in available}
    X.rename(columns=rename_map, inplace=True)

    # Normalise labels
    y = y.map(lambda v: LABEL_MAP.get(v, v))

    # Drop rows with NaN, inf, or unknown labels
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    # Drop rows with negative values in time/byte columns (CICIDS artefacts)
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X = X[X[col] >= 0]
            y = y.loc[X.index]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"\n[✓] Clean rows: {len(X):,}")
    print("\n--- Attack Label Distribution (before balancing) ---")
    for label, count in y.value_counts().items():
        bar = "█" * min(40, int(count / y.value_counts().max() * 40))
        print(f"  {str(label):<22} {count:>8,}  {bar}")

    return X, y


# ─────────────────────────────────────────────
#  STEP 4: Balance classes
# ─────────────────────────────────────────────
def balance(X: pd.DataFrame, y: pd.Series, max_per_class: int = 80_000) -> tuple:
    """Random undersampling: cap each class at max_per_class rows."""
    frames_X, frames_y = [], []
    for label in y.unique():
        idx = y[y == label].index
        if len(idx) > max_per_class:
            idx = idx[:max_per_class]
        frames_X.append(X.loc[idx])
        frames_y.append(y.loc[idx])

    Xb = pd.concat(frames_X).sample(frac=1, random_state=42).reset_index(drop=True)
    yb = pd.concat(frames_y).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n[✓] Balanced dataset: {len(Xb):,} rows  (max {max_per_class:,} per class)")
    return Xb, yb


# ─────────────────────────────────────────────
#  STEP 5 + 6: Train and evaluate
# ─────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[*] Training Random Forest …")
    print(f"    Training rows : {len(X_train):,}")
    print(f"    Testing  rows : {len(X_test):,}")
    print(f"    Features      : {list(X.columns)}")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print("[✓] Training complete.")

    preds = clf.predict(X_test)

    acc  = round(float(accuracy_score(y_test, preds)), 4)
    f1   = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
    prec = round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4)
    rec  = round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4)

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"  Accuracy           : {acc * 100:.2f}%")
    print(f"  Weighted Precision : {prec * 100:.2f}%")
    print(f"  Weighted Recall    : {rec * 100:.2f}%")
    print(f"  Weighted F1        : {f1 * 100:.2f}%")

    print("\n--- FEATURE IMPORTANCE ---")
    importances = clf.feature_importances_
    for feat, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"  {feat:<28} {imp*100:5.2f}%  {bar}")

    return {
        "model":      clf,
        "features":   list(X.columns),
        "classes":    list(clf.classes_),
        "evaluation": {
            "accuracy":            acc,
            "precision_weighted":  prec,
            "recall_weighted":     rec,
            "f1_weighted":         f1,
        },
        "training_rows": int(len(X_train)),
        "test_rows":     int(len(X_test)),
        "training_label_counts": {str(k): int(v) for k, v in Counter(y_train).items()},
        "source": "CICIDS-2017 MachineLearningCSV — 8 daily CSV files",
    }


# ─────────────────────────────────────────────
#  STEP 7: Save model
# ─────────────────────────────────────────────
def save(artifact: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    # Also save a companion metadata file for the UI /features endpoint
    meta = {k: v for k, v in artifact.items() if k != "model"}
    joblib.dump(meta, MODEL_DIR / "cicids_meta.joblib")

    print(f"\n[✓] Model saved  →  {MODEL_PATH}")
    print(f"[✓] Meta  saved  →  {MODEL_DIR / 'cicids_meta.joblib'}")
    print("\n    Your existing pipeline (server.py / app_ui.py) will")
    print("    automatically use this new model on the next upload.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    banner("CICIDS-2017 Retrain Script — LogSentinel AI")

    banner("Step 1 — Extracting zip")
    csv_dir = extract_zip()

    banner("Step 2 — Loading CSV files")
    df = load_cicids(csv_dir)

    banner("Step 3 — Cleaning & selecting features")
    X, y = prepare(df)

    banner("Step 4 — Balancing classes (max 80,000 per class)")
    X, y = balance(X, y, max_per_class=80_000)

    banner("Step 5 & 6 — Training Random Forest (500 trees)")
    artifact = train(X, y)

    banner("Step 7 — Saving model")
    save(artifact)

    banner("DONE ✓")
    print(f"\n  Accuracy : {artifact['evaluation']['accuracy']*100:.2f}%")
    print(f"  F1 Score : {artifact['evaluation']['f1_weighted']*100:.2f}%")
    print(f"  Classes  : {artifact['classes']}")
    print(f"\n  Run your server:  python server.py")
    print(f"  Then upload any log at http://localhost:8000")

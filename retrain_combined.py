"""
retrain_combined.py
===================
Trains a unified LogSentinel Random Forest model by combining:

  Source A — Your own engineered log dataset (Practical 1-6 pipeline)
             Columns: ip_request_count, time_diff_seconds,
                      status_code_404_count, is_bot_agent, url_entropy, label
             File   : engineered_features.csv  (2M rows)

  Source B — CICIDS-2017 (MachineLearningCSV.zip)
             8 CSV files, 2.8M rows, 10 attack types
             Features mapped to share-able common columns

  Source C — Logs/FINAL_UPDATED_LABELED.csv
             Your additional labelled Apache/JSON logs (2M rows)

Shared Feature Set (works across all sources):
  ip_request_count, url_entropy, is_bot_agent,
  time_diff_seconds, status_code_404_count,
  flow_duration, fwd_bytes, bwd_bytes,
  flow_pkts_per_sec, avg_packet_size

Run:
    python retrain_combined.py

Output:
    models/http_random_forest.joblib   ← replaces old model automatically
    models/combined_meta.joblib        ← metadata for /features API
"""

import glob
import math
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.resolve()
LOGS_DIR = BASE / "Logs"
ZIP_PATH = BASE / "MachineLearningCSV.zip"
CICIDS_DIR = BASE / "cicids2017" / "MachineLearningCVE"
MODEL_DIR  = BASE / "models"
MODEL_PATH = MODEL_DIR / "http_random_forest.joblib"

# ─────────────────────────────────────────────────────────────
#  SHARED FEATURE SET
#  These 10 columns will exist in the final training frame.
#  Sources that don't have a column will get a sensible default.
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "ip_request_count",      # requests per IP (all sources)
    "url_entropy",           # Shannon entropy of URL (Source A/C)
    "is_bot_agent",          # known scanner UA (Source A/C)
    "time_diff_seconds",     # Δt between requests (Source A/C)
    "status_code_404_count", # 404 frequency per IP (Source A/C)
    "flow_duration",         # connection duration µs (CICIDS)
    "fwd_bytes",             # forward payload bytes (CICIDS)
    "bwd_bytes",             # backward payload bytes (CICIDS)
    "flow_pkts_per_sec",     # packet rate (CICIDS)
    "avg_packet_size",       # average packet size (CICIDS)
]

MAX_PER_CLASS = 100_000   # rows per attack label (balancing cap)

# ─────────────────────────────────────────────────────────────
#  Label normalisation
# ─────────────────────────────────────────────────────────────
YOUR_LABEL_MAP = {
    # your pipeline labels → unified
    "benign":          "BENIGN",
    "brute_force":     "BRUTE_FORCE",
    "sqli":            "SQLI",
    "path_traversal":  "PATH_TRAVERSAL",
    "BENIGN":          "BENIGN",
    "BRUTE_FORCE":     "BRUTE_FORCE",
    "SQLI":            "SQLI",
    "PATH_TRAVERSAL":  "PATH_TRAVERSAL",
}

CICIDS_LABEL_MAP = {
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
    "Web Attack \x96 Brute Force":  "BRUTE_FORCE",
    "Web Attack \u2013 Brute Force":"BRUTE_FORCE",
    "Web Attack – Brute Force":     "BRUTE_FORCE",
    "Web Attack \x96 XSS":          "XSS",
    "Web Attack \u2013 XSS":        "XSS",
    "Web Attack – XSS":             "XSS",
    "Web Attack \x96 Sql Injection":"SQLI",
    "Web Attack \u2013 Sql Injection":"SQLI",
    "Web Attack – Sql Injection":   "SQLI",
}

LOGS_LABEL_MAP = {
    # FINAL_UPDATED_LABELED.csv final_label column
    "Browser":          "BENIGN",
    "Bot":              "BOTNET",
    "Scanner":          "PORT_SCAN",
    "Crawler":          "BOTNET",
    "Attacker":         "BRUTE_FORCE",
    "Unknown":          "BENIGN",   # conservative — treat unknowns as benign
}

BOT_TERMS = ("bot", "crawler", "spider", "python", "curl", "wget",
             "nikto", "sqlmap", "nmap", "hydra", "masscan", "zgrab")


def entropy(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    t = str(text)
    return -sum((t.count(c) / len(t)) * math.log2(t.count(c) / len(t)) for c in set(t))


def banner(msg: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {msg}")
    print("=" * 64)


def fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every FEATURES column exists; fill missing with 0."""
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURES + ["label"]].copy()


# ─────────────────────────────────────────────────────────────
#  SOURCE A — engineered_features.csv  (your Practical 1-6 data)
# ─────────────────────────────────────────────────────────────
def load_source_a() -> pd.DataFrame:
    path = BASE / "engineered_features.csv"
    if not path.exists():
        print(f"  [!] engineered_features.csv not found — skipping Source A")
        return pd.DataFrame()

    print(f"  Loading {path.name}  …", end="", flush=True)
    df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in
                     ["ip_request_count","time_diff_seconds","status_code_404_count",
                      "is_bot_agent","url_entropy","label"])
    print(f"  {len(df):,} rows")

    df["label"] = df["label"].astype(str).str.strip().str.lower().map(YOUR_LABEL_MAP)
    df = df.dropna(subset=["label"])

    # CICIDS-only features → default 0
    df = fill_defaults(df)
    return df


# ─────────────────────────────────────────────────────────────
#  SOURCE B — CICIDS-2017
# ─────────────────────────────────────────────────────────────
CICIDS_COL_MAP = {
    " Flow Duration":               "flow_duration",
    " Total Fwd Packets":           "ip_request_count",   # reuse as request count proxy
    " Total Backward Packets":      "bwd_packets",
    " Total Length of Fwd Packets": "fwd_bytes",
    " Total Length of Bwd Packets": "bwd_bytes",
    " Flow Bytes/s":                "flow_bytes_per_sec",
    " Flow Packets/s":              "flow_pkts_per_sec",
    " Average Packet Size":         "avg_packet_size",
    " Avg Fwd Segment Size":        "avg_fwd_segment",
    " Avg Bwd Segment Size":        "avg_bwd_segment",
    " Label":                       "label_raw",
}


def extract_cicids_zip() -> bool:
    if CICIDS_DIR.exists() and list(CICIDS_DIR.glob("*.csv")):
        print(f"  [✓] Already extracted — {len(list(CICIDS_DIR.glob('*.csv')))} files in {CICIDS_DIR}")
        return True
    if not ZIP_PATH.exists():
        print(f"  [!] {ZIP_PATH.name} not found — skipping CICIDS")
        return False
    print(f"  Extracting {ZIP_PATH.name}  …  (may take ~1 min)")
    CICIDS_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(BASE / "cicids2017")
    print(f"  [✓] Extracted {len(list(CICIDS_DIR.glob('*.csv')))} files")
    return True


def load_source_b() -> pd.DataFrame:
    if not extract_cicids_zip():
        return pd.DataFrame()

    frames = []
    for f in sorted(CICIDS_DIR.glob("*.csv")):
        print(f"  Loading {f.name}  …", end="", flush=True)
        try:
            raw = pd.read_csv(f, low_memory=False, usecols=list(CICIDS_COL_MAP.keys()))
            raw.rename(columns=CICIDS_COL_MAP, inplace=True)
            raw["label"] = raw["label_raw"].astype(str).str.strip().map(CICIDS_LABEL_MAP)
            raw.drop(columns=["label_raw"], inplace=True, errors="ignore")

            # Replace inf
            raw.replace([np.inf, -np.inf], np.nan, inplace=True)
            raw.dropna(subset=["label"], inplace=True)

            # Your-pipeline features not in CICIDS → default 0
            raw["url_entropy"]           = 0.0
            raw["is_bot_agent"]          = 0
            raw["time_diff_seconds"]     = 0.0
            raw["status_code_404_count"] = 0

            raw = fill_defaults(raw)
            print(f"  {len(raw):,} rows")
            frames.append(raw)
        except Exception as e:
            print(f"  ERROR: {e} — skipping")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
#  SOURCE C — Logs/FINAL_UPDATED_LABELED.csv + others
# ─────────────────────────────────────────────────────────────
def load_source_c() -> pd.DataFrame:
    target = LOGS_DIR / "FINAL_UPDATED_LABELED.csv"
    if not target.exists():
        print(f"  [!] {target.name} not found — skipping Source C")
        return pd.DataFrame()

    print(f"  Loading {target.name}  …", end="", flush=True)
    df = pd.read_csv(target, low_memory=False)
    print(f"  {len(df):,} rows")

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Find IP column
    ip_col = next((c for c in df.columns if "ip" in c and "address" in c), None) or \
             next((c for c in df.columns if "client" in c and "ip" in c), None)

    # Find label column
    lbl_col = next((c for c in df.columns if "final_label" in c), None) or \
              next((c for c in df.columns if "label" in c), None)

    # Find UA column
    ua_col  = next((c for c in df.columns if "browser" in c or "user_agent" in c or "agent" in c), None)

    if not lbl_col:
        print("  [!] No label column found — skipping Source C")
        return pd.DataFrame()

    out = pd.DataFrame()

    # Label — map to unified
    raw_label = df[lbl_col].astype(str).str.strip()
    # Keep only first word before comma for multi-label rows
    raw_label = raw_label.str.split(",").str[0].str.strip()
    out["label"] = raw_label.map(lambda v: LOGS_LABEL_MAP.get(v, None))
    out = out.dropna(subset=["label"])

    # IP request count (compute from IP column if available)
    if ip_col:
        ip_series = df.loc[out.index, ip_col].astype(str)
        counts = ip_series.map(ip_series.value_counts())
        out["ip_request_count"] = counts.values
    else:
        out["ip_request_count"] = 1

    # UA bot flag
    if ua_col:
        ua = df.loc[out.index, ua_col].astype(str).str.lower()
        out["is_bot_agent"] = ua.apply(lambda v: int(any(t in v for t in BOT_TERMS)))
    else:
        out["is_bot_agent"] = 0

    # Other features default
    out["url_entropy"]           = 0.0
    out["time_diff_seconds"]     = 0.0
    out["status_code_404_count"] = 0

    out = fill_defaults(out)
    print(f"  [✓] {len(out):,} usable rows from Source C")
    return out


# ─────────────────────────────────────────────────────────────
#  Balance (cap each class)
# ─────────────────────────────────────────────────────────────
def balance(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for lbl in df["label"].unique():
        chunk = df[df["label"] == lbl]
        if len(chunk) > MAX_PER_CLASS:
            chunk = chunk.sample(MAX_PER_CLASS, random_state=42)
        frames.append(chunk)
    out = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────
def train(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df["label"]

    # Final clean — drop inf / nan
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    # Clip extreme values
    for col in X.columns:
        q99 = X[col].quantile(0.99)
        if q99 > 0:
            X[col] = X[col].clip(upper=q99 * 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n  Training rows : {len(X_train):,}")
    print(f"  Testing  rows : {len(X_test):,}")
    print(f"  Features      : {list(X.columns)}")
    print(f"  Labels        : {sorted(y.unique())}")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print("  [✓] Training complete.")

    preds = clf.predict(X_test)
    acc   = round(float(accuracy_score(y_test, preds)), 4)
    f1    = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
    prec  = round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4)
    rec   = round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4)

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"  Accuracy           : {acc*100:.2f}%")
    print(f"  Weighted Precision : {prec*100:.2f}%")
    print(f"  Weighted Recall    : {rec*100:.2f}%")
    print(f"  Weighted F1        : {f1*100:.2f}%")

    print("\n--- FEATURE IMPORTANCE ---")
    for feat, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 80)
        print(f"  {feat:<28} {imp*100:5.2f}%  {bar}")

    return {
        "model":      clf,
        "features":   list(X.columns),
        "classes":    list(clf.classes_),
        "evaluation": {
            "accuracy":           acc,
            "precision_weighted": prec,
            "recall_weighted":    rec,
            "f1_weighted":        f1,
        },
        "training_rows": int(len(X_train)),
        "test_rows":     int(len(X_test)),
        "training_label_counts": {str(k): int(v) for k, v in Counter(y_train).items()},
        "source": "Combined: engineered_features.csv + CICIDS-2017 + Logs/FINAL_UPDATED_LABELED.csv",
    }


# ─────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────
def save(artifact: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    meta = {k: v for k, v in artifact.items() if k != "model"}
    joblib.dump(meta, MODEL_DIR / "combined_meta.joblib")
    print(f"\n  [✓] Model → {MODEL_PATH}")
    print(f"  [✓] Meta  → {MODEL_DIR / 'combined_meta.joblib'}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner("LogSentinel — Combined Retrain (A + B + C)")

    banner("Source A — Your engineered_features.csv")
    df_a = load_source_a()
    print(f"  Rows loaded: {len(df_a):,}")

    banner("Source B — CICIDS-2017 (MachineLearningCSV.zip)")
    df_b = load_source_b()
    print(f"  Rows loaded: {len(df_b):,}")

    banner("Source C — Logs/FINAL_UPDATED_LABELED.csv")
    df_c = load_source_c()
    print(f"  Rows loaded: {len(df_c):,}")

    banner("Combining all sources")
    frames = [f for f in [df_a, df_b, df_c] if len(f) > 0]
    if not frames:
        print("[✗] No data loaded from any source. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows (before balance): {len(combined):,}")

    print("\n--- Label distribution (before balance) ---")
    for lbl, cnt in combined["label"].value_counts().items():
        bar = "█" * min(50, int(cnt / combined["label"].value_counts().max() * 50))
        print(f"  {str(lbl):<22} {cnt:>9,}  {bar}")

    banner(f"Balancing (max {MAX_PER_CLASS:,} rows per class)")
    combined = balance(combined)
    print(f"  Total rows (after balance): {len(combined):,}")

    print("\n--- Label distribution (after balance) ---")
    for lbl, cnt in combined["label"].value_counts().items():
        print(f"  {str(lbl):<22} {cnt:>9,}")

    banner("Training Random Forest (500 trees, all cores)")
    artifact = train(combined)

    banner("Saving model")
    save(artifact)

    banner("ALL DONE ✓")
    ev = artifact["evaluation"]
    print(f"\n  ✅ Accuracy  : {ev['accuracy']*100:.2f}%")
    print(f"  ✅ F1 Score  : {ev['f1_weighted']*100:.2f}%")
    print(f"  ✅ Classes   : {artifact['classes']}")
    print(f"  ✅ Source    : {artifact['source']}")
    print(f"\n  Restart your server to use the new model:")
    print(f"  → python server.py")

"""
practical_10.py — Reusable Automated Log Processing Pipeline
=============================================================
Practical 10  (CO5)

Pipeline stages
---------------
1.  FORMAT DETECTION  — classify the log file using structural patterns
2.  FORMAT-SPECIFIC PARSING  — route to the correct parser (Apache / JSON-array /
    Oracle Data Loader / Java application / Generic)
3.  NORMALISED EVENTS  — unified internal representation
4.  VALIDATED METRICS  — never invent; mark fields as available / unavailable /
    not_applicable depending on the detected format
5.  FORMAT-SPECIFIC ANALYSIS  — HTTP stats for access logs; import timeline for
    Oracle; exception summary for Java application logs
6.  SECURITY ANALYSIS  — only when the log format contains network/security data
7.  ML INFERENCE  — only when the log contains labellable HTTP request data
8.  EXPORT  — CSV / Parquet for HTTP-format logs
"""

import os
import sys
import re
import json
import math
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from path_utils import resolve_input_file, resolve_export_file, BASE_DIR
from log_format_detector import detect_log_format_from_file
from log_parsers import parse_log


# ============================================================================
# Helper utilities
# ============================================================================

def _status_block(name: str, field_info: dict, value_override=None) -> str:
    """Return a formatted metric line that never hides N/A status."""
    status = field_info.get("status", "unknown")
    if status == "not_applicable":
        return f"  {name:<30} NOT APPLICABLE — {field_info.get('reason', '')}"
    if status == "not_available":
        return f"  {name:<30} UNAVAILABLE — {field_info.get('reason', '')}"
    val = value_override if value_override is not None else field_info.get("value", "—")
    return f"  {name:<30} {val}"


def _header(title: str) -> str:
    sep = "=" * 60
    return f"\n{sep}\n{title}\n{sep}"


# ============================================================================
# Apache / JSON-array HTTP pipeline (stages 3–8)
# ============================================================================

class HttpLogPipeline:
    """
    Handles Apache-access-log and JSON-array-format logs.
    Produces a labelled, feature-engineered DataFrame suitable for ML training.
    """

    APACHE_LOG_PATTERN = re.compile(
        r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
        r'"(?P<request>[^"]*)"\s+(?P<status_code>\d{3})\s+(?P<response_bytes>\S+)'
        r'(?:\s+"(?P<referrer>[^"]*)"\s+"(?P<user_agent>[^"]*)")?'
    )

    def __init__(self, filepath: str, fmt_info: dict):
        self.filepath    = filepath
        self.fmt_info    = fmt_info
        self.raw_df      = None
        self.clean_df    = None
        self.engineered_df = None
        self.model       = None

    # ------------------------------------------------------------------
    def ingest_and_parse(self, sample_limit=None):
        print(f"\n[*] Stage 1: Ingesting & Parsing  [{self.fmt_info['description']}]")

        # CSV shortcut
        if self.filepath.endswith(".csv"):
            self.raw_df = pd.read_csv(self.filepath, nrows=sample_limit)
            print(f"[+] Loaded {len(self.raw_df):,} records from CSV.")
            return self.raw_df

        parsed = []
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if sample_limit and i >= sample_limit:
                    break
                s = line.strip()
                if not s:
                    continue

                # JSON-array format
                if s.startswith("[") and s.endswith("]"):
                    try:
                        arr = json.loads(s)
                        if isinstance(arr, list) and len(arr) >= 4:
                            parsed.append({
                                "ip":            str(arr[3]) if arr[3] is not None else None,
                                "timestamp":     str(arr[2]) if arr[2] is not None else None,
                                "request_type":  None,
                                "url_path":      str(arr[8]) if len(arr) > 8 and arr[8] is not None else None,
                                "status_code":   None,
                                "response_bytes": None,
                                "user_agent":    str(arr[5]) if len(arr) > 5 and arr[5] is not None else None,
                            })
                            continue
                    except Exception:
                        pass

                # Apache Combined Log
                m = self.APACHE_LOG_PATTERN.match(s)
                if m:
                    g    = m.groupdict()
                    req  = g.get("request", "")
                    parts = req.split(" ")
                    parsed.append({
                        "ip":            g["ip"],
                        "timestamp":     g["timestamp"],
                        "request_type":  parts[0] if parts else None,
                        "url_path":      parts[1] if len(parts) > 1 else None,
                        "status_code":   int(g["status_code"]),
                        "response_bytes": None if g["response_bytes"] == "-"
                                          else int(g["response_bytes"]),
                        "user_agent":    g.get("user_agent"),
                    })

        self.raw_df = pd.DataFrame(parsed)
        print(f"[+] Parsed {len(self.raw_df):,} HTTP log entries.")
        return self.raw_df

    # ------------------------------------------------------------------
    def clean_data(self):
        print("\n[*] Stage 2: Data Cleaning & Preprocessing...")
        if self.raw_df is None or self.raw_df.empty:
            raise ValueError("Empty raw DataFrame — run ingest_and_parse() first.")

        df = self.raw_df.copy()

        # ML is only permitted for complete, source-backed HTTP fields.
        required = {"ip", "timestamp", "request_type", "url_path", "status_code", "user_agent"}
        missing = required - set(df.columns)
        if missing or any(df[column].isna().any() for column in required):
            raise ValueError("ML analysis unavailable: required HTTP fields are absent or incomplete; no defaults are invented.")

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"    Removed {before - len(df):,} duplicate records.")

        if "timestamp_dt" not in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", errors="coerce")
            if df["timestamp_dt"].isna().all():
                df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
        if df["timestamp_dt"].isna().any():
            raise ValueError("ML analysis unavailable: one or more timestamps cannot be parsed.")
        df = df.sort_values("timestamp_dt").reset_index(drop=True)

        valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
        df["request_type"] = df["request_type"].astype(str).str.upper()
        df["request_type"] = df["request_type"].apply(lambda x: x if x in valid_methods else "OTHER")

        if "user_agent" in df.columns:
            df["user_agent"] = df["user_agent"].astype(str)

        self.clean_df = df
        print(f"[+] Clean records: {len(self.clean_df):,}")
        return self.clean_df

    # ------------------------------------------------------------------
    def rule_based_labeling(self):
        print("\n[*] Stage 3: Rule-Based Attack Labeling...")
        if self.clean_df is None:
            raise ValueError("Run clean_data() first.")

        df = self.clean_df.copy()

        if "label" not in df.columns:
            sqli_kw  = ["union", "select", "insert", "drop", "concat", "char(", "--", "%27", "'", "1=1", "or 1=1"]
            path_kw  = ["../", "..\\", "%2e%2e", "/etc/passwd", "c:\\", "cmd.exe", "win.ini"]

            def _label(row):
                url  = str(row.get("url_path", "")).lower()
                ua   = str(row.get("user_agent", "")).lower()
                code = row.get("status_code", 200)
                if any(k in url for k in sqli_kw):
                    return "SQLI"
                if any(k in url for k in path_kw):
                    return "PATH_TRAVERSAL"
                if code == 403 or (code == 404 and "bot" in ua):
                    return "BRUTE_FORCE"
                return "BENIGN"

            df["label"] = df.apply(_label, axis=1)

        self.clean_df = df
        print("[+] Attack labels applied.")
        return self.clean_df

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(text):
        if not isinstance(text, str) or not text:
            return 0.0
        prob = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    def engineer_features(self):
        print("\n[*] Stage 4: Feature Engineering...")
        if self.clean_df is None:
            raise ValueError("Run rule_based_labeling() first.")

        df = self.clean_df.copy()

        if "ip_request_count" not in df.columns:
            df["ip_request_count"] = df["ip"].map(df["ip"].value_counts())
        if "time_diff_seconds" not in df.columns:
            df["time_diff_seconds"] = (
                df.groupby("ip")["timestamp_dt"].diff().dt.total_seconds().fillna(0)
            )
        if "status_code_404_count" not in df.columns:
            if "status_code" in df.columns:
                df["is_404"] = (df["status_code"] == 404).astype(int)
                df["status_code_404_count"] = df.groupby("ip")["is_404"].transform("sum")
                df.drop(columns=["is_404"], inplace=True)
            else:
                df["status_code_404_count"] = 0.0
        if "is_bot_agent" not in df.columns:
            bots = ["bot", "crawler", "spider", "python", "curl", "wget", "nikto", "sqlmap", "nmap", "hydra"]
            df["is_bot_agent"] = df["user_agent"].str.lower().apply(lambda u: 1 if any(b in u for b in bots) else 0)
        if "url_entropy" not in df.columns:
            df["url_entropy"] = df["url_path"].apply(self._entropy)

        self.engineered_df = df
        print(f"[+] Features engineered on {len(df):,} entries.")
        return self.engineered_df

    # ------------------------------------------------------------------
    def train_model(self):
        print("\n[*] Stage 5: ML Model Training (Random Forest)...")
        if self.engineered_df is None:
            raise ValueError("Run engineer_features() first.")

        feature_cols = ["ip_request_count", "time_diff_seconds",
                        "status_code_404_count", "is_bot_agent", "url_entropy"]

        train_path = resolve_input_file("train_dataset.csv")
        test_path  = resolve_input_file("test_dataset.csv")

        if os.path.exists(train_path) and os.path.exists(test_path):
            print("    Using pre-split datasets from Practical 6...")
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            valid_feat = [c for c in feature_cols if c in train_df.columns]
            X_train, y_train = train_df[valid_feat].fillna(0), train_df["label"]
            X_test,  y_test  = test_df[valid_feat].fillna(0),  test_df["label"]
        else:
            print("    No pre-split files found — performing inline 80/20 split...")
            df_m = self.engineered_df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
            valid_feat = [c for c in feature_cols if c in df_m.columns]
            X, y = df_m[valid_feat].fillna(0), df_m["label"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

        print(f"    Training: {len(X_train):,}  |  Testing: {len(X_test):,}")

        self.model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        y_pred       = self.model.predict(X_test)
        labels_order = sorted(y_test.unique())

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test,  y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test,     y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred, labels=labels_order)

        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_test, y_pred,
              target_names=[l.upper() for l in labels_order], digits=4))

        metrics_table = [
            ["Accuracy",           f"{acc  * 100:.4f}%"],
            ["Weighted Precision", f"{prec * 100:.4f}%"],
            ["Weighted Recall",    f"{rec  * 100:.4f}%"],
            ["Weighted F1-Score",  f"{f1   * 100:.4f}%"],
        ]
        print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="fancy_grid"))

        imp_df = pd.DataFrame({"Feature": list(valid_feat),
                               "Importance": self.model.feature_importances_})\
                   .sort_values("Importance", ascending=False)
        print("\n--- FEATURE IMPORTANCE ---")
        print(tabulate([[r["Feature"], f"{r['Importance']*100:.2f}%"]
                        for _, r in imp_df.iterrows()],
                       headers=["Feature", "Gini (%)"], tablefmt="fancy_grid"))

        X_infer = self.engineered_df[valid_feat].fillna(0)
        self.engineered_df["ml_predicted_label"] = self.model.predict(X_infer)

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm.tolist()}

    # ------------------------------------------------------------------
    def save_pipeline_output(self, csv_filepath="processed_log_pipeline_output.csv",
                             parquet_filepath="processed_log_pipeline_output.parquet"):
        print("\n[*] Stage 6: Exporting Pipeline Artifacts...")
        if self.engineered_df is None:
            raise ValueError("Run engineer_features() first.")

        csv_out     = resolve_export_file(csv_filepath)
        parquet_out = resolve_export_file(parquet_filepath)
        export_cols = [c for c in [
            "ip", "timestamp", "request_type", "url_path", "status_code", "user_agent",
            "ip_request_count", "time_diff_seconds", "status_code_404_count",
            "is_bot_agent", "url_entropy", "label",
        ] if c in self.engineered_df.columns]

        out_df = self.engineered_df[export_cols]
        out_df.to_csv(csv_out, index=False)
        print(f"    CSV saved: '{csv_out}' ({os.path.getsize(csv_out)/1e6:.2f} MB)")
        try:
            out_df.to_parquet(parquet_out, index=False)
            print(f"    Parquet saved: '{parquet_out}' ({os.path.getsize(parquet_out)/1e6:.2f} MB)")
        except Exception as exc:
            print(f"    [!] Parquet skipped ({exc}). Install pyarrow/fastparquet.")
        return csv_out, parquet_out

    # ------------------------------------------------------------------
    def run(self, output_csv="processed_log_pipeline_output.csv",
            output_parquet="processed_log_pipeline_output.parquet"):
        print(_header("HTTP LOG PIPELINE — Full Run"))
        self.ingest_and_parse()
        self.clean_data()
        self.rule_based_labeling()
        self.engineer_features()
        metrics = self.train_model()
        self.save_pipeline_output(csv_filepath=output_csv, parquet_filepath=output_parquet)
        print(f"\n[✔] HTTP Pipeline complete. Accuracy={metrics['accuracy']*100:.2f}% F1={metrics['f1']*100:.2f}%")
        return metrics


# ============================================================================
# Oracle Data Loader / Application log pipeline (stages 2–7 only)
# ============================================================================

class ApplicationLogPipeline:
    """
    Handles Oracle Data Loader, Java application, and generic logs.
    Does NOT produce ML-trainable DataFrames (no labellable HTTP requests).
    """

    def __init__(self, filepath: str, fmt_info: dict):
        self.filepath = filepath
        self.fmt_info = fmt_info
        self.parse_result = None

    def run(self):
        print(_header(f"APPLICATION LOG PIPELINE: {self.fmt_info['description']}"))

        # Read lines
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as fh:
            lines = [ln.rstrip("\n") for ln in fh]

        # Parse via format-specific parser
        print(f"\n[*] Stage 1: Parsing with {self.fmt_info['format']} parser...")
        self.parse_result = parse_log(lines, self.fmt_info)
        total_events = len(self.parse_result["events"])
        print(f"[+] Parsed {total_events:,} log events.")

        # Display summary
        self._display_summary()
        self._display_errors()
        self._display_timeline()
        self._display_security_analysis()

        print(f"\n[✔] Application log pipeline complete.")

    # ------------------------------------------------------------------
    def _display_summary(self):
        print(_header("ANALYSIS SUMMARY"))
        sm = self.parse_result["summary"]
        af = self.parse_result["applicable_fields"]
        fmt = self.fmt_info

        print(f"\n  Log Format            : {fmt['description']}")
        print(f"  Confidence            : {fmt['confidence'] * 100:.0f}%")
        print(f"\n  Total Log Events      : {sm.get('total_log_events', 0):,}")

        # Oracle-specific fields
        if fmt["format"] == "oracle_data_loader":
            fields = [
                ("Import Status",        af.get("import_status",  {}), sm.get("import_status")),
                ("Operation",            af.get("operation",      {}), sm.get("operation")),
                ("Record Type",          af.get("record_type",    {}), sm.get("record_type")),
                ("Threads",              af.get("threads",        {}), sm.get("thread_count")),
                ("CSV Segments",         af.get("csv_segments",   {}), sm.get("csv_segments")),
                ("Validation",           af.get("validation",     {}), sm.get("validation_status")),
                ("Submission Progress",  af.get("submission",     {}), sm.get("submission_progress")),
                ("SOAP Operations",      {"status": "available"},      sm.get("soap_operations")),
                ("Rate-Limit Errors",    {"status": "available"},      sm.get("rate_limit_errors")),
                ("Retries",              {"status": "available"},      sm.get("retries")),
                ("Warnings",             {"status": "available"},      sm.get("warnings")),
            ]
        else:
            lv = sm.get("level_distribution", {})
            fields = [
                ("ERROR Count",   {"status": "available"}, sm.get("error_count")),
                ("WARNING Count", {"status": "available"}, sm.get("warning_count")),
                ("Log Levels",    {"status": "available"}, ", ".join(f"{k}:{v}" for k, v in lv.items())),
            ]

        for name, fi, val in fields:
            print(_status_block(name, fi, val))

        # Explicitly NOT applicable fields
        not_applicable = [
            ("Total HTTP Requests",  af.get("total_requests", {})),
            ("Unique IP Addresses",  af.get("unique_ips",     {})),
            ("Data Transfer",        af.get("data_transfer",  {})),
            ("User-Agents",          af.get("user_agents",    {})),
            ("HTTP Status Codes",    af.get("http_status_codes", {})),
            ("Targeted Endpoints",   af.get("targeted_endpoints", {})),
        ]
        shown_any_na = False
        for name, fi in not_applicable:
            if fi.get("status") == "not_applicable":
                if not shown_any_na:
                    print("\n  --- Fields NOT applicable to this log format ---")
                    shown_any_na = True
                print(_status_block(name, fi))

    # ------------------------------------------------------------------
    def _display_errors(self):
        errors = self.parse_result.get("errors", [])
        if not errors:
            return
        print(_header(f"ERRORS & WARNINGS  ({len(errors)} found)"))
        for err in errors:
            code     = err.get("code", "—")
            sev      = (err.get("severity") or "error").upper()
            ts       = err.get("timestamp", "?")
            desc     = err.get("description") or err.get("message", "")[:120]
            print(f"\n  [{sev}] {code}  @{ts}")
            print(f"    {desc}")

    # ------------------------------------------------------------------
    def _display_timeline(self):
        timeline = self.parse_result.get("timeline", [])
        if not timeline:
            return
        print(_header(f"EXECUTION TIMELINE  ({len(timeline)} events)"))
        for item in timeline:
            ts  = item.get("timestamp", "?") or "?"
            evt = item.get("event", "")
            print(f"  {ts:>12}   {evt}")

    # ------------------------------------------------------------------
    def _display_security_analysis(self):
        print(_header("SECURITY ANALYSIS"))
        af  = self.parse_result["applicable_fields"]
        ti  = af.get("threat_intelligence", {})
        status = ti.get("status", "not_applicable")
        if status == "not_applicable":
            print(f"\n  Status  : NOT APPLICABLE")
            print(f"  Reason  : {ti.get('reason', 'No network security indicators present.')}")
            print("\n  [!] No IP-blocking or threat-intelligence recommendations are made.")
            print("      There are no client IP addresses in this log file.")
        elif status == "available":
            print("\n  Security analysis complete — see HTTP pipeline output.")
        else:
            print(f"\n  Status: {status.upper()}")


# ============================================================================
# Main router — LogDataPipeline
# ============================================================================

class LogDataPipeline:
    """
    Entry point for the reusable data pipeline.

    Automatically detects the log format and routes to:
      - HttpLogPipeline     (Apache / Nginx / JSON-array)
      - ApplicationLogPipeline (Oracle Data Loader / Java / Generic)
    """

    def __init__(self, log_filepath="data.log"):
        self.log_filepath = resolve_input_file(log_filepath)

    def run_full_pipeline(self, output_csv="processed_log_pipeline_output.csv",
                          output_parquet="processed_log_pipeline_output.parquet"):
        print("=" * 60)
        print("PRACTICAL 10: Reusable Automated Log Processing Pipeline")
        print("=" * 60)

        # Resolve file
        if not os.path.exists(self.log_filepath):
            for fallback in ["Logs/logs/cj.log", "structured_logs.csv", "cleaned_logs.csv"]:
                cand = resolve_input_file(fallback)
                if os.path.exists(cand):
                    self.log_filepath = cand
                    break

        if not os.path.exists(self.log_filepath):
            raise FileNotFoundError(f"Log file not found: {self.log_filepath}")

        # CSV shortcut — always HTTP pipeline
        if self.log_filepath.endswith(".csv"):
            fmt_info = {"format": "json_array", "confidence": 1.0,
                        "description": "Structured CSV (pre-parsed log data)", "parser": "json_array"}
            pipe = HttpLogPipeline(self.log_filepath, fmt_info)
            return pipe.run(output_csv=output_csv, output_parquet=output_parquet)

        # ── FORMAT DETECTION ────────────────────────────────────────────
        print(f"\n[*] Detecting log format for: '{os.path.basename(self.log_filepath)}'")
        fmt_info = detect_log_format_from_file(self.log_filepath)
        print(f"[+] Format   : {fmt_info['description']}")
        print(f"[+] Format ID: {fmt_info['format']}")
        print(f"[+] Confidence: {fmt_info['confidence'] * 100:.0f}%")

        # ── FORMAT ROUTING ───────────────────────────────────────────────
        http_formats = {"apache_access", "nginx_access", "json_array"}
        app_formats  = {"oracle_data_loader", "java_application", "generic"}

        if fmt_info["format"] in http_formats:
            pipe = HttpLogPipeline(self.log_filepath, fmt_info)
            return pipe.run(output_csv=output_csv, output_parquet=output_parquet)
        else:
            pipe = ApplicationLogPipeline(self.log_filepath, fmt_info)
            pipe.run()
            return {"format": fmt_info["format"]}


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Practical 10: Reusable Automated Log Processing Pipeline"
    )
    parser.add_argument("--input",          default="data.log",
                        help="Path to input log file (default: data.log)")
    parser.add_argument("--output_csv",     default="processed_log_pipeline_output.csv",
                        help="Output CSV path (HTTP logs only)")
    parser.add_argument("--output_parquet", default="processed_log_pipeline_output.parquet",
                        help="Output Parquet path (HTTP logs only)")
    args = parser.parse_args()

    pipeline = LogDataPipeline(log_filepath=args.input)
    pipeline.run_full_pipeline(output_csv=args.output_csv,
                               output_parquet=args.output_parquet)


if __name__ == "__main__":
    main()

"""Reusable HTTP-only ML and ATTP inference for LogSentinel.

Training is intentionally offline. Upload requests only load the saved model and
infer from fields that were actually parsed from the uploaded HTTP log.
"""

from collections import Counter
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from practical_unique_algorithm import AdaptiveTemporalThreatProfiler

BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "http_random_forest.joblib"
FEATURES = [
    "ip_request_count", "url_entropy", "is_bot_agent", "time_diff_seconds",
    "status_code_404_count", "flow_duration", "fwd_bytes", "bwd_bytes",
    "flow_pkts_per_sec", "avg_packet_size"
]
BOT_TERMS = ("bot", "crawler", "spider", "python", "curl", "wget", "nikto", "sqlmap", "nmap", "hydra")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    return -sum((text.count(char) / len(text)) * math.log2(text.count(char) / len(text)) for char in set(text))


def train_and_save_model() -> Dict[str, Any]:
    """Train the Practical 9-style Random Forest once and save a model artifact."""
    train = pd.read_csv(BASE_DIR / "train_dataset.csv")
    test = pd.read_csv(BASE_DIR / "test_dataset.csv")
    required = FEATURES + ["label"]
    if any(column not in train.columns for column in required) or any(column not in test.columns for column in required):
        raise ValueError("Training/test datasets do not contain the required feature schema.")
    train = train.dropna(subset=required)
    test = test.dropna(subset=required)
    model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["label"])
    predicted = model.predict(test[FEATURES])
    evaluation = {
        "accuracy": round(float(accuracy_score(test["label"], predicted)), 4),
        "precision_weighted": round(float(precision_score(test["label"], predicted, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(test["label"], predicted, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(test["label"], predicted, average="weighted", zero_division=0)), 4),
    }
    artifact = {
        "model": model, "features": FEATURES, "classes": list(model.classes_), "evaluation": evaluation,
        "training_rows": int(len(train)), "test_rows": int(len(test)),
        "training_label_counts": {str(key): int(value) for key, value in train["label"].value_counts().items()},
        "source": "Practical 5 features; Practical 6 split; Practical 9/10 Random Forest configuration",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    _load_artifact.cache_clear()
    return {key: value for key, value in artifact.items() if key != "model"}


@lru_cache(maxsize=1)
def _load_artifact() -> Dict[str, Any] | None:
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def feature_frame_from_events(events: List[Dict[str, Any]]) -> Tuple[pd.DataFrame | None, str | None]:
    """Return features extracted from events that have at least ip."""
    valid_events = [e for e in events if e.get("ip")]
    if not valid_events:
        return None, "No events found with an 'ip' field."
    
    rows = []
    for index, event in enumerate(valid_events):
        rows.append({
            "event_index": index,
            "ip": str(event["ip"]),
            "timestamp": event.get("timestamp") or "1970-01-01 00:00:00",
            "url": str(event.get("url", "")),
            "status_code": event.get("status_code", 0),
            "user_agent": str(event.get("user_agent", "")),
            "flow_duration": float(event.get("flow_duration", 0.0)),
            "fwd_bytes": float(event.get("fwd_bytes", 0.0)),
            "bwd_bytes": float(event.get("bwd_bytes", 0.0)),
            "flow_pkts_per_sec": float(event.get("flow_pkts_per_sec", 0.0)),
            "avg_packet_size": float(event.get("avg_packet_size", 0.0))
        })
    
    frame = pd.DataFrame(rows)
    # Timestamps parsing
    frame["timestamp_dt"] = pd.to_datetime(
        frame["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", errors="coerce", utc=True
    )
    unresolved = frame["timestamp_dt"].isna()
    if unresolved.any():
        frame.loc[unresolved, "timestamp_dt"] = pd.to_datetime(
            frame.loc[unresolved, "timestamp"], format="mixed", errors="coerce", utc=True
        )
    if frame["timestamp_dt"].isna().any():
        return None, "ML requires parseable timestamps; one or more timestamps could not be parsed."
        
    frame = frame.sort_values(["ip", "timestamp_dt"]).reset_index(drop=True)
    frame["ip_request_count"] = frame["ip"].map(frame["ip"].value_counts())
    frame["time_diff_seconds"] = frame.groupby("ip")["timestamp_dt"].diff().dt.total_seconds().fillna(0.0)
    
    frame["status_code"] = pd.to_numeric(frame["status_code"], errors="coerce").fillna(0)
    frame["status_code_404_count"] = frame.groupby("ip")["status_code"].transform(lambda values: (values == 404).sum())
    
    frame["is_bot_agent"] = frame["user_agent"].str.lower().apply(lambda value: int(any(term in value for term in BOT_TERMS)))
    frame["url_entropy"] = frame["url"].apply(_entropy)
    
    return frame, None


def analyze_http_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run saved Random Forest and ATTP against a compatible upload."""
    artifact = _load_artifact()
    if artifact is None:
        return {"status": "unavailable", "reason": "No trained model artifact exists. Run: python ml_engine.py --train"}
    frame, reason = feature_frame_from_events(events)
    if frame is None:
        return {"status": "not_applicable", "reason": reason, "model": {"status": "not_applicable"},
                "attp": {"status": "not_applicable"}}

    model = artifact["model"]
    model_features = artifact["features"]
    
    # Ensure all required model features are in the frame, filling missing with 0.0
    for col in model_features:
        if col not in frame.columns:
            frame[col] = 0.0

    probabilities = model.predict_proba(frame[model_features])
    predictions = model.predict(frame[model_features])
    confidence = probabilities.max(axis=1)
    model_rows = [{"event_index": int(row.event_index), "label": str(label), "confidence": round(float(conf), 4)}
                  for row, label, conf in zip(frame.itertuples(), predictions, confidence)]
    model_summary = {
        "status": "available", "algorithm": "Random Forest", "predictions": model_rows[:100],
        "label_distribution": {str(key): int(value) for key, value in Counter(predictions).items()},
        "mean_confidence": round(float(confidence.mean()), 4), "training_evaluation": artifact["evaluation"],
        "training_rows": artifact["training_rows"], "source": artifact["source"],
        "warning": "Predictions are limited to the training data's web-request labels and are not threat-intelligence verdicts.",
    }
    if len(frame) < 3:
        attp_summary = {"status": "unavailable", "reason": "ATTP needs at least three complete HTTP events."}
    else:
        profiler = AdaptiveTemporalThreatProfiler(decay_lambda=0.001, anomaly_threshold=2.0, blend_alpha=0.70)
        with redirect_stdout(StringIO()):
            labels = profiler.fit_predict(frame)
        scores = profiler.score_series()
        attp_summary = {
            "status": "available", "algorithm": "Adaptive Temporal Threat Profiler (ATTP)",
            "label_distribution": {str(key): int(value) for key, value in Counter(labels).items()},
            "mean_score": round(float(scores.mean()), 4), "threshold": 2.0,
            "predictions": [{"event_index": int(row.event_index), "label": str(label), "score": float(score)}
                            for row, label, score in zip(frame.itertuples(), labels, scores)][:100],
            "warning": "ATTP is an anomaly score, not a malicious-IP or threat-intelligence verdict.",
        }
    return {"status": "available", "reason": "Extracted features from parsed IP and timestamp events.",
            "feature_schema": model_features, "records_analyzed": int(len(frame)), "model": model_summary, "attp": attp_summary}


def model_metadata() -> Dict[str, Any]:
    """Small, UI-safe description of the saved model; never exposes the model object."""
    artifact = _load_artifact()
    if artifact is None:
        return {"status": "unavailable", "features": [], "importances": {}, "ml_labels": {}}
    importances = {
        feature: round(float(value), 6)
        for feature, value in zip(artifact["features"], artifact["model"].feature_importances_)
    }
    return {
        "status": "available", "algorithm": "Random Forest", "features": artifact["features"],
        "importances": importances,
        "ml_labels": {str(label): "Training label" for label in artifact["classes"]},
        "evaluation": artifact["evaluation"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save the reusable HTTP model")
    args = parser.parse_args()
    if args.train:
        result = train_and_save_model()
        print(f"Saved model to {MODEL_PATH}")
        print(result["evaluation"])
    else:
        parser.print_help()

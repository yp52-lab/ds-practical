"""LogSentinel API: deterministic, format-aware log analysis."""

from collections import Counter, defaultdict
from pathlib import Path
import ipaddress
import re
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from log_format_detector import detect_log_format
from log_parsers import parse_log
from ml_engine import analyze_http_events, model_metadata

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Report helpers (produce the rich structured sections shown in the UI)
# ─────────────────────────────────────────────────────────────────────────────

_AUTH_RE = re.compile(
    r'\b(login|logon|logout|authentication|auth|sign.?in|sign.?out|'
    r'session|connect|disconnect|credential|password|passwd|'
    r'failed.?login|login.?failed|invalid.?pass|access.?denied|'
    r'permission.?denied|unauthorized|unauthenticated)\b', re.I
)
_AUTH_FAIL_RE = re.compile(
    r'\b(fail|failed|invalid|denied|unauthorized|error|wrong|bad|'
    r'incorrect|reject|rejected)\b', re.I
)
_AUTH_OK_RE   = re.compile(r'\b(success|successful|accepted|granted|ok|logged.?in|connected)\b', re.I)
_DISC_RE      = re.compile(r'\b(disconnect|logout|sign.?out|session.?end|session.?close|timeout)\b', re.I)

_DB_RE = re.compile(
    r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|REPLACE|MERGE|'
    r'EXEC|EXECUTE|CALL|COMMIT|ROLLBACK|TRANSACTION|QUERY|SQL)\b', re.I
)
_DB_OP_RE = re.compile(
    r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|REPLACE|MERGE|'
    r'EXEC|EXECUTE|CALL|COMMIT|ROLLBACK)\b', re.I
)

_IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')

_LEVEL_RE = re.compile(r'\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ALERT|CRITICAL|FATAL|SEVERE|EMERGENCY)\b', re.I)

# Normalise a raw line into a simple word-pattern for frequency analysis
_TOKENISE_RE = re.compile(r'[^\w\s\[\]|/-]')
_NUM_RE       = re.compile(r'\b\d[\d.,:]*\b')
_IPADDR_RE    = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')


def _normalise_line(line: str, max_len: int = 120) -> str:
    """Replace variable parts so structurally identical lines cluster."""
    line = line[:max_len]
    line = _IPADDR_RE.sub('N.N.N.N', line)
    # Dates / times
    line = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?', 'DATETIME', line)
    line = re.sub(r'\d{2}/\w+/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}', 'DATETIME', line)
    line = re.sub(r'\d{2}:\d{2}:\d{2}(?:[.,]\d+)?', 'TIME', line)
    # Large numbers (IDs, bytes, counts)
    line = _NUM_RE.sub('N', line)
    # Collapse whitespace
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def _dynamic_report(events: List[Dict], lines: List[str], fmt: Dict) -> Dict[str, Any]:
    """Build the rich structured sections shown in the Dynamic Analysis Report UI."""

    # ── 1. Application Log Analysis ────────────────────────────────────────
    level_counts: Counter = Counter()
    for ev in events:
        lvl = (ev.get('level') or '').upper()
        if lvl:
            level_counts[lvl] += 1
    # Also scan raw lines for levels missed by parsers
    for ln in lines:
        m = _LEVEL_RE.search(ln)
        if m:
            lvl = m.group(1).upper()
            if lvl not in level_counts:
                level_counts[lvl] += 1

    total_events = len(events) or len(lines)
    error_count  = sum(level_counts.get(k, 0) for k in ('ERROR', 'CRITICAL', 'FATAL', 'SEVERE'))
    warn_count   = sum(level_counts.get(k, 0) for k in ('WARN', 'WARNING', 'ALERT'))
    debug_count  = sum(level_counts.get(k, 0) for k in ('DEBUG', 'TRACE'))
    info_count   = level_counts.get('INFO', 0)

    app_log_analysis = {
        "total_log_events": total_events,
        "error_events":  error_count,
        "warn_events":   warn_count,
        "debug_events":  debug_count,
        "info_events":   info_count,
        "level_distribution": [
            {"value": k, "count": v}
            for k, v in sorted(level_counts.items(), key=lambda x: -x[1])
        ],
    }

    # ── 2. Authentication Analysis ─────────────────────────────────────────
    auth_lines      = []
    auth_fail_lines = []
    auth_ok_lines   = []
    disc_lines      = []
    auth_ips: Counter = Counter()

    for ln in lines:
        if _AUTH_RE.search(ln):
            auth_lines.append(ln)
            ip_m = _IP_RE.search(ln)
            if ip_m:
                auth_ips[ip_m.group(1)] += 1
            if _AUTH_FAIL_RE.search(ln):
                auth_fail_lines.append(ln)
            elif _AUTH_OK_RE.search(ln):
                auth_ok_lines.append(ln)
            if _DISC_RE.search(ln):
                disc_lines.append(ln)

    total_auth      = len(auth_lines)
    failed_auth     = len(auth_fail_lines)
    successful_auth = len(auth_ok_lines)
    disc_events     = len(disc_lines)
    failure_rate    = round(failed_auth / total_auth * 100, 1) if total_auth else 0.0

    top_auth_ips = [
        {"value": ip, "count": cnt}
        for ip, cnt in auth_ips.most_common(10)
    ]

    auth_analysis = {
        "total_auth_events":         total_auth,
        "successful_auth":           successful_auth,
        "failed_auth":               failed_auth,
        "logout_disconnect_events":  disc_events,
        "auth_failure_rate":         f"{failure_rate}%",
        "unique_auth_source_ips":    len(auth_ips),
        "top_auth_source_ips":       top_auth_ips,
    }

    # ── 3. Database Log Analysis ───────────────────────────────────────────
    db_lines = []
    db_ops: Counter = Counter()
    for ln in lines:
        if _DB_RE.search(ln):
            db_lines.append(ln)
            op_m = _DB_OP_RE.search(ln)
            if op_m:
                db_ops[op_m.group(1).upper()] += 1

    db_op_distribution = [
        {"operation": op, "count": cnt}
        for op, cnt in sorted(db_ops.items(), key=lambda x: -x[1])
    ]

    db_analysis = {
        "total_db_events":   len(db_lines),
        "op_distribution":   db_op_distribution,
    }

    # ── 4. Threat / Anomaly Detection ────────────────────────────────────
    _THREAT_RE = re.compile(
        r'\b(attack|exploit|injection|traversal|brute.force|dos|ddos|'
        r'xss|csrf|overflow|malware|ransomware|phishing|suspicious|anomal|'
        r'intrusion|compromise|unauthorized.access|privilege.escalation)\b', re.I
    )
    threat_lines = [ln for ln in lines if _THREAT_RE.search(ln)]
    threats_detected = len(threat_lines) > 0

    # ── 5. Analysis Coverage Notes ─────────────────────────────────────────
    coverage_notes = []
    is_http = fmt.get('format') in ('apache_access', 'nginx_access', 'json_array')
    ts_count = sum(1 for ev in events if ev.get('timestamp'))
    if not is_http:
        coverage_notes.append({
            "type": "info",
            "title": "HTTP Security Analysis",
            "message": (
                "Insufficient data for HTTP request fields (method, status code, client IP) "
                "were detected in this log file. HTTP-specific threat analysis is not applicable."
            )
        })
    if ts_count < 10:
        coverage_notes.append({
            "type": "info",
            "title": "Insufficient Data for Time-Based Anomaly Detection",
            "message": (
                f"Only {ts_count} events have timestamps. "
                "Time-based anomaly detection requires at least 10 timestamped events."
            )
        })

    # ── 6. Frequent Log Messages ───────────────────────────────────────────
    pattern_counts: Counter = Counter()
    for ln in lines[:50_000]:   # cap to protect memory
        pattern_counts[_normalise_line(ln)] += 1
    frequent_messages = [
        {"message_pattern": pat, "count": cnt}
        for pat, cnt in pattern_counts.most_common(10)
        if cnt > 1
    ]

    # ── 7. Detected Fields ─────────────────────────────────────────────────
    field_samples: Dict[str, str] = {}
    for ev in events[:500]:
        for k, v in ev.items():
            if v and k not in ('metadata', 'event_type') and k not in field_samples:
                field_samples[k] = str(v)[:60]
    # Also scan raw lines for key=value patterns
    _KV_RE = re.compile(r'\b([\w.-]{2,20})\s*[=:]\s*([^\s,;|"\']{1,50})')
    for ln in lines[:200]:
        for m in _KV_RE.finditer(ln):
            k, v = m.group(1), m.group(2)
            if k.lower() not in ('http', 'for', 'in', 'at', 'on', 'the') and k not in field_samples:
                field_samples[k] = v[:60]

    detected_fields = [{"key": k, "sample": v} for k, v in list(field_samples.items())[:15]]

    return {
        "app_log_analysis":  app_log_analysis,
        "auth_analysis":     auth_analysis,
        "db_analysis":       db_analysis,
        "threats_detected":  threats_detected,
        "threat_count":      len(threat_lines),
        "coverage_notes":    coverage_notes,
        "frequent_messages": frequent_messages,
        "detected_fields":   detected_fields,
    }

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "ui_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
LEGACY_UI_DIR = BASE_DIR / "ui"
# The React frontend is the primary UI. The second candidate keeps this
# workspace compatible with its existing nested frontend location.
FRONTEND_DIRS = [
    BASE_DIR / "logsentinel-ui",
    BASE_DIR / "home" / "yug" / "My_Practice" / "ds lab" / "logsentinel-ui",
]
REACT_DIST = next((directory / "dist" for directory in FRONTEND_DIRS
                   if (directory / "dist" / "index.html").exists()), None)

app = FastAPI(title="LogSentinel", version="2.0.0",
              description="Evidence-based, format-aware log analysis.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
if REACT_DIST and (REACT_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_DIST / "assets")), name="react-assets")
if LEGACY_UI_DIR.exists():
    app.mount("/legacy", StaticFiles(directory=str(LEGACY_UI_DIR), html=True), name="legacy-ui")

METRIC_NAMES = {
    "total_log_events": "Total Log Events", "total_requests": "Total Requests",
    "unique_ips": "Unique IPs", "data_transfer": "Total Data Transfer",
    "user_agents": "User Agents", "http_methods": "HTTP Methods",
    "status_codes": "HTTP Status Codes", "endpoints": "Endpoints",
    "error_rate": "HTTP Error Rate", "request_rate": "Request Rate",
    "log_levels": "Log Levels", "components": "Components",
    "execution_duration": "Execution Duration", "parsing": "Parsing",
    "threat_intelligence": "Threat Intelligence",
}
HTTP_FORMATS = {"apache_access", "nginx_access", "json_array"}
SUSPICIOUS_PATHS = ("../", "..\\", "%2e%2e", "/etc/passwd", "cmd.exe", "win.ini")
SQLI = ("union", "select", "insert", "drop", "%27", "'", "--", "1=1", "or 1=1")
UPLOAD_CHUNK_BYTES = 1024 * 1024
DETECTION_SAMPLE_BYTES = 64 * 1024
MAX_ANALYSIS_LINES = 25_000
MAX_RESPONSE_EVENTS = 200
MAX_RESPONSE_ITEMS = 100


@app.get("/", include_in_schema=False)
def root():
    if REACT_DIST:
        return FileResponse(REACT_DIST / "index.html")
    return {
        "status": "frontend_not_built",
        "message": "Run python server.py to build and start the combined application.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_version": "2.0"}


def _safe_name(filename: str | None) -> str:
    return Path(filename or "upload.log").name or "upload.log"


def _metric_list(fields: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = []
    for key, info in fields.items():
        status = info.get("status", "unavailable")
        if status == "not_available":  # compatibility with old parser results
            status = "unavailable"
        metrics.append({
            "id": key, "name": METRIC_NAMES.get(key, key.replace("_", " ").title()),
            "value": info.get("value"), "status": status, "reason": info.get("reason"),
        })
    return metrics


def _is_valid_ip(value: Any) -> bool:
    try:
        ipaddress.ip_address(str(value))
        return True
    except ValueError:
        return False


def _security_analysis(events: List[Dict[str, Any]], is_http: bool) -> Dict[str, Any]:
    """Rule-based findings. A finding always includes the parsed evidence used."""
    valid_events = [event for event in events if _is_valid_ip(event.get("ip"))]
    if not valid_events:
        return {"status": "not_applicable", "reason": "No valid client IP addresses were parsed.",
                "findings": [], "threat_intelligence": {"status": "not_applicable"}}

    findings, auth_failures = [], Counter()
    suspicious_by_ip = Counter()
    event_counts = Counter()
    
    generic_threats = ["failed", "denied", "unauthorized", "reject", "drop", "suspicious", "attack", "exploit", "timeout"]

    for event in valid_events:
        ip = event["ip"]
        event_counts[ip] += 1
        url = (event.get("url") or "").lower()
        status = event.get("status_code")
        message = (event.get("message") or "").lower()

        if any(token in url for token in SUSPICIOUS_PATHS) and len(findings) < MAX_RESPONSE_ITEMS:
            findings.append({"type": "suspicious_path", "ip": ip, "evidence": {"url": event.get("url")},
                             "confidence": 0.9, "message": "Parsed URL contains a path-traversal indicator."})
            suspicious_by_ip[ip] += 1
        if any(token in url for token in SQLI) and len(findings) < MAX_RESPONSE_ITEMS:
            findings.append({"type": "suspicious_query", "ip": ip, "evidence": {"url": event.get("url")},
                             "confidence": 0.8, "message": "Parsed URL contains a SQL-injection-like token."})
            suspicious_by_ip[ip] += 1
        if status in (401, 403):
            auth_failures[ip] += 1
            
        # Generic message check
        if any(t in message for t in generic_threats) and len(findings) < MAX_RESPONSE_ITEMS:
            # Add to findings occasionally to prevent flooding
            if suspicious_by_ip[ip] < 3:
                findings.append({"type": "suspicious_log_message", "ip": ip, "evidence": {"message": event.get("message")},
                                 "confidence": 0.65, "message": "Log message contains a suspicious security keyword."})
            suspicious_by_ip[ip] += 1

    for ip, count in auth_failures.items():
        if count >= 5 and len(findings) < MAX_RESPONSE_ITEMS:
            findings.append({"type": "repeated_authentication_failure", "ip": ip,
                             "evidence": {"http_statuses": [401, 403], "count": count}, "confidence": 0.75,
                             "message": "Five or more parsed authentication/authorization failures were observed."})
            suspicious_by_ip[ip] += count
            
    for ip, count in event_counts.items():
        if count >= 50 and len(findings) < MAX_RESPONSE_ITEMS:
            findings.append({"type": "high_volume_activity", "ip": ip,
                             "evidence": {"event_count": count}, "confidence": 0.85,
                             "message": "IP generated an unusually high volume of events (Possible DoS/Scanner)."})
            suspicious_by_ip[ip] += count

    recommendations = [
        {"action": "investigate", "ip": ip, "reason": "Parsed suspicious behavior is listed in the findings.",
         "confidence": round(max(f["confidence"] for f in findings if f["ip"] == ip), 2)}
        for ip in sorted(suspicious_by_ip)
    ]
    return {
        "status": "available", "reason": "Rule-based analysis evaluated extracted IPs, URLs, and generic messages.",
        "findings": findings,
        "threat_intelligence": {"status": "unavailable", "reason": "No threat-intelligence provider is configured."},
        "recommendations": recommendations,
    }


def _empty_analysis(filename: str) -> Dict[str, Any]:
    return {
        "file": filename, "log_format": {"name": "Empty log", "id": "empty", "confidence": 1.0},
        "summary": {"status": "unavailable", "reason": "The uploaded log file is empty."},
        "metrics": [], "parsing": {"total_lines": 0, "analyzed_lines": 0, "parsed_lines": 0, "failed_lines": 0},
        "events": [], "timeline": [], "errors": [], "warnings": [],
        "security_analysis": {"status": "not_applicable", "reason": "The log is empty.", "findings": []},
        "recommendations": [],
        "ml_analysis": {"status": "not_applicable", "reason": "The log is empty.",
                        "model": {"status": "not_applicable"}, "attp": {"status": "not_applicable"}},
        "filename": filename, "format": "empty", "description": "Empty log file", "confidence": 1.0,
        "is_http": False, "field_badges": [], "top_ips": [], "top_user_agents": [], "label_distribution": {},
    }


def _analysis_from_lines(filename: str, lines: List[str], fmt: Dict[str, Any], total_lines: int) -> Dict[str, Any]:
    if fmt["format"] == "empty":
        return _empty_analysis(filename)

    parsed = parse_log(lines, fmt)
    events, summary, fields = parsed["events"], parsed["summary"], parsed["applicable_fields"]
    failed = int(summary.get("parse_errors", 0))
    is_http = fmt["format"] in HTTP_FORMATS
    security = _security_analysis(events, is_http)
    ml_analysis = analyze_http_events(events)
    errors = [item for item in parsed.get("errors", []) if item.get("severity") != "warning"]
    warnings = [item for item in parsed.get("errors", []) if item.get("severity") == "warning"]
    metrics = _metric_list(fields)
    top_ips = [{"ip": ip, "count": count} for ip, count in summary.get("top_ips", [])]
    top_uas = [{"ua": ua, "count": count} for ua, count in summary.get("top_user_agents", [])]
    # Keep a legacy chart populated only by actual rule findings, never predictions.
    labels = Counter(finding["type"] for finding in security.get("findings", []))
    label_distribution = {name: {"count": count, "color": "#d4d4d4", "description": "Evidence-based rule finding"}
                          for name, count in labels.items()}
    dynamic_report = _dynamic_report(events, lines, fmt)
    result = {
        "file": filename,
        "log_format": {"name": fmt["description"], "id": fmt["format"], "confidence": fmt["confidence"],
                       "components": fmt.get("components", [])},
        "summary": {"status": "available", "total_events": len(events), **summary},
        "metrics": metrics,
        "analysis_scope": {
            "status": "sampled" if total_lines > len(lines) else "full",
            "total_nonempty_lines": total_lines,
            "analyzed_lines": len(lines),
            "reason": (f"Only the first {len(lines):,} non-empty lines were analysed to protect memory."
                       if total_lines > len(lines) else "The complete file was analysed."),
        },
        "parsing": {"total_lines": total_lines, "analyzed_lines": len(lines),
                    "parsed_lines": max(0, len(events) - failed), "failed_lines": failed},
        "events": events[:MAX_RESPONSE_EVENTS],
        "timeline": parsed.get("timeline", [])[:MAX_RESPONSE_ITEMS],
        "errors": errors[:MAX_RESPONSE_ITEMS], "warnings": warnings[:MAX_RESPONSE_ITEMS],
        "security_analysis": security, "recommendations": security.get("recommendations", []),
        "ml_analysis": ml_analysis,
        "dynamic_report": dynamic_report,
        # Backward-compatible presentation fields. They remain generated from the data above.
        "filename": filename, "format": fmt["format"], "description": fmt["description"],
        "confidence": fmt["confidence"], "is_http": is_http,
        "field_badges": [{"key": metric["id"], "label": metric["name"], "status": metric["status"],
                          "value": metric["value"], "reason": metric["reason"] or ""} for metric in metrics],
        "top_ips": top_ips, "top_user_agents": top_uas, "label_distribution": label_distribution,
        "ml_results": None,
    }
    return result


def _analysis(filename: str, content: bytes) -> Dict[str, Any]:
    """Small-input helper retained for tests and programmatic callers."""
    text = content.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return _analysis_from_lines(filename, lines[:MAX_ANALYSIS_LINES], detect_log_format(text), len(lines))


def _read_analysis_sample(path: Path) -> tuple[str, List[str], int]:
    """Read a bounded line sample while counting the full file without retaining it."""
    with path.open("rb") as handle:
        detection_text = handle.read(DETECTION_SAMPLE_BYTES).decode("utf-8", errors="replace")
    lines: List[str] = []
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            total_lines += 1
            if len(lines) < MAX_ANALYSIS_LINES:
                lines.append(line)
    return detection_text, lines, total_lines


async def _save_upload(file: UploadFile) -> tuple[str, Path, int]:
    """Stream an upload to disk. Never call UploadFile.read() without a limit."""
    filename = _safe_name(file.filename)
    path = UPLOAD_DIR / filename
    byte_count = 0
    with path.open("wb") as handle:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            handle.write(chunk)
            byte_count += len(chunk)
    return filename, path, byte_count


@app.post("/detect")
async def detect_format(file: UploadFile = File(...)):
    filename, path, size_bytes = await _save_upload(file)
    text, lines, total_lines = _read_analysis_sample(path)
    fmt = detect_log_format(text)
    return {"filename": filename, "saved_path": str(UPLOAD_DIR / filename), **fmt,
            "total_lines": total_lines, "size_bytes": size_bytes, "sample_lines": lines[:12],
            "analysis_limit": MAX_ANALYSIS_LINES}


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    filename, path, _ = await _save_upload(file)
    text, lines, total_lines = _read_analysis_sample(path)
    if not lines:
        return _empty_analysis(filename)
    return _analysis_from_lines(filename, lines, detect_log_format(text), total_lines)


@app.get("/features")
def features():
    return model_metadata()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_ui:app", host="0.0.0.0", port=8000, reload=True)

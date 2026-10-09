"""
log_parsers.py
==============
Format-specific parsers that convert raw log lines into a list of
NormalizedEvent dicts.

Each parser returns a ParseResult containing:
  - events       : list of NormalizedEvent dicts
  - format_info  : format detection result
  - summary      : format-specific summary metrics
  - errors       : list of detected errors/warnings
  - timeline     : list of timeline events (for application logs)

NormalizedEvent schema
----------------------
{
    "timestamp"  : str | None,
    "level"      : str | None,   # INFO / WARN / ERROR / DEBUG — null for Apache
    "source"     : str | None,   # thread / component — null for Apache
    "event_type" : str,          # HTTP_REQUEST | SOAP_REQUEST | IMPORT_VALIDATION | …
    "ip"         : str | None,   # null for non-HTTP logs
    "method"     : str | None,   # HTTP method — null for non-HTTP logs
    "url"        : str | None,   # null for non-HTTP logs
    "status_code": int | None,   # null for non-HTTP logs
    "bytes_sent" : int | None,   # null for non-HTTP logs
    "user_agent" : str | None,   # null for non-HTTP logs
    "message"    : str | None,
    "metadata"   : dict,
}
"""

import re
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# NormalizedEvent factory
# ---------------------------------------------------------------------------

def _event(
    event_type: str,
    timestamp: Optional[str] = None,
    level: Optional[str] = None,
    source: Optional[str] = None,
    component: Optional[str] = None,
    ip: Optional[str] = None,
    method: Optional[str] = None,
    url: Optional[str] = None,
    status_code: Optional[int] = None,
    bytes_sent: Optional[int] = None,
    user_agent: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    return {
        "timestamp":   timestamp,
        "level":       level,
        "source":      source,
        "component":   component or source,
        "event_type":  event_type,
        "ip":          ip,
        "method":      method,
        "url":         url,
        "status_code": status_code,
        "bytes_sent":  bytes_sent,
        "user_agent":  user_agent,
        "message":     message,
        "metadata":    metadata or {},
    }


# ---------------------------------------------------------------------------
# ParseResult factory
# ---------------------------------------------------------------------------

def _parse_result(
    events: List[Dict],
    format_info: Dict,
    summary: Dict,
    errors: List[Dict],
    timeline: List[Dict],
    applicable_fields: Dict,
) -> Dict[str, Any]:
    return {
        "events":           events,
        "format_info":      format_info,
        "summary":          summary,
        "errors":           errors,
        "timeline":         timeline,
        "applicable_fields": applicable_fields,
    }


# ==========================================================================
# Apache / Nginx Parser
# ==========================================================================

_APACHE_RE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<bytes>\S+)'
    r'(?:\s+"(?P<referrer>[^"]*)"\s+"(?P<user_agent>[^"]*)")?'
)


def parse_apache(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Parse Apache / Nginx Combined Log Format."""
    events: List[Dict] = []
    parse_errors = 0

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        m = _APACHE_RE.match(ln)
        if not m:
            parse_errors += 1
            continue
        g = m.groupdict()
        req_parts = g.get("request", "").split(" ", 2)
        method = req_parts[0] if len(req_parts) > 0 else None
        url    = req_parts[1] if len(req_parts) > 1 else None
        try:
            status = int(g["status"])
        except (TypeError, ValueError):
            status = None
        try:
            bsent = int(g["bytes"]) if g["bytes"] != "-" else None
        except (TypeError, ValueError):
            bsent = None

        events.append(_event(
            event_type  = "HTTP_REQUEST",
            timestamp   = g.get("timestamp"),
            ip          = g.get("ip"),
            method      = method,
            url         = url,
            status_code = status,
            bytes_sent  = bsent,
            user_agent  = g.get("user_agent"),
            message     = g.get("request"),
            metadata    = {"referrer": g.get("referrer", "-")},
        ))

    # Build summary
    total = len(events)
    status_dist: Dict[int, int] = {}
    method_dist: Dict[str, int] = {}
    ip_dist: Dict[str, int] = {}
    endpoint_dist: Dict[str, int] = {}
    user_agent_dist: Dict[str, int] = {}
    byte_values: List[int] = []

    for ev in events:
        sc = ev["status_code"]
        if sc is not None:
            status_dist[sc] = status_dist.get(sc, 0) + 1
        m = ev["method"]
        if m:
            method_dist[m] = method_dist.get(m, 0) + 1
        ip = ev["ip"]
        if ip:
            ip_dist[ip] = ip_dist.get(ip, 0) + 1
        if ev["url"]:
            endpoint_dist[ev["url"]] = endpoint_dist.get(ev["url"], 0) + 1
        if ev["user_agent"] and ev["user_agent"] != "-":
            user_agent_dist[ev["user_agent"]] = user_agent_dist.get(ev["user_agent"], 0) + 1
        if ev["bytes_sent"] is not None:
            byte_values.append(ev["bytes_sent"])

    unique_ips = len(ip_dist)
    top_ips = sorted(ip_dist.items(), key=lambda x: x[1], reverse=True)[:10]

    dated = []
    for ev in events:
        try:
            dated.append(datetime.strptime(ev["timestamp"], "%d/%b/%Y:%H:%M:%S %z"))
        except (TypeError, ValueError):
            pass
    seconds = (max(dated) - min(dated)).total_seconds() if len(dated) >= 2 else None
    error_4xx = sum(count for status, count in status_dist.items() if 400 <= status < 500)
    error_5xx = sum(count for status, count in status_dist.items() if 500 <= status < 600)
    known_bytes = len(byte_values) == total
    summary = {
        "total_log_events":    total,
        "total_requests":      total,
        "unique_ips":          unique_ips,
        "total_bytes":         sum(byte_values) if known_bytes else None,
        "status_distribution": status_dist,
        "method_distribution": method_dist,
        "top_ips":             top_ips,
        "top_endpoints":       sorted(endpoint_dist.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_user_agents":     sorted(user_agent_dist.items(), key=lambda x: x[1], reverse=True)[:10],
        "error_4xx_count":     error_4xx,
        "error_5xx_count":     error_5xx,
        "error_rate":          (error_4xx + error_5xx) / total if total else None,
        "request_rate":        total / seconds if seconds and seconds > 0 else None,
        "duration_seconds":    seconds,
        "parse_errors":        parse_errors,
    }

    applicable_fields = {
        "total_requests":      {"status": "available", "value": total},
        "unique_ips":          {"status": "available", "value": unique_ips},
        "data_transfer":       ({"status": "available", "value": sum(byte_values)} if known_bytes else
                                  {"status": "unavailable", "value": None,
                                   "reason": "One or more response byte counts were not recorded."}),
        "user_agents":         ({"status": "available", "value": len(user_agent_dist)} if user_agent_dist else
                                  {"status": "not_applicable", "value": None,
                                   "reason": "No User-Agent field exists in parsed records."}),
        "http_methods":        {"status": "available", "value": method_dist},
        "status_codes":        {"status": "available", "value": status_dist},
        "endpoints":           {"status": "available", "value": len(endpoint_dist)},
        "error_rate":          {"status": "available", "value": (error_4xx + error_5xx) / total if total else 0},
        "request_rate":        ({"status": "available", "value": total / seconds} if seconds and seconds > 0 else
                                  {"status": "unavailable", "value": None,
                                   "reason": "At least two parseable timestamps are required."}),
        "threat_intelligence": {"status": "unavailable", "value": None,
                                "reason": "No threat-intelligence provider is configured."},
        "import_status":       {"status": "not_applicable",
                                "reason": "This is an HTTP access log, not an import log."},
    }

    return _parse_result(events, format_info, summary, [], [], applicable_fields)


# ==========================================================================
# JSON Array Parser  (cj.log format)
# ==========================================================================

def parse_json_array(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Parse JSON-array-per-line log format (cj.log)."""
    events: List[Dict] = []
    parse_errors = 0

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            arr = json.loads(ln)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(arr, list) or len(arr) < 4:
            parse_errors += 1
            continue

        ip        = str(arr[3]) if arr[3] is not None else None
        timestamp = str(arr[2]) if len(arr) > 2 and arr[2] is not None else None
        ua        = str(arr[5]) if len(arr) > 5 and arr[5] is not None else None
        url       = str(arr[8]) if len(arr) > 8 and arr[8] is not None else None

        events.append(_event(
            event_type = "HTTP_REQUEST",
            timestamp  = timestamp,
            ip         = ip,
            method     = None,
            url        = url,
            user_agent = ua,
            message    = None,
            metadata   = {"raw_array_length": len(arr)},
        ))

    ip_dist = {}
    for ev in events:
        ip = ev["ip"]
        if ip:
            ip_dist[ip] = ip_dist.get(ip, 0) + 1

    total = len(events)
    summary = {
        "total_log_events": total,
        "total_requests":   total,
        "unique_ips":       len(ip_dist),
        "parse_errors":     parse_errors,
    }

    applicable_fields = {
        "total_requests":      {"status": "available", "value": total},
        "unique_ips":          {"status": "available", "value": len(ip_dist)},
        "data_transfer":       {"status": "not_applicable",
                                "reason": "Byte transfer size is not recorded in this log format."},
        "status_codes":        {"status": "not_applicable",
                                "reason": "HTTP status codes are not present in the JSON array format."},
        "import_status":       {"status": "not_applicable",
                                "reason": "This is a web-request log, not an import log."},
        "threat_intelligence": {"status": "unavailable", "value": None,
                                "reason": "No threat-intelligence provider is configured."},
        "user_agents":         ({"status": "available", "value": sum(1 for ev in events if ev["user_agent"])}
                                if any(ev["user_agent"] for ev in events) else
                                {"status": "not_applicable", "value": None,
                                 "reason": "No User-Agent field exists in parsed records."}),
    }

    return _parse_result(events, format_info, summary, [], [], applicable_fields)


# ==========================================================================
# Oracle Data Loader / SOAP Parser
# ==========================================================================

# Timestamp at start of line or after bracket
_ORA_TS     = re.compile(r'(?:^|\s)(\d{2}:\d{2}:\d{2})\b')
_ORA_LEVEL  = re.compile(r'\b(INFO|WARN|WARNING|ERROR|DEBUG|FATAL)\b')
_ORA_THREAD = re.compile(r'\[([^\]]+)\]')

# Specific event patterns
_ORA_LOGIN   = re.compile(r'(login\s+success|loginresult\s+success|logged\s+in|authentication\s+success|session\s+token\s+acquired)', re.IGNORECASE)
_ORA_LOGOUT  = re.compile(r'(logout|logged\s+out|session\s+terminated)', re.IGNORECASE)
_ORA_SOAP_RQ = re.compile(r'<soapenv:|SOAP\s+Request|Sending\s+SOAP', re.IGNORECASE)
_ORA_SOAP_RS = re.compile(r'SOAP\s+Response|Received\s+SOAP', re.IGNORECASE)
_ORA_VAL_ST  = re.compile(r'(Validat|DataValidat)', re.IGNORECASE)
_ORA_VAL_OK  = re.compile(r'Validation\s+(Passed|Success|OK|Complete)', re.IGNORECASE)
_ORA_IMPORT  = re.compile(r'(Import|Submission|SubmitStatus)', re.IGNORECASE)
_ORA_CSV_SEG = re.compile(r'Segment\s+(\d+)\s+of\s+(\d+)', re.IGNORECASE)
_ORA_THREADS = re.compile(r'threads?\s*[=:]\s*(\d+)', re.IGNORECASE)
_ORA_RETRY   = re.compile(r'(retry\s+attempt|retrying|attempt\s+\d+\s*:)', re.IGNORECASE)
_ORA_ERR_CODE = re.compile(r'SBL-[A-Z]+-\d+')
_ORA_RATE_LIM = re.compile(r'SBL-ODU-01005', re.IGNORECASE)
_ORA_EXEC_ST  = re.compile(r'(Execution\s+Start|Starting\s+import|Beginning\s+data)', re.IGNORECASE)
_ORA_EXEC_END = re.compile(r'(Execution\s+(Complete|End|Finish)|Import\s+(Complete|Finish|Success))', re.IGNORECASE)
_ORA_SUBMIT   = re.compile(r'(\d+)%\s*(complete|submitted|processed)', re.IGNORECASE)
_ORA_SUCCESS  = re.compile(r'(SUCCESS|Successfully\s+completed|Import\s+Succeeded)', re.IGNORECASE)
_ORA_FAIL     = re.compile(r'(FAILED|Import\s+Failed|Submission\s+Failed)', re.IGNORECASE)
_ORA_OP       = re.compile(r'operation\s*[=:]\s*(insert|upsert|delete|export)', re.IGNORECASE)
_ORA_RECTYPE  = re.compile(r'(record\s+type|object)\s*[=:]\s*(\w+)', re.IGNORECASE)


def parse_oracle_data_loader(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Parse Oracle Data Loader / SOAP application log."""
    events: List[Dict] = []
    error_events: List[Dict] = []
    timeline: List[Dict] = []

    # Extracted metadata
    operation       = None
    record_type     = None
    thread_count    = None
    csv_segments    = None
    validation_status = None
    import_status   = None
    submission_pct  = None
    start_ts        = None
    end_ts          = None
    rate_limit_errors = 0
    retries         = 0
    warnings        = 0

    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue

        # Extract timestamp
        ts_m = _ORA_TS.search(raw)
        ts   = ts_m.group(1) if ts_m else None

        # Extract level
        lvl_m = _ORA_LEVEL.search(raw)
        level = lvl_m.group(1) if lvl_m else "INFO"

        # Extract source/thread
        thread_m = _ORA_THREAD.search(raw)
        source   = thread_m.group(1) if thread_m else None

        # --- Classify event type ---
        # Priority order: specific business events BEFORE generic SOAP catch-alls.
        event_type = "LOG_ENTRY"
        meta: Dict[str, Any] = {}

        if _ORA_LOGIN.search(raw):
            event_type = "LOGIN"
            timeline.append({"timestamp": ts, "event": "Login successful"})
        elif _ORA_LOGOUT.search(raw):
            event_type = "LOGOUT"
            timeline.append({"timestamp": ts, "event": "Logout completed"})
        elif _ORA_RETRY.search(raw):
            event_type = "RETRY"
            retries += 1
            timeline.append({"timestamp": ts, "event": "Retry performed"})
        elif _ORA_SOAP_RQ.search(raw):
            event_type = "SOAP_REQUEST"
            meta["direction"] = "outbound"
        elif _ORA_SOAP_RS.search(raw):
            event_type = "SOAP_RESPONSE"
            meta["direction"] = "inbound"
        elif _ORA_VAL_OK.search(raw):
            event_type = "IMPORT_VALIDATION"
            validation_status = "PASSED"
            meta["result"] = "PASSED"
            timeline.append({"timestamp": ts, "event": "Validation passed"})
        elif _ORA_VAL_ST.search(raw):
            event_type = "IMPORT_VALIDATION"
            timeline.append({"timestamp": ts, "event": "Import validation started"})
        elif _ORA_EXEC_ST.search(raw):
            event_type = "EXECUTION_START"
            start_ts = ts
            timeline.append({"timestamp": ts, "event": "Execution started"})
        elif _ORA_EXEC_END.search(raw):
            event_type = "EXECUTION_END"
            end_ts = ts
        elif _ORA_IMPORT.search(raw):
            event_type = "IMPORT_SUBMISSION"
            sub_m = _ORA_SUBMIT.search(raw)
            if sub_m:
                submission_pct = int(sub_m.group(1))
                meta["progress"] = f"{submission_pct}%"
                if submission_pct == 100:
                    timeline.append({"timestamp": ts, "event": "Submission reached 100%"})

        # --- Error detection ---
        if _ORA_RATE_LIM.search(raw):
            event_type = "RATE_LIMIT_ERROR"
            rate_limit_errors += 1
            error_events.append({
                "code":        "SBL-ODU-01005",
                "timestamp":   ts,
                "severity":    "retryable",
                "message":     raw,
                "description": (
                    "SOAP request rate-limit error. The application "
                    "automatically retried after a short delay."
                ),
            })
            timeline.append({"timestamp": ts, "event": "Rate-limit error detected (SBL-ODU-01005)"})
        elif _ORA_ERR_CODE.search(raw) and level in ("ERROR", "FATAL"):
            err_code = _ORA_ERR_CODE.search(raw).group(0)
            error_events.append({
                "code":      err_code,
                "timestamp": ts,
                "severity":  "error",
                "message":   raw,
            })

        if level in ("WARN", "WARNING"):
            warnings += 1

        # --- Extract metadata fields ---
        if operation is None:
            op_m = _ORA_OP.search(raw)
            if op_m:
                operation = op_m.group(1).lower()

        if record_type is None:
            rt_m = _ORA_RECTYPE.search(raw)
            if rt_m:
                record_type = rt_m.group(2).lower()

        if thread_count is None:
            th_m = _ORA_THREADS.search(raw)
            if th_m:
                thread_count = int(th_m.group(1))

        if csv_segments is None:
            seg_m = _ORA_CSV_SEG.search(raw)
            if seg_m:
                csv_segments = f"{seg_m.group(1)} of {seg_m.group(2)}"

        # Import final status
        if _ORA_SUCCESS.search(raw):
            import_status = "SUCCESS"
            timeline.append({"timestamp": ts, "event": "Import completed successfully"})
        elif _ORA_FAIL.search(raw) and import_status != "SUCCESS":
            import_status = "FAILED"

        events.append(_event(
            event_type = event_type,
            timestamp  = ts,
            level      = level,
            source     = source,
            message    = raw,
            metadata   = meta,
        ))

    # Build summary
    total = len(events)
    level_dist: Dict[str, int] = {}
    for ev in events:
        lv = ev["level"] or "UNKNOWN"
        level_dist[lv] = level_dist.get(lv, 0) + 1

    soap_count    = sum(1 for ev in events if ev["event_type"] in ("SOAP_REQUEST", "SOAP_RESPONSE"))
    import_events = sum(1 for ev in events if ev["event_type"] == "IMPORT_SUBMISSION")

    summary = {
        "total_log_events":    total,
        "operation":           operation,
        "record_type":         record_type,
        "thread_count":        thread_count,
        "csv_segments":        csv_segments,
        "validation_status":   validation_status,
        "import_status":       import_status or "UNKNOWN",
        "submission_progress": f"{submission_pct}%" if submission_pct is not None else "N/A",
        "soap_operations":     soap_count,
        "rate_limit_errors":   rate_limit_errors,
        "retries":             retries,
        "warnings":            warnings,
        "level_distribution":  level_dist,
    }

    applicable_fields = {
        "total_log_events":    {"status": "available", "value": total},
        "import_status":       {"status": "available" if import_status else "not_available",
                                "value": import_status,
                                "reason": None if import_status else "No import status line found."},
        "operation":           {"status": "available" if operation else "not_available",
                                "value": operation},
        "record_type":         {"status": "available" if record_type else "not_available",
                                "value": record_type},
        "threads":             {"status": "available" if thread_count is not None else "not_available",
                                "value": thread_count},
        "csv_segments":        {"status": "available" if csv_segments else "not_available",
                                "value": csv_segments},
        "validation":          {"status": "available" if validation_status else "not_available",
                                "value": validation_status},
        "submission":          {"status": "available" if submission_pct is not None else "not_available",
                                "value": f"{submission_pct}%" if submission_pct is not None else None},
        "total_requests":      {"status": "not_applicable",
                                "reason": "This is an application/import log; HTTP requests are not applicable."},
        "unique_ips":          {"status": "not_applicable",
                                "reason": "No client IP addresses were found in this log."},
        "data_transfer":       {"status": "not_applicable",
                                "reason": "The log does not contain HTTP response/request size information."},
        "user_agents":         {"status": "not_applicable",
                                "reason": "No User-Agent fields were present in this log."},
        "http_status_codes":   {"status": "not_applicable",
                                "reason": "This is not an HTTP access log."},
        "threat_intelligence": {"status": "not_applicable",
                                "reason": (
                                    "This is an application/import log and does not contain "
                                    "sufficient network/security indicators for IP-based threat analysis."
                                )},
        "targeted_endpoints":  {"status": "not_applicable",
                                "reason": "This is not an HTTP access log; no HTTP endpoints are present."},
    }

    return _parse_result(events, format_info, summary, error_events, timeline, applicable_fields)


# ==========================================================================
# Java Application Log Parser
# ==========================================================================

_JAVA_LINE_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+'
    r'(?:\[(?P<thread>[^\]]+)\]\s+)?(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|SEVERE)\s+'
    r'(?P<source>[\w.$-]+)\s*[-–:]\s*'
    r'(?P<message>.+)$'
)
_JAVA_EX_RE  = re.compile(r'(Exception|Error):\s*(.+)', re.IGNORECASE)


def parse_java_application(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Parse generic Java application log (log4j / logback style)."""
    events: List[Dict] = []
    error_events: List[Dict] = []
    timeline: List[Dict] = []
    level_dist: Dict[str, int] = {}
    parse_errors = 0

    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        m = _JAVA_LINE_RE.match(raw)
        if m:
            g        = m.groupdict()
            level    = g["level"]
            level_dist[level] = level_dist.get(level, 0) + 1
            msg_lower = g["message"].lower()
            evt_type = "LOG_ENTRY"
            if level in ("ERROR", "FATAL", "SEVERE"):
                evt_type = "EXCEPTION" if "exception" in msg_lower else "ERROR"
                ex_m = _JAVA_EX_RE.search(g["message"])
                error_events.append({
                    "timestamp":   g["timestamp"],
                    "severity":    level.lower(),
                    "message":     g["message"],
                    "exception":   ex_m.group(0) if ex_m else None,
                })
            elif level in ("WARN", "WARNING"):
                evt_type = "WARNING"
                error_events.append({"timestamp": g["timestamp"], "severity": "warning", "message": g["message"]})
            elif "retry" in msg_lower:
                evt_type = "RETRY"
            elif any(term in msg_lower for term in ("started", "starting")):
                evt_type = "APPLICATION_START"
            elif any(term in msg_lower for term in ("completed", "success", "succeeded", "finished")):
                evt_type = "PROCESS_COMPLETE"
            elif any(term in msg_lower for term in ("stopped", "shutting down")):
                evt_type = "APPLICATION_STOP"
            events.append(_event(
                event_type = evt_type,
                timestamp  = g["timestamp"],
                level      = level,
                source     = g["source"],
                component  = g["source"],
                message    = g["message"],
                metadata   = {"thread": g.get("thread")} if g.get("thread") else {},
            ))
            timeline.append({"timestamp": g["timestamp"], "event": g["message"], "event_type": evt_type})
        else:
            parse_errors += 1
            events.append(_event(
                event_type = "LOG_CONTINUATION" if raw.startswith(("at ", "Caused by:")) else "UNKNOWN_EVENT",
                message    = raw,
            ))

    total = len(events)
    timestamps = []
    for event in events:
        try:
            timestamps.append(datetime.fromisoformat(event["timestamp"].replace(",", ".")))
        except (AttributeError, ValueError):
            pass
    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None
    components = sorted({event["component"] for event in events if event["component"]})
    retries = sum(event["event_type"] == "RETRY" for event in events)
    successes = sum(event["event_type"] == "PROCESS_COMPLETE" for event in events)
    failures = sum(event["event_type"] in ("ERROR", "EXCEPTION") for event in events)
    recovered = min(failures, retries, successes) if failures and retries and successes else 0
    summary = {
        "total_log_events":  total,
        "level_distribution": level_dist,
        "error_count":        level_dist.get("ERROR", 0) + level_dist.get("FATAL", 0) + level_dist.get("SEVERE", 0),
        "warning_count":      level_dist.get("WARN", 0) + level_dist.get("WARNING", 0),
        "exception_count":    sum(event["event_type"] == "EXCEPTION" for event in events),
        "components":         components,
        "retry_count":        retries,
        "successful_operations": successes,
        "failed_operations":  failures,
        "recovered_failures": recovered,
        "duration_seconds":   duration,
        "parse_errors":       parse_errors,
    }

    applicable_fields = {
        "total_log_events":    {"status": "available", "value": total},
        "log_levels":          {"status": "available"},
        "components":          {"status": "available", "value": components},
        "execution_duration":  ({"status": "available", "value": duration} if duration is not None else
                                  {"status": "unavailable", "value": None,
                                   "reason": "At least two parseable timestamps are required."}),
        "total_requests":      {"status": "not_applicable",
                                "reason": "This is an application log; HTTP requests are not applicable."},
        "unique_ips":          {"status": "not_applicable",
                                "reason": "No IP addresses were found."},
        "data_transfer":       {"status": "not_applicable",
                                "reason": "No byte transfer information is present."},
        "threat_intelligence": {"status": "not_applicable",
                                "reason": "This is an application log without network security indicators."},
    }

    return _parse_result(events, format_info, summary, error_events, timeline, applicable_fields)


# ==========================================================================
# Generic, application, and mixed parsers
# ==========================================================================

_GENERIC_LINE_RE = re.compile(
    r'^(?P<timestamp>(?:\d{4}[-/]\d{2}[-/]\d{2}[ T])?\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?\s*'
    r'(?:\[(?P<source>[^\]]+)\]\s*)?'
    r'(?:(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|SEVERE)\b)?\s*'
    r'(?P<message>.*)$', re.I
)

_IP_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def parse_generic(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Conservative fallback: only extract timestamp/level/source when explicit."""
    events, errors, timeline = [], [], []
    levels, components = Counter(), set()
    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        match = _GENERIC_LINE_RE.match(raw)
        groups = match.groupdict() if match else {}
        timestamp = groups.get("timestamp") or None
        level = (groups.get("level") or "").upper() or None
        source = groups.get("source") or None
        message = groups.get("message") or raw
        event_type = "UNKNOWN_EVENT"
        if level in ("ERROR", "FATAL", "SEVERE"):
            event_type = "ERROR"
            errors.append({"timestamp": timestamp, "severity": level.lower(), "message": message})
        elif level in ("WARN", "WARNING"):
            event_type = "WARNING"
            errors.append({"timestamp": timestamp, "severity": "warning", "message": message})
        if level:
            levels[level] += 1
        if source:
            components.add(source)
            
        ip = None
        ip_match = _IP_REGEX.search(raw)
        if ip_match:
            ip = ip_match.group(0)
            
        events.append(_event(event_type=event_type, timestamp=timestamp, level=level,
                             source=source, component=source, message=message, ip=ip))
        if timestamp:
            timeline.append({"timestamp": timestamp, "event": message, "event_type": event_type})
    summary = {
        "total_log_events": len(events), "note": "Format not recognised; only explicit fields were extracted.",
        "level_distribution": dict(levels),
        "error_count": sum(event["event_type"] == "ERROR" for event in events),
        "warning_count": sum(event["event_type"] == "WARNING" for event in events),
        "components": sorted(components),
        "parse_errors": sum(1 for event in events if event["event_type"] == "UNKNOWN_EVENT"
                            and not event["timestamp"] and not event["level"] and not event["source"]),
    }
    fields = {
        "total_log_events": {"status": "available", "value": len(events)},
        "total_requests": {"status": "not_applicable", "value": None,
                           "reason": "The log was not validated as an HTTP access log."},
        "unique_ips": {"status": "not_applicable", "value": None,
                       "reason": "No reliable client-IP field was identified."},
        "data_transfer": {"status": "not_applicable", "value": None,
                          "reason": "No reliable response-size field was identified."},
        "threat_intelligence": {"status": "not_applicable", "value": None,
                                "reason": "An unrecognised log has no validated security schema."},
        "log_levels": ({"status": "available", "value": dict(levels)} if levels else
                       {"status": "unavailable", "value": None, "reason": "No explicit log levels were found."}),
        "components": ({"status": "available", "value": sorted(components)} if components else
                       {"status": "unavailable", "value": None, "reason": "No explicit component fields were found."}),
    }
    return _parse_result(events, format_info, summary, errors, timeline, fields)


def parse_application_log(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    return parse_generic(lines, format_info)


def parse_mixed(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """Parse identifiable lines independently and preserve their source format."""
    events, errors, timeline, failed = [], [], [], 0
    for line in lines:
        if not line.strip():
            continue
        if _APACHE_RE.match(line):
            candidate, source_format = parse_apache([line], {"format": "apache_access"}), "apache_access"
        elif line.strip().startswith("["):
            candidate, source_format = parse_json_array([line], {"format": "json_array"}), "json_array"
        elif _JAVA_LINE_RE.match(line):
            candidate, source_format = parse_java_application([line], {"format": "java_application"}), "java_application"
        else:
            candidate, source_format = parse_generic([line], {"format": "generic"}), "generic"
        if not candidate["events"]:
            failed += 1
        for event in candidate["events"]:
            event["metadata"]["source_format"] = source_format
            events.append(event)
        errors.extend(candidate["errors"])
        timeline.extend(candidate["timeline"])
    summary = {"total_log_events": len(events), "parse_errors": failed,
               "components": format_info.get("components", []),
               "note": "Metrics requiring a single schema are not applicable for mixed logs."}
    fields = {
        "total_log_events": {"status": "available", "value": len(events)},
        "parsing": {"status": "available", "value": {"parsed_lines": len(events), "failed_lines": failed}},
        "total_requests": {"status": "not_applicable", "value": None,
                           "reason": "Mixed schemas cannot produce one reliable request metric."},
        "threat_intelligence": {"status": "not_applicable", "value": None,
                                "reason": "Security analysis requires a single validated schema."},
    }
    return _parse_result(events, format_info, summary, errors, timeline, fields)


# ==========================================================================
# Router: get_parser + parse
# ==========================================================================

def parse_log(lines: List[str], format_info: Dict) -> Dict[str, Any]:
    """
    Route to the correct parser based on the detected format.

    Parameters
    ----------
    lines       : non-empty log lines (list of str)
    format_info : result from log_format_detector.detect_log_format()

    Returns
    -------
    ParseResult dict (see module docstring).
    """
    fmt = format_info.get("format", "generic")

    dispatch = {
        "apache_access":      parse_apache,
        "nginx_access":       parse_apache,   # same pattern
        "json_array":         parse_json_array,
        "oracle_data_loader": parse_oracle_data_loader,
        "java_application":   parse_java_application,
        "application_log":    parse_application_log,
        "mixed":              parse_mixed,
        "empty":              parse_generic,
        "malformed":          parse_generic,
        "generic":            parse_generic,
    }

    parser_fn = dispatch.get(fmt, parse_generic)
    return parser_fn(lines, format_info)


# ---------------------------------------------------------------------------
# CLI: python log_parsers.py <logfile>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json as _json
    from log_format_detector import detect_log_format_from_file

    if len(sys.argv) < 2:
        print("Usage: python log_parsers.py <logfile>")
        sys.exit(1)

    path = sys.argv[1]
    fmt  = detect_log_format_from_file(path)
    print(f"Detected format: {fmt}")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw_lines = [l.rstrip("\n") for l in fh.readlines()]

    result = parse_log(raw_lines, fmt)
    print(f"\nTotal events   : {len(result['events'])}")
    print(f"Summary        :")
    print(_json.dumps(result["summary"], indent=2))
    if result["errors"]:
        print(f"\nErrors/Warnings ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"  [{e.get('severity','?').upper()}] {e.get('code','—')} — {e.get('description', e.get('message',''))[:120]}")
    if result["timeline"]:
        print(f"\nTimeline ({len(result['timeline'])} events):")
        for t in result["timeline"]:
            print(f"  {t.get('timestamp','?'):>10}  {t.get('event','')}")

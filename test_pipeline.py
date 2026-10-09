"""
test_pipeline.py — Regression & Validation Tests for the Log Pipeline
=======================================================================
Tests every parser and the format detector.

Run with:  python test_pipeline.py
"""

import os
import sys
import textwrap

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from log_format_detector import detect_log_format, detect_log_format_from_file
from log_parsers import parse_log
from ml_engine import analyze_http_events

# ============================================================================
# Lightweight test harness
# ============================================================================

PASS = "✅ PASS"
FAIL = "❌ FAIL"

_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================================
# Sample log fixtures (inline — no file I/O required for most tests)
# ============================================================================

APACHE_SAMPLE = textwrap.dedent("""\
    127.0.0.1 - frank [10/Oct/2023:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://www.example.com/" "Mozilla/4.08 [en]"
    192.168.0.1 - - [11/Oct/2023:14:56:40 -0700] "POST /login.php HTTP/1.1" 401 512 "-" "Mozilla/5.0"
    10.0.0.5 - - [12/Oct/2023:09:12:00 -0700] "GET /index.html HTTP/1.1" 200 1024 "http://google.com/" "Chrome/110"
    8.8.8.8 - - [13/Oct/2023:18:22:11 -0700] "GET /images/logo.png HTTP/1.1" 404 315 "-" "Safari/537.36"
    172.16.0.2 - admin [14/Oct/2023:20:00:00 -0700] "GET /dashboard HTTP/1.1" 200 8492 "http://example.com/login" "Firefox/102"
""")

JSON_ARRAY_SAMPLE = textwrap.dedent("""\
    [null,null,"2023-01-08 08:07:15","104.28.209.153","61901","Mozilla/5.0 (Windows NT 10.0)","en",null]
    [null,null,"2023-01-08 08:08:22","8.8.8.8","45120","python-requests/2.28","en",null]
    [null,null,"2023-01-08 08:09:01","192.168.1.100","55432","Chrome/110","en",null]
""")

ORACLE_SAMPLE = textwrap.dedent("""\
    07:51:54 INFO  [main] BulkOpsClient - Execution Start: Oracle Siebel Data Loader v8.2
    07:51:54 INFO  [main] BulkOpsClient - Operation: insert
    07:51:54 INFO  [main] BulkOpsClient - Record Type: account
    07:51:54 INFO  [main] BulkOpsClient - Threads: 1
    07:51:55 INFO  [main] BulkOpsClient - SOAP Response received: LoginResult SUCCESS
    07:51:55 INFO  [main] BulkOpsClient - Login successful. Session token acquired.
    07:51:55 INFO  [main] BulkOpsClient - CSV split into 1 of 1 segment(s)
    07:51:55 INFO  [main] BulkOpsClient - Starting Data Validation for Segment 1 of 1
    07:51:55 INFO  [main] BulkOpsClient - Sending SOAP Request: DataValidation
    07:51:55 ERROR [main] BulkOpsClient - SBL-ODU-01005: SOAP Request rate limit exceeded. Retrying in 1 second.
    07:51:56 INFO  [main] BulkOpsClient - Retry attempt 1: Sending SOAP Request for DataValidation
    07:52:01 INFO  [main] BulkOpsClient - Validation Passed. 187 of 187 records valid.
    07:52:01 INFO  [main] BulkOpsClient - ValidationStatus: PASSED
    07:52:01 INFO  [main] BulkOpsClient - Starting CSV submission for Segment 1 of 1
    07:52:06 INFO  [main] BulkOpsClient - ImportStatus: 100% complete — 187 of 187 records submitted
    07:52:07 INFO  [main] BulkOpsClient - Import operation completed with status: SUCCESS
    07:52:08 INFO  [main] BulkOpsClient - Sending SOAP logout request
    07:52:09 INFO  [main] BulkOpsClient - Session terminated. Goodbye.
""")

JAVA_SAMPLE = textwrap.dedent("""\
    2023-10-08 14:32:01.123 INFO  com.example.Application - Server started on port 8080
    2023-10-08 14:32:05.456 DEBUG com.example.db.ConnectionPool - Pool size: 10
    2023-10-08 14:33:10.789 WARN  com.example.security.Filter - Invalid token received
    2023-10-08 14:33:11.000 ERROR com.example.auth.Service - Authentication failed: NullPointerException
    2023-10-08 14:34:00.333 INFO  com.example.Application - Shutting down
""")

EMPTY_LOG    = ""
MALFORMED_LOG = textwrap.dedent("""\
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    !!!!!!!!!!!!garbage!!!!!!!!!!!!!!!!
    ===================================
    12345 xyz ??? ??? ??? ???
""")

MIXED_LOG = APACHE_SAMPLE[:200] + "\n" + ORACLE_SAMPLE[:200]


# ============================================================================
# 1. Format Detection Tests
# ============================================================================

section("1. FORMAT DETECTION")

# Apache
r = detect_log_format(APACHE_SAMPLE)
check("Apache sample → detected as apache_access",     r["format"] == "apache_access",
      f"got {r['format']} ({r['confidence']})")
check("Apache confidence ≥ 0.75",                       r["confidence"] >= 0.75)

# JSON array
r = detect_log_format(JSON_ARRAY_SAMPLE)
check("JSON-array sample → detected as json_array",    r["format"] == "json_array",
      f"got {r['format']} ({r['confidence']})")

# Oracle Data Loader
r = detect_log_format(ORACLE_SAMPLE)
check("Oracle sample → detected as oracle_data_loader", r["format"] == "oracle_data_loader",
      f"got {r['format']} ({r['confidence']})")
check("Oracle NOT detected as apache_access",           r["format"] != "apache_access",
      f"got {r['format']}")
check("Oracle confidence ≥ 0.60",                       r["confidence"] >= 0.60)

# Java
r = detect_log_format(JAVA_SAMPLE)
check("Java sample → detected as java_application",    r["format"] == "java_application",
      f"got {r['format']} ({r['confidence']})")

# Empty
r = detect_log_format(EMPTY_LOG)
check("Empty log → format returned (no crash)",         r["format"] is not None)

# Malformed
r = detect_log_format(MALFORMED_LOG)
check("Malformed log → generic or low confidence",
      r["format"] == "generic" or r["confidence"] <= 0.55,
      f"got {r['format']} ({r['confidence']})")


# ============================================================================
# 2. Apache Parser Tests
# ============================================================================

section("2. APACHE PARSER")

fmt = {"format": "apache_access", "confidence": 0.99, "description": "Apache Log", "parser": "apache_access"}
result = parse_log(APACHE_SAMPLE.splitlines(), fmt)

check("Apache: events parsed > 0",            len(result["events"]) > 0,
      str(len(result["events"])))
check("Apache: event_type = HTTP_REQUEST",
      all(e["event_type"] == "HTTP_REQUEST" for e in result["events"]))
check("Apache: IP addresses present",
      all(e["ip"] is not None for e in result["events"]))
check("Apache: total_requests > 0",            result["summary"]["total_requests"] > 0)
check("Apache: unique_ips > 0",                result["summary"]["unique_ips"] > 0)
check("Apache: 'total_requests' is available",
      result["applicable_fields"]["total_requests"]["status"] == "available")
check("Apache: 'unique_ips' is available",
      result["applicable_fields"]["unique_ips"]["status"] == "available")
check("Apache: 'import_status' is not_applicable",
      result["applicable_fields"]["import_status"]["status"] == "not_applicable")

# Nginx combined access logs share Apache's public combined field structure.
nginx_fmt = {"format": "nginx_access", "confidence": 0.99, "description": "Nginx", "parser": "nginx_access"}
nginx_result = parse_log(APACHE_SAMPLE.splitlines(), nginx_fmt)
check("Nginx combined format: routes to HTTP parser", nginx_result["summary"]["total_requests"] == 5)


# ============================================================================
# 3. Oracle Data Loader Parser Tests  (REGRESSION: sample.log)
# ============================================================================

section("3. ORACLE DATA LOADER PARSER (sample.log regression)")

fmt = {"format": "oracle_data_loader", "confidence": 0.95, "description": "Oracle Data Loader", "parser": "oracle_data_loader"}
result = parse_log(ORACLE_SAMPLE.splitlines(), fmt)

sm = result["summary"]
af = result["applicable_fields"]
errors = result["errors"]
timeline = result["timeline"]

check("Oracle: events parsed > 0",                     len(result["events"]) > 0,
      str(len(result["events"])))
check("Oracle: login event detected",
      any(e["event_type"] == "LOGIN" for e in result["events"]))
check("Oracle: import validation detected",
      any(e["event_type"] == "IMPORT_VALIDATION" for e in result["events"]))
check("Oracle: SOAP request detected",
      any(e["event_type"] == "SOAP_REQUEST" for e in result["events"]))
check("Oracle: SBL-ODU-01005 detected",
      any(e.get("code") == "SBL-ODU-01005" for e in errors),
      f"errors={errors}")
check("Oracle: retry detected",                        sm.get("retries", 0) > 0,
      f"retries={sm.get('retries')}")
check("Oracle: validation status = PASSED",            sm.get("validation_status") == "PASSED",
      f"got {sm.get('validation_status')}")
check("Oracle: import status = SUCCESS",               sm.get("import_status") == "SUCCESS",
      f"got {sm.get('import_status')}")
check("Oracle: CSV segments detected",                 sm.get("csv_segments") is not None,
      f"got {sm.get('csv_segments')}")
check("Oracle: logout in timeline",
      any("logout" in (t.get("event") or "").lower() for t in timeline))
check("Oracle: 'total_requests' is NOT_APPLICABLE",
      af["total_requests"]["status"] == "not_applicable")
check("Oracle: 'unique_ips' is NOT_APPLICABLE",
      af["unique_ips"]["status"] == "not_applicable")
check("Oracle: 'data_transfer' is NOT_APPLICABLE",
      af["data_transfer"]["status"] == "not_applicable")
check("Oracle: 'user_agents' is NOT_APPLICABLE",
      af["user_agents"]["status"] == "not_applicable")
check("Oracle: 'threat_intelligence' is NOT_APPLICABLE",
      af["threat_intelligence"]["status"] == "not_applicable")
check("Oracle: NO IP addresses in events",
      all(e["ip"] is None for e in result["events"]))

# Test SBL-ODU-01005 is retryable (not classified as attack)
if errors:
    sbl_err = next((e for e in errors if e.get("code") == "SBL-ODU-01005"), None)
    check("Oracle: SBL-ODU-01005 severity = retryable (not fatal/attack)",
          sbl_err is not None and sbl_err.get("severity") == "retryable",
          f"severity={sbl_err.get('severity') if sbl_err else 'N/A'}")


# ============================================================================
# 4. Java Application Parser Tests
# ============================================================================

section("4. JAVA APPLICATION PARSER")

fmt = {"format": "java_application", "confidence": 0.90, "description": "Java App Log", "parser": "java_application"}
result = parse_log(JAVA_SAMPLE.splitlines(), fmt)

sm = result["summary"]
af = result["applicable_fields"]

check("Java: events parsed > 0",              len(result["events"]) > 0)
check("Java: error count ≥ 1",               sm.get("error_count", 0) >= 1)
check("Java: warning count ≥ 1",             sm.get("warning_count", 0) >= 1)
check("Java: 'total_requests' NOT_APPLICABLE",
      af["total_requests"]["status"] == "not_applicable")
check("Java: 'unique_ips' NOT_APPLICABLE",
      af["unique_ips"]["status"] == "not_applicable")

# ML/ATTP activation is conditional on complete source-backed HTTP fields.
ml_http = parse_log(APACHE_SAMPLE.splitlines(), {"format": "apache_access", "confidence": 0.99,
                    "description": "Apache", "parser": "apache_access"})
ml_result = analyze_http_events(ml_http["events"])
check("HTTP log: saved Random Forest runs", ml_result["model"]["status"] == "available",
      ml_result.get("reason", ""))
check("HTTP log: ATTP runs", ml_result["attp"]["status"] == "available",
      ml_result["attp"].get("reason", ""))
ml_java = analyze_http_events(result["events"])
check("Application log: ML skips without HTTP events", ml_java["status"] == "not_applicable")


# ============================================================================
# 5. Empty / Malformed / Mixed log
# ============================================================================

section("5. EDGE CASES")

# Empty
fmt = {"format": "generic", "confidence": 0.5, "description": "Generic", "parser": "generic"}
result = parse_log([], fmt)
check("Empty log: no crash, 0 events",         len(result["events"]) == 0)

# Malformed
result = parse_log(MALFORMED_LOG.splitlines(), fmt)
check("Malformed log: no crash",               result is not None)
check("Malformed log: 'total_requests' NOT_APPLICABLE",
      result["applicable_fields"]["total_requests"]["status"] == "not_applicable")
explicit_malformed = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET / HTTP/1.1" broken'
r = detect_log_format(explicit_malformed)
check("Partial HTTP record → malformed", r["format"] == "malformed", f"got {r['format']}")

# Generic application log
GENERIC_APP = "12:00:01 INFO [worker-1] job started\n12:00:02 ERROR [worker-1] job failed"
r = detect_log_format(GENERIC_APP)
check("Generic application sample → application_log", r["format"] == "application_log",
      f"got {r['format']}")
result = parse_log(GENERIC_APP.splitlines(), r)
check("Generic application: level metrics extracted",
      result["summary"].get("level_distribution", {}).get("INFO") == 1)
check("Generic application: error extracted", result["summary"].get("error_count") == 1)

# Mixed full records (two distinct validated structures)
MIXED_FULL = "\n".join([
    APACHE_SAMPLE.splitlines()[0],
    JAVA_SAMPLE.splitlines()[0],
])
r = detect_log_format(MIXED_FULL)
check("Mixed log → mixed", r["format"] == "mixed", f"got {r['format']} ({r['components']})")
result = parse_log(MIXED_FULL.splitlines(), r)
check("Mixed log: source format retained", all("source_format" in e["metadata"] for e in result["events"]))

# Concrete Apache metric checks
fmt = {"format": "apache_access", "confidence": 0.99, "description": "Apache Log", "parser": "apache_access"}
result = parse_log(APACHE_SAMPLE.splitlines(), fmt)
check("Apache: endpoint metric available", result["applicable_fields"]["endpoints"]["status"] == "available")
check("Apache: user-agent metric available", result["applicable_fields"]["user_agents"]["status"] == "available")
check("Apache: data transfer is exact", result["summary"]["total_bytes"] == 12669,
      f"got {result['summary']['total_bytes']}")

# Java timeline/performance checks
fmt = {"format": "java_application", "confidence": 0.9, "description": "Java", "parser": "java_application"}
result = parse_log(JAVA_SAMPLE.splitlines(), fmt)
check("Java: timeline produced from timestamps", len(result["timeline"]) == 5)
check("Java: execution duration available", result["applicable_fields"]["execution_duration"]["status"] == "available")

# sample.log file on disk (if it exists)
sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.log")
if os.path.exists(sample_path):
    section("6. sample.log FILE — LIVE REGRESSION TEST")
    r = detect_log_format_from_file(sample_path)
    check("sample.log → oracle_data_loader (not apache)",
          r["format"] == "oracle_data_loader",
          f"got {r['format']} ({r['confidence']})")
    check("sample.log NOT classified as apache",
          r["format"] != "apache_access",
          f"got {r['format']}")
else:
    print("\n  [SKIP] sample.log not found — skipping live file regression test.")


# ============================================================================
# Summary
# ============================================================================

section("TEST SUMMARY")
passed = sum(1 for s, *_ in _results if s == PASS)
failed = sum(1 for s, *_ in _results if s == FAIL)
total  = len(_results)
print(f"\n  Total   : {total}")
print(f"  Passed  : {passed}")
print(f"  Failed  : {failed}")

if failed == 0:
    print(f"\n  🎉 All {total} tests passed!")
else:
    print(f"\n  ⚠️  {failed} test(s) failed:")
    for status, name, detail in _results:
        if status == FAIL:
            print(f"     {FAIL}  {name}  [{detail}]")
    sys.exit(1)

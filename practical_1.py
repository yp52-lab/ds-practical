import os
import re
import json
import random
import pandas as pd
from log_format_detector import detect_log_format_from_file

# ---------------------------------------------------------------------------
# PRACTICAL 1 — Load & Explore Unstructured Access Log Data
# Objective: Import raw server logs and understand their structure.
# ---------------------------------------------------------------------------

LOG_FILE_PATH = "Logs/logs/cj.log"


def load_and_explore_log(log_file_path=LOG_FILE_PATH, sample_size=1000):
    print("=" * 60)
    print("PRACTICAL 1: Load & Explore Unstructured Access Log Data")
    print("=" * 60)

    if not os.path.exists(log_file_path):
        print(f"[!] File not found: {log_file_path}")
        return

    # ------------------------------------------------------------------
    # STEP 0: Auto-detect log format BEFORE any parsing attempt
    # ------------------------------------------------------------------
    print(f"\n[Step 0] Auto-detecting log format for: '{log_file_path}'")
    fmt_info = detect_log_format_from_file(log_file_path)
    print(f"  → Detected Format : {fmt_info['description']}")
    print(f"  → Format ID       : {fmt_info['format']}")
    print(f"  → Confidence      : {fmt_info['confidence'] * 100:.0f}%")

    # ------------------------------------------------------------------
    # STEP 1: Load a sample of raw lines
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Loading up to {sample_size:,} raw lines with open()...")
    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = []
        for i, raw_line in enumerate(file):
            if i >= sample_size:
                break
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)

    print(f"  → Loaded {len(lines):,} non-empty lines.")

    # ------------------------------------------------------------------
    # STEP 2: Extract 7 random lines for manual inspection
    # ------------------------------------------------------------------
    print("\n[Step 2] Extracting 7 random lines for manual inspection...")
    random_lines = random.sample(lines, min(7, len(lines)))
    for idx, line in enumerate(random_lines, 1):
        print(f"\n  Line {idx}:")
        print(f"    {line[:200]}{'…' if len(line) > 200 else ''}")

    # ------------------------------------------------------------------
    # STEP 3: Understand structure — format-specific explanation
    # ------------------------------------------------------------------
    print("\n[Step 3] Analysing log structure...")

    fmt = fmt_info["format"]
    if fmt == "json_array":
        _explore_json_array(lines, random_lines)
    elif fmt in ("apache_access", "nginx_access"):
        _explore_apache(lines, random_lines)
    elif fmt == "oracle_data_loader":
        _explore_oracle(lines, random_lines)
    elif fmt == "java_application":
        _explore_java(lines, random_lines)
    else:
        _explore_generic(lines, random_lines)

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Format-specific exploration helpers
# ---------------------------------------------------------------------------

def _explore_json_array(lines, random_lines):
    """Explore cj.log JSON array format."""
    print("  Format: JSON Array (one record per line)")
    print("  Typical structure: [field0, field1, timestamp, ip, port, user_agent, lang, extra_ip]")

    COLUMNS = ["request_resource", "request_data", "date_time",
               "ip_address", "port", "user_agent", "language", "additional_ip"]

    regex_pattern = re.compile(
        r'\[.*?,.*?,\s*"(?P<Timestamp>[^"]+)",\s*"(?P<IP>[^"]+)",'
        r'\s*"(?P<Port>[^"]+)",\s*"(?P<UserAgent>[^"]+)"'
    )

    parsed_data = []
    for line in random_lines:
        try:
            record = json.loads(line)
            if isinstance(record, list) and len(record) >= 4:
                row = {COLUMNS[i]: record[i] for i in range(min(len(COLUMNS), len(record)))}
                parsed_data.append(row)
        except json.JSONDecodeError:
            m = regex_pattern.search(line)
            if m:
                parsed_data.append(m.groupdict())

    df = pd.DataFrame(parsed_data)
    print("\n--- Identified Fields (sample) ---")
    print(df.to_string(index=False))
    print("\nUseful fields for analysis:")
    print("  • date_time (timestamp)  → temporal analysis")
    print("  • ip_address             → source identification, threat detection")
    print("  • user_agent             → bot detection")
    print("  • request_resource       → URL / endpoint analysis")


def _explore_apache(lines, random_lines):
    """Explore Apache Combined Log Format."""
    print("  Format: Apache / Nginx Combined Log")
    print("  Typical structure: IP - USER [TIMESTAMP] \"METHOD URL PROTO\" STATUS BYTES \"REFERRER\" \"UA\"")

    pattern = re.compile(
        r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
        r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<bytes>\S+)'
        r'(?:\s+"(?P<referrer>[^"]*)"\s+"(?P<user_agent>[^"]*)")?'
    )

    parsed = []
    for line in random_lines:
        m = pattern.match(line)
        if m:
            parsed.append(m.groupdict())

    df = pd.DataFrame(parsed)
    print("\n--- Identified Fields (sample) ---")
    if not df.empty:
        print(df.to_string(index=False))
    print("\nUseful fields for analysis:")
    print("  • ip           → attacker identification, geolocation")
    print("  • timestamp    → time-series analysis")
    print("  • request      → HTTP method + URL + protocol version")
    print("  • status       → 404/403 for scanning, 200/302 for success")
    print("  • bytes        → data exfiltration detection")
    print("  • user_agent   → bot vs browser, tool identification")
    print("  • referrer     → social engineering / phishing chains")


def _explore_oracle(lines, random_lines):
    """Explore Oracle Data Loader / SOAP log."""
    print("  Format: Oracle Data Loader / SOAP Application Log")
    print("  Typical structure: HH:MM:SS LEVEL [thread] Component - Message")

    pattern = re.compile(
        r'^(?P<timestamp>\d{2}:\d{2}:\d{2})\s+'
        r'(?P<level>\w+)\s+'
        r'(?:\[(?P<thread>[^\]]+)\]\s+)?'
        r'(?P<component>\S+)\s+-\s+'
        r'(?P<message>.+)$'
    )

    parsed = []
    for line in random_lines:
        m = pattern.match(line)
        if m:
            parsed.append(m.groupdict())

    df = pd.DataFrame(parsed)
    print("\n--- Identified Fields (sample) ---")
    if not df.empty:
        print(df.to_string(index=False))
    print("\nUseful fields for analysis:")
    print("  • timestamp  → execution timeline reconstruction")
    print("  • level      → INFO / ERROR / WARN event severity")
    print("  • thread     → concurrency / multi-thread analysis")
    print("  • component  → BulkOpsClient, SoapClient, etc.")
    print("  • message    → SOAP results, validation status, import progress")

    print("\n  [!] Note: This log does NOT contain IP addresses, HTTP endpoints,")
    print("      or User-Agents. Security threat analysis is NOT applicable.")


def _explore_java(lines, random_lines):
    """Explore generic Java application log."""
    print("  Format: Java Application Log (log4j / logback)")
    print("  Typical structure: YYYY-MM-DD HH:MM:SS.mmm LEVEL component - message")

    pattern = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[.,]\d+)\s+'
        r'(?P<level>\w+)\s+(?P<source>[\w.$]+)\s+-\s+(?P<message>.+)$'
    )

    parsed = []
    for line in random_lines:
        m = pattern.match(line)
        if m:
            parsed.append(m.groupdict())

    df = pd.DataFrame(parsed)
    print("\n--- Identified Fields (sample) ---")
    if not df.empty:
        print(df.to_string(index=False))
    print("\nUseful fields for analysis:")
    print("  • timestamp → event timing, performance analysis")
    print("  • level     → DEBUG / INFO / WARN / ERROR / FATAL")
    print("  • source    → class / component name")
    print("  • message   → event description, exceptions")


def _explore_generic(lines, random_lines):
    """Fallback for unrecognised formats."""
    print("  Format: Generic / Unrecognised")
    print("  Raw lines captured. Manual inspection required.")
    for i, ln in enumerate(random_lines[:5], 1):
        print(f"    [{i}] {ln[:200]}")


if __name__ == "__main__":
    load_and_explore_log(LOG_FILE_PATH)

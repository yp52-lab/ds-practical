"""Content-based log-format detection.

Detection deliberately uses line structure, never filenames. Apache and Nginx
combined access logs are structurally compatible; a file is only called Nginx
when it carries an Nginx-specific signal. Otherwise it is labelled
Apache-compatible rather than pretending that the producer is known.
"""

import re
from typing import Any, Dict, List

_HTTP = re.compile(
    r'^\S+\s+\S+\s+\S+\s+\[\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4}\]\s+"[A-Z]+\s+\S+\s+HTTP/\d(?:\.\d)?"\s+\d{3}\s+(?:\d+|-)' 
)
_JSON_ARRAY = re.compile(r'^\s*\[(?:null|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)(?:\s*,\s*(?:null|"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?)){3,}\s*\]\s*$')
_JAVA = re.compile(
    r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s+'
    r'(?:\[[^\]]+\]\s+)?(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|SEVERE)\s+'
    r'(?:\[[^\]]+\]\s+)?[\w.$-]+\s*(?:[-–:]\s*.+)?$', re.I
)
_APPLICATION = re.compile(
    r'^(?:\[?\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\]?|\d{2}:\d{2}:\d{2})\s+'
    r'.*\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|SEVERE)\b', re.I
)
_ORACLE_SIGNALS = ("bulkopsclient", "sbl-odu-", "soapenv:", "siebelmessage")


def _result(fmt: str, confidence: float, description: str, parser: str | None = None,
            components: List[str] | None = None) -> Dict[str, Any]:
    return {
        "format": fmt,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "description": description,
        "parser": parser or fmt,
        "components": components or [fmt],
    }


def _line_kinds(lines: List[str]) -> Dict[str, int]:
    kinds = {"apache_access": 0, "json_array": 0, "java_application": 0,
             "application_log": 0, "oracle_data_loader": 0}
    for line in lines:
        low = line.lower()
        if _HTTP.match(line):
            kinds["apache_access"] += 1
        elif _JSON_ARRAY.match(line):
            kinds["json_array"] += 1
        elif any(signal in low for signal in _ORACLE_SIGNALS):
            kinds["oracle_data_loader"] += 1
        elif _JAVA.match(line):
            kinds["java_application"] += 1
        elif _APPLICATION.match(line):
            kinds["application_log"] += 1
    return kinds


def detect_log_format(file_content: str, sample_lines: int = 200) -> Dict[str, Any]:
    """Detect a format from actual content and report uncertainty explicitly."""
    lines = [line.strip() for line in file_content.splitlines() if line.strip()][:sample_lines]
    if not lines:
        return _result("empty", 1.0, "Empty log file", "empty")

    kinds = _line_kinds(lines)
    recognized = {kind: count for kind, count in kinds.items() if count}
    minimum_component_lines = 1 if len(lines) <= 10 else max(2, int(len(lines) * 0.10))
    meaningful = {kind: count for kind, count in recognized.items()
                  if count >= minimum_component_lines}
    if len(meaningful) >= 2:
        components = sorted(meaningful, key=meaningful.get, reverse=True)
        coverage = sum(meaningful.values()) / len(lines)
        return _result("mixed", min(0.95, 0.55 + coverage * 0.4),
                       "Mixed-format log", "mixed", components)

    if not recognized:
        http_like = sum(1 for line in lines if 'HTTP/' in line or ('[' in line and ']' in line and '"' in line))
        if http_like >= max(1, len(lines) // 3):
            return _result("malformed", 0.75, "Malformed or partially structured log", "generic")
        return _result("generic", 0.35, "Generic / unrecognised log format", "generic")

    best, hits = max(recognized.items(), key=lambda item: item[1])
    coverage = hits / len(lines)
    if best == "apache_access":
        text = "\n".join(lines).lower()
        if "nginx/" in text or "nginx:" in text:
            best, description = "nginx_access", "Nginx HTTP Access Log"
        else:
            description = "Apache-compatible HTTP Access Log"
    else:
        description = {
            "json_array": "JSON Array Request Log",
            "java_application": "Java Application Log",
            "application_log": "Application Log",
            "oracle_data_loader": "Oracle Data Loader / SOAP Application Log",
        }[best]
    return _result(best, min(0.99, 0.5 + coverage * 0.49), description, best)


def detect_log_format_from_file(filepath: str, read_bytes: int = 65536) -> Dict[str, Any]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            return detect_log_format(handle.read(read_bytes))
    except OSError as exc:
        return _result("unreadable", 1.0, f"Could not read file: {exc}", "generic")


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python log_format_detector.py <logfile>")
    print(json.dumps(detect_log_format_from_file(sys.argv[1]), indent=2))

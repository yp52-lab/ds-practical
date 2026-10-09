"""Single-command LogSentinel server.

Run ``python server.py``. It builds the React frontend and serves both the UI
and analysis API at http://localhost:8000, so no separate frontend port or
second terminal is needed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIRS = [
    BASE_DIR / "logsentinel-ui",
    BASE_DIR / "home" / "yug" / "My_Practice" / "ds lab" / "logsentinel-ui",
]


def _frontend_dir() -> Path:
    directory = next((path for path in FRONTEND_DIRS if (path / "package.json").exists()), None)
    if directory is None:
        raise FileNotFoundError("React frontend was not found.")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the unified LogSentinel web application.")
    parser.add_argument("--no-build", action="store_true", help="Serve the current React build without rebuilding it.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    frontend = _frontend_dir()
    if not args.no_build:
        print("Building the React interface...")
        subprocess.run(["npm", "run", "build"], cwd=frontend, check=True)

    # Import only after the build exists: app_ui discovers the generated files.
    from app_ui import app
    import uvicorn

    address = f"http://{args.host}:{args.port}"
    print(f"\nLogSentinel is running at {address}")
    print("The React chat UI and log-analysis API use this one address.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Frontend build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

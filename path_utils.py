"""
path_utils.py — Shared path resolution utilities for DS Lab practicals (7–10).

Ensures every practical can locate input/output files correctly whether run
from the project root or from any working directory.
"""

import os

# ── Base directory is the folder containing this file ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_input_file(filename: str) -> str:
    """
    Resolve a relative or absolute file path to an absolute path.

    Search order:
      1. Absolute path as given (if it already is one and exists).
      2. Relative to BASE_DIR (the project root, same folder as this file).
      3. Current working directory.

    Parameters
    ----------
    filename : str
        A filename or relative/absolute path to resolve.

    Returns
    -------
    str
        The best-guess absolute path (may or may not exist).
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    candidate_base = os.path.join(BASE_DIR, filename)
    if os.path.exists(candidate_base):
        return candidate_base

    candidate_cwd = os.path.join(os.getcwd(), filename)
    if os.path.exists(candidate_cwd):
        return candidate_cwd

    # Fall back to BASE_DIR resolution even if not found yet
    return candidate_base


def resolve_export_file(filename: str) -> str:
    """
    Resolve an output/export file path.

    If the path is already absolute it is returned as-is; otherwise it is
    placed in BASE_DIR so all pipeline outputs land in the project folder.

    Parameters
    ----------
    filename : str
        A filename or path for the file to be written.

    Returns
    -------
    str
        Absolute path where the file will be written.
    """
    if os.path.isabs(filename):
        return filename
    return os.path.join(BASE_DIR, filename)

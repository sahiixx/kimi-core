"""Filesystem tools for Kimi-Core."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{path}' not found"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def glob_files(pattern: str, path: str = ".") -> list[str]:
    """Find files matching a glob pattern."""
    matches = []
    for root, _, files in os.walk(path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matches.append(os.path.join(root, filename))
    return matches[:50]


def grep_files(pattern: str, path: str = ".") -> list[str]:
    """Search for regex pattern in files under path."""
    matches = []
    compiled = re.compile(pattern)
    for root, _, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if compiled.search(line):
                            matches.append(f"{filepath}:{i}: {line.strip()}")
                            if len(matches) >= 50:
                                return matches
            except Exception:
                continue
    return matches

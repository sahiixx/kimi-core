"""Shell execution tool for Kimi-Core."""

from __future__ import annotations

import subprocess


def run_shell(command: str, timeout: float = 60.0) -> str:
    """Run a shell command and capture output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"

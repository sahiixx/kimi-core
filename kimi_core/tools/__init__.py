"""Tool registry and definitions."""

from kimi_core.tools.fs import glob_files, grep_files, read_file, write_file
from kimi_core.tools.shell import run_shell

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates directories if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "glob_files",
        "description": "Find files matching a glob pattern under a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'"},
                "path": {"type": "string", "description": "Directory to search", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search for a regex pattern in files under a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {"type": "string", "description": "Directory to search", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": "Run a shell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds", "default": 60},
            },
            "required": ["command"],
        },
    },
]

TOOL_HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "glob_files": glob_files,
    "grep_files": grep_files,
    "run_shell": run_shell,
}

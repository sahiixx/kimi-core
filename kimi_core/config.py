"""Configuration loader for Kimi-Core."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Config:
    agent_name: str = "Kimi Code CLI"
    version: str = "0.1.0"
    llm_backend: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:14b"
    max_context_tokens: int = 128000
    working_memory_items: int = 10
    parallel_tool_calls: bool = True
    confirm_destructive_actions: bool = True
    max_command_timeout_seconds: int = 86400
    memory_dir: str = field(default_factory=lambda: str(Path.home() / ".kimi" / "memory"))
    reflection_log_path: str = field(default_factory=lambda: str(Path.home() / ".kimi" / "reflections.jsonl"))


def _default_config() -> dict[str, Any]:
    return {
        "agent_name": "Kimi Code CLI",
        "version": "0.1.0",
        "llm_backend": "ollama",
        "ollama_host": "http://localhost:11434",
        "ollama_model": "qwen2.5-coder:14b",
        "max_context_tokens": 128000,
        "working_memory_items": 10,
        "parallel_tool_calls": True,
        "confirm_destructive_actions": True,
        "max_command_timeout_seconds": 86400,
    }


def load_config(path: str | None = None) -> Config:
    """Load configuration from YAML file or return defaults."""
    defaults = _default_config()

    if path is None:
        search_paths = [
            Path.cwd() / "agent-config.yaml",
            Path.home() / ".kimi" / "agent-config.yaml",
            Path.home() / ".config" / "kimi" / "agent-config.yaml",
        ]
        for p in search_paths:
            if p.exists():
                path = str(p)
                break

    if path and yaml is not None and Path(path).exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        defaults.update(data)

    return Config(
        agent_name=defaults.get("agent_name", "Kimi Code CLI"),
        version=defaults.get("version", "0.1.0"),
        llm_backend=defaults.get("llm_backend", "ollama"),
        ollama_host=defaults.get("ollama_host", "http://localhost:11434"),
        ollama_model=defaults.get("ollama_model", "qwen2.5-coder:14b"),
        max_context_tokens=defaults.get("max_context_tokens", 128000),
        working_memory_items=defaults.get("working_memory_items", 10),
        parallel_tool_calls=defaults.get("parallel_tool_calls", True),
        confirm_destructive_actions=defaults.get("confirm_destructive_actions", True),
        max_command_timeout_seconds=defaults.get("max_command_timeout_seconds", 86400),
    )

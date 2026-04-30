"""Persistent memory for Kimi-Core."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PersistentMemory:
    conversation: list[dict] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    file_cache: dict[str, str] = field(default_factory=dict)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    memory_dir: str = field(default_factory=lambda: str(Path.home() / ".kimi" / "memory"))

    def add_interaction(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def add_tool_output(self, tool_name: str, arguments: dict, result: Any) -> None:
        self.tool_outputs.append({"tool": tool_name, "args": arguments, "result": result})

    def save(self) -> None:
        d = Path(self.memory_dir)
        d.mkdir(parents=True, exist_ok=True)
        data = {
            "conversation": self.conversation,
            "tool_outputs": self.tool_outputs,
            "file_cache": self.file_cache,
            "user_preferences": self.user_preferences,
        }
        with open(d / "session.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> None:
        path = Path(self.memory_dir) / "session.json"
        if not path.exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.conversation = data.get("conversation", [])
        self.tool_outputs = data.get("tool_outputs", [])
        self.file_cache = data.get("file_cache", {})
        self.user_preferences = data.get("user_preferences", {})

    def get_recent_tools(self, n: int = 5) -> list[dict]:
        return self.tool_outputs[-n:]

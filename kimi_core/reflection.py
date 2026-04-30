"""Reflection system for Kimi-Core."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any


class DecisionOutcome(Enum):
    SUCCESS = auto()
    PARTIAL = auto()
    FAILURE = auto()
    UNKNOWN = auto()


@dataclass
class DecisionRecord:
    timestamp: str
    task_type: str
    strategy: str
    confidence: str
    input_summary: str
    tools_used: list[str]
    outcome: str
    duration_ms: float
    error: str | None = None
    lesson: str | None = None


class ReflectionLog:
    def __init__(self, log_path: str | None = None) -> None:
        if log_path is None:
            log_path = str(Path.home() / ".kimi" / "reflections.jsonl")
        self.log_path = Path(log_path)
        self._entries: list[DecisionRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._entries.append(DecisionRecord(**data))
        except (json.JSONDecodeError, TypeError):
            pass

    def append(self, record: DecisionRecord) -> None:
        self._entries.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def get_stats(self) -> dict[str, Any]:
        if not self._entries:
            return {"total": 0}
        total = len(self._entries)
        outcomes = {}
        for entry in self._entries:
            outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1
        success_rate = outcomes.get("SUCCESS", 0) / total if total > 0 else 0
        return {
            "total": total,
            "success_rate": round(success_rate, 2),
            "outcome_distribution": outcomes,
        }


class DecisionTracker:
    def __init__(self, log: ReflectionLog | None = None) -> None:
        self.log = log or ReflectionLog()
        self._active: dict[str, float] = {}

    def start_decision(self, decision_id: str) -> None:
        self._active[decision_id] = time.time()

    def record_decision(
        self,
        decision_id: str,
        task_type: str,
        strategy: str,
        confidence: str,
        input_summary: str,
        tools_used: list[str],
        outcome: DecisionOutcome,
        error: str | None = None,
        lesson: str | None = None,
    ) -> DecisionRecord:
        start = self._active.pop(decision_id, time.time())
        duration_ms = (time.time() - start) * 1000
        record = DecisionRecord(
            timestamp=datetime.now().isoformat(),
            task_type=task_type,
            strategy=strategy,
            confidence=confidence,
            input_summary=input_summary[:200],
            tools_used=tools_used,
            outcome=outcome.name,
            duration_ms=round(duration_ms, 2),
            error=error,
            lesson=lesson,
        )
        self.log.append(record)
        return record

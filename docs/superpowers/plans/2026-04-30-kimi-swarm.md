# Kimi-Swarm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a meta-spawner that orchestrates multiple `KimiCore` agents for parallel and decomposed task execution.

**Architecture:** `Swarm` creates `KimiCore` instances with role-specific prompts. `run_parallel` uses `asyncio.gather` over thread-pool-spawned agents. `run_decompose` chains planner → parser → workers → aggregator, with fallback on parse failure.

**Tech Stack:** Python 3.11+, `asyncio`, reuses kimi-core `KimiCore`, `PersistentMemory`, `Config`.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `kimi_core/swarm.py` | `AgentRole` dataclass, `Swarm` orchestrator, default roles |
| `tests/test_swarm.py` | Unit tests: spawn, parallel, decompose, fallback |

---

## Task 1: Swarm Module + AgentRole

**Files:**
- Create: `kimi_core/swarm.py`
- Test: `tests/test_swarm.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_swarm.py`:

```python
import pytest
from kimi_core.swarm import AgentRole, Swarm


def test_agent_role_defaults():
    role = AgentRole(name="planner")
    assert role.name == "planner"
    assert role.system_prompt == ""
    assert role.model is None
    assert role.tools_subset is None
    assert role.max_rounds == 10


def test_swarm_spawn():
    swarm = Swarm()
    role = AgentRole(name="test", system_prompt="You are a test agent.")
    result = swarm.spawn(role, "say hello")
    assert isinstance(result, str)
```

Run:
```bash
pytest tests/test_swarm.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'kimi_core.swarm'`

- [ ] **Step 2: Implement `kimi_core/swarm.py`**

```python
"""Swarm orchestrator for multi-agent parallel execution."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from kimi_core.agent import KimiCore
from kimi_core.config import Config, load_config
from kimi_core.llm.ollama import OllamaProvider
from kimi_core.memory import PersistentMemory


@dataclass
class AgentRole:
    name: str
    system_prompt: str = ""
    model: str | None = None
    tools_subset: list[str] | None = None
    max_rounds: int = 10


PLANNER_ROLE = AgentRole(
    name="planner",
    system_prompt="Break this task into numbered subtasks. Each subtask should be independently executable. Return ONLY a numbered list.",
)

WORKER_ROLE = AgentRole(
    name="worker",
    system_prompt="You are a worker agent. Execute the given subtask using available tools. Return concise results.",
)

AGGREGATOR_ROLE = AgentRole(
    name="aggregator",
    system_prompt="Summarize these worker results into a coherent final answer.",
)


def _parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s+(.*)$", line.strip())
        if match:
            items.append(match.group(1).strip())
    return items


class Swarm:
    def __init__(self, config: Config | None = None, shared_memory: PersistentMemory | None = None) -> None:
        self.config = config or load_config()
        self.memory = shared_memory or PersistentMemory(memory_dir=self.config.memory_dir)
        self.memory.load()

    def spawn(self, role: AgentRole, task: str) -> str:
        provider = OllamaProvider(
            host=self.config.ollama_host,
            model=role.model or self.config.ollama_model,
        )
        agent = KimiCore(provider=provider, config=self.config)
        agent.memory = self.memory
        if role.system_prompt:
            agent.memory.add_interaction("system", role.system_prompt)
        return agent.run(task)

    async def spawn_async(self, role: AgentRole, task: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.spawn, role, task)

    def run_parallel(self, tasks: list[tuple[AgentRole, str]]) -> list[str]:
        return asyncio.run(self._run_parallel_async(tasks))

    async def _run_parallel_async(self, tasks: list[tuple[AgentRole, str]]) -> list[str]:
        coros = [self.spawn_async(role, task) for role, task in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)

    def run_decompose(
        self,
        task: str,
        planner_role: AgentRole | None = None,
        worker_role: AgentRole | None = None,
        aggregator_role: AgentRole | None = None,
    ) -> str:
        planner = planner_role or PLANNER_ROLE
        worker = worker_role or WORKER_ROLE
        aggregator = aggregator_role or AGGREGATOR_ROLE

        plan = self.spawn(planner, f"Break this task into subtasks:\n{task}")
        subtasks = _parse_numbered_list(plan)

        if not subtasks:
            return self.spawn(worker, task)

        results = self.run_parallel([(worker, st) for st in subtasks])

        combined = "\n\n".join(
            f"Worker {i+1} result:\n{r}" for i, r in enumerate(results)
        )
        final = self.spawn(aggregator, f"Summarize these results into a final answer for: {task}\n\n{combined}")
        return final
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_swarm.py -v
```
Expected: PASS (spawn returns string, role defaults correct)

- [ ] **Step 4: Commit**

```bash
git add kimi_core/swarm.py tests/test_swarm.py
git commit -m "feat: swarm orchestrator with AgentRole, spawn, and parallel execution"
```

---

## Task 2: Decompose + Fallback Tests

**Files:**
- Modify: `tests/test_swarm.py`

- [ ] **Step 1: Write failing tests for decompose**

Append to `tests/test_swarm.py`:

```python
from unittest.mock import MagicMock, patch


def test_decompose_happy_path():
    swarm = Swarm()
    with patch.object(swarm, "spawn") as mock_spawn:
        mock_spawn.side_effect = [
            "1. Read README\n2. List files",   # planner
            "readme content",                   # worker 1
            "file list",                        # worker 2
            "final answer",                     # aggregator
        ]
        result = swarm.run_decompose("analyze repo")
        assert result == "final answer"
        assert mock_spawn.call_count == 4


def test_decompose_fallback_on_bad_plan():
    swarm = Swarm()
    with patch.object(swarm, "spawn") as mock_spawn:
        mock_spawn.side_effect = [
            "garbage text with no numbers",   # planner fails
            "fallback answer",                # single agent fallback
        ]
        result = swarm.run_decompose("analyze repo")
        assert result == "fallback answer"
        assert mock_spawn.call_count == 2
```

Run:
```bash
pytest tests/test_swarm.py::test_decompose_happy_path -v
```
Expected: FAIL — tests don't exist yet

- [ ] **Step 2: Verify decompose logic already works**

The `run_decompose` implementation from Task 1 already handles fallback (empty subtasks → single agent). No code changes needed.

Run:
```bash
pytest tests/test_swarm.py -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_swarm.py
git commit -m "test: decompose happy path and fallback scenarios"
```

---

## Task 3: Parallel Execution Test + Edge Cases

**Files:**
- Modify: `tests/test_swarm.py`

- [ ] **Step 1: Write parallel execution test**

Append to `tests/test_swarm.py`:

```python
import asyncio


def test_run_parallel():
    swarm = Swarm()
    with patch.object(swarm, "spawn_async") as mock_spawn:
        async def side_effect(role, task):
            return f"result-{task}"
        mock_spawn.side_effect = side_effect
        results = swarm.run_parallel([
            (AgentRole(name="w1"), "task1"),
            (AgentRole(name="w2"), "task2"),
        ])
        assert results == ["result-task1", "result-task2"]
```

Run:
```bash
pytest tests/test_swarm.py::test_run_parallel -v
```
Expected: PASS

- [ ] **Step 2: Add error handling test**

Append to `tests/test_swarm.py`:

```python
def test_run_parallel_with_exception():
    swarm = Swarm()
    with patch.object(swarm, "spawn_async") as mock_spawn:
        async def side_effect(role, task):
            if task == "bad":
                raise RuntimeError("boom")
            return f"ok-{task}"
        mock_spawn.side_effect = side_effect
        results = swarm.run_parallel([
            (AgentRole(name="w1"), "good"),
            (AgentRole(name="w2"), "bad"),
        ])
        assert results[0] == "ok-good"
        assert isinstance(results[1], RuntimeError)
```

Run:
```bash
pytest tests/test_swarm.py::test_run_parallel_with_exception -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_swarm.py
git commit -m "test: parallel execution and error propagation"
```

---

## Task 4: Integration & Full Test Suite

**Files:**
- Modify: `tests/test_swarm.py` (if needed)

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```
Expected: ALL PASS (previous tests + new swarm tests)

- [ ] **Step 2: Import check**

```bash
python3 -c "from kimi_core.swarm import Swarm, AgentRole, PLANNER_ROLE, WORKER_ROLE, AGGREGATOR_ROLE; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: full suite passing, swarm integration verified"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| AgentRole dataclass | Task 1 |
| Swarm.spawn sync | Task 1 |
| Swarm.spawn_async | Task 1 |
| run_parallel with asyncio.gather | Task 1 |
| run_decompose planner→workers→aggregator | Task 1 |
| Fallback on bad plan | Task 2 |
| Error propagation in parallel | Task 3 |
| Shared PersistentMemory | Task 1 |

No gaps found.

## Placeholder Scan

- No "TBD", "TODO", "implement later" found.
- All steps contain exact code and exact commands.
- Type names consistent with kimi-core (`KimiCore`, `PersistentMemory`, `Config`, `OllamaProvider`).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-30-kimi-swarm.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

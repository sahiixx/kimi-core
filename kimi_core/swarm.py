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

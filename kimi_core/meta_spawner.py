"""Meta-spawner: multi-agent orchestrator with A2A integration.

Builds on swarm.py to add:
- Agent registry with health checks and auto-restart
- Priority task queue with capability-based routing
- Hybrid local swarm + distributed A2A execution
- Event logging for observability
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from kimi_core.agent import KimiCore
from kimi_core.config import Config, load_config
from kimi_core.llm.ollama import OllamaProvider
from kimi_core.memory import PersistentMemory
from kimi_core.swarm import AgentRole, Swarm, PLANNER_ROLE, WORKER_ROLE, AGGREGATOR_ROLE


class AgentStatus(Enum):
    SPAWNING = auto()
    HEALTHY = auto()
    BUSY = auto()
    UNRESPONSIVE = auto()
    DEAD = auto()


class TaskStatus(Enum):
    QUEUED = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class AgentInstance:
    instance_id: str
    role: AgentRole
    config: Config
    port: int | None = None
    status: AgentStatus = AgentStatus.SPAWNING
    last_heartbeat: float = field(default_factory=time.time)
    spawn_time: float = field(default_factory=time.time)
    total_tasks: int = 0
    failed_tasks: int = 0
    capabilities: list[str] = field(default_factory=list)
    _agent: KimiCore | None = field(default=None, repr=False)


@dataclass(order=True)
class TaskItem:
    priority: int
    task_id: str = field(compare=False)
    task_text: str = field(compare=False)
    required_capabilities: list[str] = field(default_factory=list, compare=False)
    assigned_agent: str | None = field(default=None, compare=False)
    status: TaskStatus = field(default=TaskStatus.QUEUED, compare=False)
    result: Any = field(default=None, compare=False)
    error: str | None = field(default=None, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: float | None = field(default=None, compare=False)
    completed_at: float | None = field(default=None, compare=False)


@dataclass
class EventLog:
    event_type: str
    timestamp: str
    details: dict[str, Any]


class CapabilityRegistry:
    """Maps task keywords to best agent roles."""

    def __init__(self) -> None:
        self._roles: dict[str, AgentRole] = {}
        self._capabilities: dict[str, list[str]] = {}

    def register(self, name: str, role: AgentRole, capabilities: list[str]) -> None:
        self._roles[name] = role
        self._capabilities[name] = capabilities

    def find_best_role(self, required: list[str]) -> AgentRole | None:
        best_name = None
        best_score = 0
        for name, caps in self._capabilities.items():
            score = sum(1 for r in required if r in caps)
            if score > best_score:
                best_score = score
                best_name = name
        return self._roles.get(best_name) if best_name else None

    def list_roles(self) -> list[tuple[str, AgentRole, list[str]]]:
        return [(n, self._roles[n], self._capabilities[n]) for n in self._roles]


class A2ABridge:
    """Lightweight A2A client for external agent discovery and task routing.
    Gracefully degrades if agency-agents a2a_protocol is unavailable."""

    def __init__(self) -> None:
        self._client_class: Any = None
        self._make_tools: Callable | None = None
        self._discovered: dict[str, dict] = {}
        self._load_a2a()

    def _load_a2a(self) -> None:
        try:
            import sys
            sys.path.insert(0, "/home/sahiix/agency-agents")
            from a2a_protocol import A2AClient, make_a2a_tools  # type: ignore
            self._client_class = A2AClient
            self._make_tools = make_a2a_tools
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._client_class is not None

    def discover(self, url: str) -> dict:
        if not self._client_class:
            return {}
        try:
            client = self._client_class(url)
            card = client.discover()
            self._discovered[url] = card
            return card
        except Exception as e:
            return {"error": str(e)}

    def send_task(self, url: str, text: str) -> str:
        if not self._client_class:
            return f"A2A unavailable: cannot send task to {url}"
        try:
            client = self._client_class(url)
            task = client.send_task(text)
            return client.get_result_text(task)
        except Exception as e:
            return f"A2A error: {e}"

    def get_tools(self, urls: list[str]) -> list:
        if not self._make_tools:
            return []
        try:
            return self._make_tools(urls)
        except Exception:
            return []


class MetaSpawner:
    """Orchestrates multiple Kimi-Core agents with health checks,
    task routing, and A2A integration."""

    def __init__(
        self,
        config: Config | None = None,
        shared_memory: PersistentMemory | None = None,
        max_workers: int = 10,
        health_check_interval: float = 30.0,
        task_timeout: float = 300.0,
    ) -> None:
        self.config = config or load_config()
        self.memory = shared_memory or PersistentMemory(memory_dir=self.config.memory_dir)
        self.memory.load()
        self.swarm = Swarm(config=self.config, shared_memory=self.memory)
        self.registry = CapabilityRegistry()
        self.a2a = A2ABridge()

        self._agents: dict[str, AgentInstance] = {}
        self._task_queue: deque[TaskItem] = deque()
        self._completed: dict[str, TaskItem] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._health_interval = health_check_interval
        self._task_timeout = task_timeout
        self._events: list[EventLog] = []
        self._shutdown = False
        self._lock = threading.RLock()
        self._base_port = 9000

        self._default_roles()

    def _default_roles(self) -> None:
        self.registry.register("planner", PLANNER_ROLE, ["planning", "decomposition"])
        self.registry.register("worker", WORKER_ROLE, ["execution", "coding", "analysis"])
        self.registry.register("aggregator", AGGREGATOR_ROLE, ["synthesis", "summary"])

    def _log(self, event_type: str, **details: Any) -> None:
        self._events.append(EventLog(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
        ))

    def spawn(
        self,
        role: AgentRole | str,
        instance_id: str | None = None,
        port: int | None = None,
    ) -> AgentInstance:
        """Spawn a new agent instance."""
        if isinstance(role, str):
            role_entry = self.registry.find_best_role([role])
            if not role_entry:
                raise ValueError(f"No role found for capability: {role}")
            role = role_entry

        iid = instance_id or f"agent_{uuid.uuid4().hex[:8]}"

        provider = OllamaProvider(
            host=self.config.ollama_host,
            model=role.model or self.config.ollama_model,
        )
        agent = KimiCore(provider=provider, config=self.config)
        agent.memory = self.memory
        if role.system_prompt:
            agent.memory.add_interaction("system", role.system_prompt)

        instance = AgentInstance(
            instance_id=iid,
            role=role,
            config=self.config,
            port=port or self._next_port(),
            status=AgentStatus.HEALTHY,
            _agent=agent,
        )

        with self._lock:
            self._agents[iid] = instance

        self._log("agent_spawn", instance_id=iid, role=role.name, port=instance.port)
        return instance

    def _next_port(self) -> int:
        with self._lock:
            self._base_port += 1
            return self._base_port

    def kill(self, instance_id: str) -> None:
        """Terminate an agent instance."""
        with self._lock:
            inst = self._agents.pop(instance_id, None)
        if inst:
            inst.status = AgentStatus.DEAD
            self._log("agent_kill", instance_id=instance_id)

    def health_check(self, instance_id: str) -> bool:
        """Check if agent is responsive."""
        with self._lock:
            inst = self._agents.get(instance_id)
        if not inst or not inst._agent:
            return False
        try:
            # Lightweight ping: ask agent to echo
            result = inst._agent.run("echo heartbeat")
            healthy = "heartbeat" in result.lower() or len(result) > 0
            inst.last_heartbeat = time.time()
            inst.status = AgentStatus.HEALTHY if healthy else AgentStatus.UNRESPONSIVE
            self._log("health_check", instance_id=instance_id, healthy=healthy)
            return healthy
        except Exception as e:
            inst.status = AgentStatus.UNRESPONSIVE
            self._log("health_check_fail", instance_id=instance_id, error=str(e))
            return False

    def health_check_all(self) -> dict[str, bool]:
        """Check all agents."""
        results = {}
        with self._lock:
            agents = list(self._agents.keys())
        for aid in agents:
            results[aid] = self.health_check(aid)
        return results

    def restart_unhealthy(self) -> list[str]:
        """Restart any unresponsive agents. Returns restarted IDs."""
        restarted = []
        with self._lock:
            for iid, inst in list(self._agents.items()):
                if inst.status in (AgentStatus.UNRESPONSIVE, AgentStatus.DEAD):
                    self._log("agent_restart", instance_id=iid, role=inst.role.name)
                    self._agents.pop(iid, None)
                    new_inst = self.spawn(inst.role, instance_id=iid)
                    restarted.append(new_inst.instance_id)
        return restarted

    def submit_task(
        self,
        task_text: str,
        priority: int = 5,
        required_capabilities: list[str] | None = None,
    ) -> str:
        """Queue a task. Returns task_id."""
        task = TaskItem(
            priority=priority,
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_text=task_text,
            required_capabilities=required_capabilities or [],
        )
        with self._lock:
            self._task_queue.append(task)
        self._log("task_submit", task_id=task.task_id, priority=priority)
        return task.task_id

    def route_tasks(self) -> list[str]:
        """Assign queued tasks to available agents. Returns assigned task IDs."""
        assigned: list[str] = []
        with self._lock:
            available = [
                (iid, inst) for iid, inst in self._agents.items()
                if inst.status == AgentStatus.HEALTHY
            ]
            if not available or not self._task_queue:
                return assigned

            # Sort queue by priority (lowest number = highest priority)
            sorted_tasks = sorted(self._task_queue, key=lambda t: t.priority)
            remaining = deque()

            for task in sorted_tasks:
                matched = False
                for iid, inst in available:
                    if inst.status != AgentStatus.HEALTHY:
                        continue
                    # Capability matching
                    if task.required_capabilities:
                        role_caps = self.registry._capabilities.get(inst.role.name, [])
                        if not any(c in role_caps for c in task.required_capabilities):
                            continue

                    task.assigned_agent = iid
                    task.status = TaskStatus.ASSIGNED
                    assigned.append(task.task_id)
                    inst.status = AgentStatus.BUSY
                    matched = True
                    break

                if not matched:
                    remaining.append(task)

            self._task_queue = remaining

        for tid in assigned:
            self._log("task_assigned", task_id=tid)
        return assigned

    def run_task(self, task_id: str) -> str:
        """Execute an assigned task and return result."""
        with self._lock:
            task = None
            for t in list(self._task_queue):
                if t.task_id == task_id:
                    task = t
                    break
            # Also check if it's already assigned but not running
            if not task:
                for t in self._completed.values():
                    if t.task_id == task_id:
                        return t.result or t.error or ""

            inst = self._agents.get(task.assigned_agent) if task else None

        if not task or not inst:
            raise ValueError(f"Task {task_id} not found or not assigned")

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._log("task_start", task_id=task_id, agent=inst.instance_id)

        try:
            result = inst._agent.run(task.task_text)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            inst.total_tasks += 1
            inst.status = AgentStatus.HEALTHY
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            inst.failed_tasks += 1
            inst.status = AgentStatus.UNRESPONSIVE
            self._log("task_fail", task_id=task_id, error=str(e))

        with self._lock:
            self._completed[task.task_id] = task

        return task.result or task.error or ""

    def run_parallel(self, tasks: list[tuple[AgentRole, str]]) -> list[str]:
        """Spawn agents for each task and run in parallel."""
        instances = []
        for role, text in tasks:
            inst = self.spawn(role)
            instances.append((inst, text))

        def _run(inst: AgentInstance, text: str) -> str:
            try:
                return inst._agent.run(text)
            except Exception as e:
                return f"Error: {e}"
            finally:
                inst.status = AgentStatus.HEALTHY

        futures = [self._executor.submit(_run, inst, text) for inst, text in instances]
        results = [f.result(timeout=self._task_timeout) for f in futures]

        for inst, _ in instances:
            inst.status = AgentStatus.HEALTHY

        return results

    def run_decompose(self, task_text: str) -> str:
        """Planner → Workers → Aggregator with full orchestration."""
        planner = self.spawn(PLANNER_ROLE)
        plan = planner._agent.run(f"Break this task into subtasks:\n{task_text}")

        from kimi_core.swarm import _parse_numbered_list
        subtasks = _parse_numbered_list(plan)

        if not subtasks:
            worker = self.spawn(WORKER_ROLE)
            return worker._agent.run(task_text)

        tasks = [(WORKER_ROLE, st) for st in subtasks]
        results = self.run_parallel(tasks)

        combined = "\n\n".join(
            f"Worker {i+1} result:\n{r}" for i, r in enumerate(results)
        )
        aggregator = self.spawn(AGGREGATOR_ROLE)
        final = aggregator._agent.run(
            f"Summarize these results into a final answer for: {task_text}\n\n{combined}"
        )
        return final

    def a2a_discover(self, url: str) -> dict:
        """Discover an external A2A agent."""
        result = self.a2a.discover(url)
        self._log("a2a_discover", url=url, result=result)
        return result

    def a2a_route_task(self, url: str, task_text: str) -> str:
        """Send a task to an external A2A agent."""
        self._log("a2a_route", url=url, task_preview=task_text[:100])
        return self.a2a.send_task(url, task_text)

    def get_agent_status(self) -> dict[str, dict]:
        """Get status snapshot of all agents."""
        with self._lock:
            return {
                iid: {
                    "role": inst.role.name,
                    "status": inst.status.name,
                    "port": inst.port,
                    "total_tasks": inst.total_tasks,
                    "failed_tasks": inst.failed_tasks,
                    "uptime": time.time() - inst.spawn_time,
                    "last_heartbeat": inst.last_heartbeat,
                }
                for iid, inst in self._agents.items()
            }

    def get_queue_status(self) -> dict:
        """Get task queue snapshot."""
        with self._lock:
            return {
                "queued": len(self._task_queue),
                "completed": len(self._completed),
                "agents": len(self._agents),
            }

    def get_events(self, limit: int = 100) -> list[EventLog]:
        """Get recent event logs."""
        return self._events[-limit:]

    def shutdown_all(self) -> None:
        """Kill all agents and shutdown executor."""
        self._shutdown = True
        with self._lock:
            for iid in list(self._agents.keys()):
                self.kill(iid)
        self._executor.shutdown(wait=True)
        self._log("shutdown")

    def __enter__(self) -> MetaSpawner:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown_all()

# Kimi-Swarm Design

**Date:** 2026-04-30  
**Scope:** MVP meta-spawner for multiple `KimiCore` agent instances  
**Status:** Draft

---

## Goal

Allow a single user task to be decomposed and executed in parallel by multiple specialized agent instances, with automatic aggregation of results.

## Architecture

`Swarm` orchestrates `KimiCore` agents. It supports three execution modes:

1. **Spawn** — single agent, single task (baseline)
2. **Parallel** — N agents, N independent tasks (batch execution)
3. **Decompose** — planner agent breaks task into subtasks → worker agents execute in parallel → aggregator combines results

## Components

### `AgentRole` dataclass

```python
@dataclass
class AgentRole:
    name: str
    system_prompt: str = ""
    model: str | None = None
    tools_subset: list[str] | None = None
    max_rounds: int = 10
```

### `Swarm` class

```python
class Swarm:
    def __init__(self, config: Config | None = None, shared_memory: PersistentMemory | None = None) -> None
    def spawn(self, role: AgentRole, task: str) -> str
    async def spawn_async(self, role: AgentRole, task: str) -> str
    def run_parallel(self, tasks: list[tuple[AgentRole, str]]) -> list[str]
    def run_decompose(self, task: str, planner_role: AgentRole | None = None, worker_role: AgentRole | None = None) -> str
```

**Spawn:** Creates a `KimiCore` with role-specific config, runs `agent.run(task)`, returns result.

**Parallel:** Spawns N agents concurrently via `asyncio.gather`, returns ordered list of results.

**Decompose:**
1. Planner agent breaks task into numbered subtasks (e.g., "1. Read README\n2. List files\n3. Check imports")
2. Swarm parses numbered lines into subtask strings
3. If parsing fails → fallback to single-agent execution
4. Workers execute each subtask in parallel
5. Aggregator agent combines worker outputs into final answer
6. All results saved to shared memory

### Shared Memory

All agents receive the same `PersistentMemory` instance. Each agent's conversation and tool outputs are appended to the shared memory. After a swarm run, `memory.save()` persists everything.

### Error Handling

- Worker failure → error message included in aggregation, `DecisionOutcome.FAILURE` logged
- Planner parse failure → fallback to single-agent execution of original task
- Individual worker timeout → treated as failure, error included in aggregation
- All errors propagate to shared memory and reflection log

### Default Roles

```python
PLANNER_ROLE = AgentRole(
    name="planner",
    system_prompt="Break this task into numbered subtasks. Each subtask should be independently executable. Return ONLY numbered list.",
)

WORKER_ROLE = AgentRole(
    name="worker",
    system_prompt="You are a worker agent. Execute the given subtask using available tools. Return concise results.",
)

AGGREGATOR_ROLE = AgentRole(
    name="aggregator",
    system_prompt="Summarize these worker results into a coherent final answer.",
)
```

## Files

| File | Responsibility |
|------|---------------|
| `kimi_core/swarm.py` | `AgentRole`, `Swarm`, default roles |
| `tests/test_swarm.py` | Unit tests with mock LLM for spawn, parallel, decompose |

## Testing Strategy

- **test_spawn:** Mock LLM returns "done". Assert `Swarm.spawn()` returns "done".
- **test_parallel:** Mock LLM returns indexed results. Assert `run_parallel` returns ordered list.
- **test_decompose:** Mock planner returns "1. A\n2. B". Mock workers return "a" and "b". Mock aggregator returns "a b". Assert final result is "a b".
- **test_decompose_fallback:** Mock planner returns garbage. Assert fallback to single agent.

## Data Flow

```
User task
  → Swarm.run_decompose()
    → Planner agent → numbered subtask list
    → Parser → list[str]
    → Workers (parallel) → list[str]
    → Aggregator agent → final answer
    → Shared memory save
    → Reflection log
```

## Spec Coverage Check

| Requirement | Section |
|-------------|---------|
| Spawn single agent | `spawn()` |
| Parallel execution | `run_parallel()` |
| Decompose → workers → aggregate | `run_decompose()` |
| Shared memory | Shared `PersistentMemory` |
| Error handling | Error Handling section |
| Testing | Testing Strategy section |

## Placeholder Scan

- No "TBD", "TODO", or vague requirements.
- All type names match kimi-core conventions.
- All files have exact paths.

## Ambiguity Check

- Subtask parsing: split by newline, keep lines matching `^\d+\.\s+`. If no matches, fallback.
- Result ordering: parallel results maintain input order (indexed gather).
- Shared memory thread safety: Python GIL protects list append; safe for async single-threaded event loop.

---

**Next step:** Write implementation plan after approval.

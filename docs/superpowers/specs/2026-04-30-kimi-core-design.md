# Kimi-Core MVP Design

## Overview

Kimi-Core is a deployable, self-hosted AI assistant CLI that uses a local LLM (Ollama) as its reasoning engine. It takes the existing `kimi-self` reference implementation and makes it a production-ready tool that can read/write files, execute shell commands, and self-improve through reflection.

## Goals

- Provide a standalone CLI agent that operates on the user's local filesystem.
- Use Ollama as the default LLM backend with a pluggable provider interface.
- Reuse and extend the existing `kimi-self` components (Agent Loop, Tool Router, Reflection).
- Support both single-task execution (`--task`) and interactive REPL mode.
- Persist memory and reflection logs across sessions.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI /     │────▶│   Agent     │────▶│   LLM       │
│  REPL       │     │   Loop      │     │ Provider    │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                     │
                           ▼                     ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ ToolRouter  │◀────│  Tool Call  │
                    │  (exists)   │     │  Parser     │
                    └──────┬──────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Tools:    │
                    │ read, write,│
                    │ shell, glob,│
                    │ grep, agent │
                    └─────────────┘
```

The core loop follows the existing `perceive → orient → decide → act → verify` cycle. The existing `agent_loop.py` becomes the real engine. An adapter layer connects it to the LLM backend.

## Components

| Component | Source | Role |
|-----------|--------|------|
| `kimi_core/agent.py` | extends `agent_loop.py` | Main loop, orchestration |
| `kimi_core/llm/` | new | `LLMProvider` base, `OllamaProvider` |
| `kimi_core/tools/` | extends `tool_router.py` | Concrete tool implementations |
| `kimi_core/memory.py` | extends existing `Memory` | Working + episodic memory |
| `kimi_core/reflection.py` | use existing | Decision tracking, self-improvement |
| `kimi_core/config.py` | use existing YAML | Runtime configuration |
| `kimi_core/cli.py` | new | Entry point |

### Key Additions

- **OllamaProvider**: Streams chat, parses tool calls from the LLM response, and formats available tools into the OpenAI-compatible function-calling schema.
- **Real Tools**: Tool implementations call the real filesystem and shell instead of stubs.
- **Persistent Memory**: Memory persists to `~/.kimi/memory/` across sessions.
- **CLI**: Supports single-task mode (`--task`) and interactive REPL mode.

## Data Flow

1. User input → `agent.py::run()`
2. `perceive()` → intent classify, extract entities (existing logic)
3. `orient()` → load config, check memory, load skills (existing logic + persistence)
4. `decide()` → if complex, build plan; else route to LLM directly
5. `act()` → LLM receives prompt + available tools + system prompt
6. LLM responds → if tool calls, `ToolRouter` executes; else respond directly
7. Tool outputs fed back to LLM as function results
8. `verify()` → check success (tool exit codes, heuristic)
9. `reflection.py` logs decision

## Error Handling

| Failure | Response |
|---------|----------|
| LLM unreachable | Retry 2× with backoff, then exit with error |
| Tool fails | Return error to LLM as tool result, let it retry |
| Tool timeout | Kill process, return timeout error |
| Invalid tool call | Parse error sent to LLM, request correction |
| Plan gets stuck | After 3 retries, escalate to user |

## Testing

- **Unit**: `OllamaProvider` tool formatting, `ToolRouter` dependency resolution.
- **Integration**: Run agent against mock LLM, verify tool chain executes.
- **Smoke**: `python -m kimi_core --task "what is 2+2"` with real Ollama.

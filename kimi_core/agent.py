"""Main agent loop for Kimi-Core."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from kimi_core.config import Config, load_config
from kimi_core.llm.base import LLMProvider, ToolCall
from kimi_core.llm.ollama import OllamaProvider
from kimi_core.memory import PersistentMemory
from kimi_core.reflection import DecisionOutcome, DecisionTracker
from kimi_core.tool_router import ToolRequest, ToolRouter
from kimi_core.tools import TOOL_DEFINITIONS, TOOL_HANDLERS


SYSTEM_PROMPT = """You are Kimi-Core, a self-hosted AI assistant. You help users with software engineering tasks by reading files, writing code, and running shell commands.

You have access to the following tools. Use them when needed. When you need to run a command or read a file, emit a tool call. Wait for the result, then respond to the user.

Rules:
- Read files before writing.
- Make minimal changes.
- Follow existing project style.
- Run tests after changing code.
"""


class KimiCore:
    def __init__(self, provider: LLMProvider | None = None, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.memory = PersistentMemory(memory_dir=self.config.memory_dir)
        self.memory.load()
        self.tools = ToolRouter()
        self._register_tools()
        self.tracker = DecisionTracker()
        self.tracker.log.log_path = Path(self.config.reflection_log_path)
        self.provider = provider or OllamaProvider(
            host=self.config.ollama_host,
            model=self.config.ollama_model,
        )

    def _register_tools(self) -> None:
        for name, handler in TOOL_HANDLERS.items():
            self.tools.register(name, handler)

    def run(self, user_input: str) -> str:
        decision_id = f"run_{uuid.uuid4().hex[:8]}"
        self.tracker.start_decision(decision_id)
        self.memory.add_interaction("user", user_input)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for turn in self.memory.conversation[-(self.config.working_memory_items * 2):]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        tools_used: list[str] = []
        outcome = DecisionOutcome.SUCCESS
        error = None
        final_content = ""

        try:
            for _ in range(10):  # max tool call rounds
                resp = self._sync_chat(messages, tools=TOOL_DEFINITIONS)

                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        tools_used.append(tc.name)
                        req = ToolRequest(tc.name, tc.arguments)
                        try:
                            result = self.tools.route(req)
                            self.memory.add_tool_output(tc.name, tc.arguments, result)
                        except Exception as e:
                            result = f"Error: {e}"
                            self.memory.add_tool_output(tc.name, tc.arguments, result)

                        messages.append({
                            "role": "assistant",
                            "content": f"Using tool {tc.name}",
                        })
                        messages.append({
                            "role": "tool",
                            "content": str(result),
                        })
                    continue

                if resp.content:
                    final_content = resp.content
                    break
                break

        except Exception as e:
            final_content = f"Error: {e}"
            outcome = DecisionOutcome.FAILURE
            error = str(e)

        self.memory.add_interaction("assistant", final_content)
        self.memory.save()

        self.tracker.record_decision(
            decision_id=decision_id,
            task_type="REQUEST",
            strategy="direct" if not tools_used else "plan_and_execute",
            confidence="HIGH",
            input_summary=user_input,
            tools_used=tools_used,
            outcome=outcome,
            error=error,
        )

        return final_content

    def _sync_chat(self, messages: list[dict], tools: list | None = None):
        import asyncio
        return asyncio.run(self.provider.chat(messages, tools=tools))

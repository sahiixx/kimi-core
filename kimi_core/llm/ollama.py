"""Ollama LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any

import httpx

from kimi_core.llm.base import LLMProvider, LLMResponse, ToolCall


class OllamaProvider(LLMProvider):
    """Ollama backend using its OpenAI-compatible chat API."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5-coder:14b") -> None:
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send chat request to Ollama."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = self.format_tools(tools)

        resp = await self._client.post(f"{self.host}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content")
        tool_calls = self.parse_tool_calls(data)

        return LLMResponse(content=content, tool_calls=tool_calls, raw=data)

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool defs to Ollama/OpenAI function schema."""
        formatted = []
        for t in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return formatted

    def parse_tool_calls(self, raw_response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool_calls from Ollama response."""
        calls: list[ToolCall] = []
        message = raw_response.get("message", {})
        raw_calls = message.get("tool_calls", [])
        for rc in raw_calls:
            fn = rc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw
            calls.append(ToolCall(name=name, arguments=args))
        return calls

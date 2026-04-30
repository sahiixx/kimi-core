"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A parsed tool call from an LLM response."""
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Pluggable LLM backend interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a standardized response."""

    @abstractmethod
    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal tool definitions to provider-specific schema."""

    @abstractmethod
    def parse_tool_calls(self, raw_response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from provider-specific response format."""

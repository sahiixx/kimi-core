"""Extended ToolRouter with result capture."""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class ToolStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ToolRequest:
    tool_name: str
    arguments: dict[str, Any]
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    dependencies: list[str] = field(default_factory=list)
    status: ToolStatus = ToolStatus.PENDING
    result: Any = None
    error: str | None = None


class ToolRouter:
    def __init__(self, max_workers: int = 10) -> None:
        self._tools: dict[str, Callable] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._history: list[ToolRequest] = []

    def register(self, name: str, handler: Callable, description: str = "") -> None:
        self._tools[name] = handler

    def route(self, request: ToolRequest) -> Any:
        if request.tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {request.tool_name}")
        request.status = ToolStatus.RUNNING
        try:
            result = self._tools[request.tool_name](**request.arguments)
            request.result = result
            request.status = ToolStatus.COMPLETED
        except Exception as e:
            request.error = str(e)
            request.status = ToolStatus.FAILED
            raise
        finally:
            self._history.append(request)
        return result

    async def route_async(self, request: ToolRequest) -> Any:
        if request.tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {request.tool_name}")
        request.status = ToolStatus.RUNNING
        try:
            handler = self._tools[request.tool_name]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**request.arguments)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor, lambda: handler(**request.arguments)
                )
            request.result = result
            request.status = ToolStatus.COMPLETED
            return result
        except Exception as e:
            request.error = str(e)
            request.status = ToolStatus.FAILED
            raise
        finally:
            self._history.append(request)

    def execute_batch(self, requests: list[ToolRequest]) -> list[ToolRequest]:
        return asyncio.run(self.execute_batch_async(requests))

    async def execute_batch_async(self, requests: list[ToolRequest]) -> list[ToolRequest]:
        completed = set()
        pending = {r.request_id: r for r in requests}

        while pending:
            ready = [
                r for r in pending.values()
                if all(dep in completed for dep in r.dependencies)
                and r.status == ToolStatus.PENDING
            ]
            if not ready and pending:
                raise RuntimeError(f"Circular dependencies: {list(pending.keys())}")
            tasks = [self.route_async(r) for r in ready]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for r in ready:
                if r.status in (ToolStatus.COMPLETED, ToolStatus.FAILED):
                    completed.add(r.request_id)
                    pending.pop(r.request_id, None)
        return requests

    def get_history(self) -> list[ToolRequest]:
        return self._history.copy()

import pytest
from kimi_core.agent import KimiCore
from kimi_core.llm.base import LLMProvider, LLMResponse, ToolCall


class FakeProvider(LLMProvider):
    def __init__(self, responses):
        self.responses = responses
        self.idx = 0

    async def chat(self, messages, tools=None, model=None):
        resp = self.responses[self.idx]
        self.idx += 1
        return resp

    def format_tools(self, tools):
        return tools

    def parse_tool_calls(self, raw):
        return raw.get("tool_calls", [])


def test_agent_direct_answer():
    provider = FakeProvider([LLMResponse(content="The answer is 42")])
    agent = KimiCore(provider=provider)
    result = agent.run("what is 2+2")
    assert "42" in result


def test_agent_tool_call():
    provider = FakeProvider([
        LLMResponse(tool_calls=[ToolCall("read_file", {"path": "/tmp/test.txt"})]),
        LLMResponse(content="File contains hello"),
    ])
    agent = KimiCore(provider=provider)
    result = agent.run("read /tmp/test.txt")
    assert "File contains hello" in result

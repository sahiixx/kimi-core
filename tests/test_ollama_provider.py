import pytest
from kimi_core.llm.base import LLMProvider, ToolCall
from kimi_core.llm.ollama import OllamaProvider


def test_ollama_provider_init():
    p = OllamaProvider(host="http://localhost:11434", model="qwen2.5-coder:14b")
    assert p.host == "http://localhost:11434"
    assert p.model == "qwen2.5-coder:14b"


def test_format_tools():
    p = OllamaProvider()
    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        }
    ]
    formatted = p.format_tools(tools)
    assert len(formatted) == 1
    assert formatted[0]["type"] == "function"
    assert formatted[0]["function"]["name"] == "read_file"
    assert "path" in formatted[0]["function"]["parameters"]["properties"]


def test_parse_tool_calls():
    p = OllamaProvider()
    raw = {
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/etc/passwd"}',
                    }
                }
            ]
        }
    }
    calls = p.parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/etc/passwd"}


def test_parse_no_tool_calls():
    p = OllamaProvider()
    raw = {"message": {"content": "Hello"}}
    calls = p.parse_tool_calls(raw)
    assert calls == []

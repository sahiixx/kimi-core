import os
import tempfile

from kimi_core.config import Config, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.agent_name == "Kimi Code CLI"
    assert cfg.llm_backend == "ollama"
    assert cfg.ollama_model == "qwen2.5-coder:14b"
    assert cfg.ollama_host == "http://localhost:11434"


def test_load_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("agent_name: TestAgent\nllm_backend: openai\n")
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.agent_name == "TestAgent"
        assert cfg.llm_backend == "openai"
    finally:
        os.unlink(path)

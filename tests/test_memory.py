import os
import tempfile

from kimi_core.memory import PersistentMemory


def test_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        mem = PersistentMemory(memory_dir=td)
        mem.add_interaction("user", "hello")
        mem.add_tool_output("read_file", {"path": "/tmp/x"}, "content")
        mem.save()

        mem2 = PersistentMemory(memory_dir=td)
        mem2.load()
        assert len(mem2.conversation) == 1
        assert mem2.conversation[0]["content"] == "hello"
        assert len(mem2.tool_outputs) == 1


def test_file_cache():
    with tempfile.TemporaryDirectory() as td:
        mem = PersistentMemory(memory_dir=td)
        mem.file_cache["/tmp/test.py"] = "print(1)"
        mem.save()

        mem2 = PersistentMemory(memory_dir=td)
        mem2.load()
        assert mem2.file_cache["/tmp/test.py"] == "print(1)"

import os
import tempfile

from kimi_core.tools.fs import read_file, write_file, glob_files, grep_files
from kimi_core.tools.shell import run_shell


def test_read_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        assert read_file(path) == "hello world"
    finally:
        os.unlink(path)


def test_read_file_missing():
    result = read_file("/tmp/nonexistent_file_12345.txt")
    assert result.startswith("Error:")


def test_write_file():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.txt")
        result = write_file(path, "content")
        assert "Success" in result
        with open(path) as f:
            assert f.read() == "content"


def test_glob_files():
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "a.py"), "w").close()
        open(os.path.join(td, "b.txt"), "w").close()
        files = glob_files("*.py", td)
        assert any("a.py" in f for f in files)


def test_grep_files():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "x.py"), "w") as f:
            f.write("target_word = 1\n")
        results = grep_files("target_word", td)
        assert len(results) == 1
        assert "target_word" in results[0]


def test_run_shell_echo():
    result = run_shell("echo hello")
    assert "hello" in result


def test_run_shell_timeout():
    result = run_shell("sleep 10", timeout=0.1)
    assert "timed out" in result.lower() or "timeout" in result.lower()

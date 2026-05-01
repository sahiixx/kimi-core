from unittest.mock import patch, MagicMock

from kimi_core.cli import main


def test_cli_task_mode():
    with patch("sys.argv", ["kimi", "run", "--task", "hello"]), \
         patch("kimi_core.cli.KimiCore") as MockAgent:
        instance = MockAgent.return_value
        instance.run.return_value = "Hi"
        assert main() == 0
        instance.run.assert_called_once_with("hello")


def test_cli_swarm_decompose():
    with patch("sys.argv", ["kimi", "swarm", "--decompose", "analyze repo"]), \
         patch("kimi_core.cli.Swarm") as MockSwarm:
        instance = MockSwarm.return_value
        instance.run_decompose.return_value = "done"
        assert main() == 0
        instance.run_decompose.assert_called_once_with("analyze repo")


def test_cli_swarm_parallel():
    with patch("sys.argv", ["kimi", "swarm", "--parallel", "task1, task2"]), \
         patch("kimi_core.cli.Swarm") as MockSwarm:
        instance = MockSwarm.return_value
        instance.run_parallel.return_value = ["r1", "r2"]
        assert main() == 0
        instance.run_parallel.assert_called_once()
        call_args = instance.run_parallel.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0][1] == "task1"
        assert call_args[1][1] == "task2"

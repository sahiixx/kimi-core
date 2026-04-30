from unittest.mock import patch

from kimi_core.cli import main


def test_cli_task_mode():
    with patch("sys.argv", ["kimi", "--task", "hello"]), \
         patch("kimi_core.cli.KimiCore") as MockAgent:
        instance = MockAgent.return_value
        instance.run.return_value = "Hi"
        main()
        instance.run.assert_called_once_with("hello")

import pytest
from unittest.mock import patch

from kimi_core.swarm import AgentRole, Swarm


def test_agent_role_defaults():
    role = AgentRole(name="planner")
    assert role.name == "planner"
    assert role.system_prompt == ""
    assert role.model is None
    assert role.tools_subset is None
    assert role.max_rounds == 10


def test_swarm_spawn():
    swarm = Swarm()
    role = AgentRole(name="test", system_prompt="You are a test agent.")
    result = swarm.spawn(role, "say hello")
    assert isinstance(result, str)


def test_decompose_happy_path():
    swarm = Swarm()
    with patch.object(swarm, "spawn") as mock_spawn:
        mock_spawn.side_effect = [
            "1. Read README\n2. List files",
            "readme content",
            "file list",
            "final answer",
        ]
        result = swarm.run_decompose("analyze repo")
        assert result == "final answer"
        assert mock_spawn.call_count == 4


def test_decompose_fallback_on_bad_plan():
    swarm = Swarm()
    with patch.object(swarm, "spawn") as mock_spawn:
        mock_spawn.side_effect = [
            "garbage text with no numbers",
            "fallback answer",
        ]
        result = swarm.run_decompose("analyze repo")
        assert result == "fallback answer"
        assert mock_spawn.call_count == 2


def test_run_parallel():
    swarm = Swarm()
    with patch.object(swarm, "spawn_async") as mock_spawn:
        async def side_effect(role, task):
            return f"result-{task}"
        mock_spawn.side_effect = side_effect
        results = swarm.run_parallel([
            (AgentRole(name="w1"), "task1"),
            (AgentRole(name="w2"), "task2"),
        ])
        assert results == ["result-task1", "result-task2"]


def test_run_parallel_with_exception():
    swarm = Swarm()
    with patch.object(swarm, "spawn_async") as mock_spawn:
        async def side_effect(role, task):
            if task == "bad":
                raise RuntimeError("boom")
            return f"ok-{task}"
        mock_spawn.side_effect = side_effect
        results = swarm.run_parallel([
            (AgentRole(name="w1"), "good"),
            (AgentRole(name="w2"), "bad"),
        ])
        assert results[0] == "ok-good"
        assert isinstance(results[1], RuntimeError)

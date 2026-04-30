import pytest
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

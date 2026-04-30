import pytest
from kimi_core.tool_router import ToolRouter, ToolRequest


def test_register_and_execute():
    router = ToolRouter()
    router.register("add", lambda a, b: a + b, description="Add two numbers")
    req = ToolRequest("add", {"a": 2, "b": 3})
    result = router.route(req)
    assert result == 5
    assert req.status.name == "COMPLETED"


def test_batch_parallel():
    router = ToolRouter()
    router.register("double", lambda x: x * 2)
    reqs = [
        ToolRequest("double", {"x": 1}),
        ToolRequest("double", {"x": 2}),
    ]
    results = router.execute_batch(reqs)
    assert len(results) == 2
    assert results[0].result == 2
    assert results[1].result == 4


def test_dependencies():
    router = ToolRouter()
    router.register("inc", lambda x: x + 1)
    req1 = ToolRequest("inc", {"x": 0}, request_id="r1")
    req2 = ToolRequest("inc", {"x": 0}, request_id="r2", dependencies=["r1"])
    results = router.execute_batch([req1, req2])
    assert results[0].status.name == "COMPLETED"
    assert results[1].status.name == "COMPLETED"

"""Tests for meta_spawner module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from kimi_core.meta_spawner import (
    AgentInstance,
    AgentRole,
    AgentStatus,
    CapabilityRegistry,
    EventLog,
    MetaSpawner,
    TaskItem,
    TaskStatus,
)


class TestCapabilityRegistry:
    def test_register_and_find(self):
        reg = CapabilityRegistry()
        role = AgentRole(name="coder", system_prompt="You code.")
        reg.register("coder", role, ["python", "javascript"])

        found = reg.find_best_role(["python"])
        assert found == role

    def test_find_best_multi_match(self):
        reg = CapabilityRegistry()
        r1 = AgentRole(name="frontend", system_prompt="FE")
        r2 = AgentRole(name="fullstack", system_prompt="FS")
        reg.register("frontend", r1, ["react", "css"])
        reg.register("fullstack", r2, ["react", "css", "python", "sql"])

        found = reg.find_best_role(["react", "python"])
        assert found == r2

    def test_find_none(self):
        reg = CapabilityRegistry()
        assert reg.find_best_role(["unknown"]) is None

    def test_list_roles(self):
        reg = CapabilityRegistry()
        role = AgentRole(name="test")
        reg.register("test", role, ["a"])
        assert len(reg.list_roles()) == 1


class TestMetaSpawnerBasics:
    def test_init(self):
        spawner = MetaSpawner()
        assert spawner.config is not None
        assert spawner.registry is not None
        assert spawner.a2a is not None
        assert spawner.get_queue_status()["agents"] == 0

    def test_default_roles_registered(self):
        spawner = MetaSpawner()
        roles = spawner.registry.list_roles()
        names = [n for n, _, _ in roles]
        assert "planner" in names
        assert "worker" in names
        assert "aggregator" in names

    def test_spawn_agent(self):
        spawner = MetaSpawner()
        role = AgentRole(name="test_worker", system_prompt="Test.")

        with patch.object(spawner, "_next_port", return_value=9001):
            inst = spawner.spawn(role)

        assert inst.instance_id.startswith("agent_")
        assert inst.role.name == "test_worker"
        assert inst.port == 9001
        assert inst.status == AgentStatus.HEALTHY
        assert inst._agent is not None

    def test_spawn_by_capability(self):
        spawner = MetaSpawner()
        spawner.registry.register("db", AgentRole(name="db"), ["sql", "postgres"])

        with patch.object(spawner, "_next_port", return_value=9002):
            inst = spawner.spawn("sql")

        assert inst.role.name == "db"

    def test_spawn_unknown_capability(self):
        spawner = MetaSpawner()
        with pytest.raises(ValueError):
            spawner.spawn("unknown_capability")

    def test_kill_agent(self):
        spawner = MetaSpawner()
        role = AgentRole(name="temp")

        with patch.object(spawner, "_next_port", return_value=9003):
            inst = spawner.spawn(role)

        spawner.kill(inst.instance_id)
        assert inst.status == AgentStatus.DEAD
        assert inst.instance_id not in spawner._agents

    def test_health_check_healthy(self):
        spawner = MetaSpawner()
        role = AgentRole(name="hc")

        with patch.object(spawner, "_next_port", return_value=9004):
            inst = spawner.spawn(role)

        mock_agent = Mock()
        mock_agent.run = Mock(return_value="heartbeat ok")
        inst._agent = mock_agent

        assert spawner.health_check(inst.instance_id) is True
        assert inst.status == AgentStatus.HEALTHY

    def test_health_check_unresponsive(self):
        spawner = MetaSpawner()
        role = AgentRole(name="hc_bad")

        with patch.object(spawner, "_next_port", return_value=9005):
            inst = spawner.spawn(role)

        mock_agent = Mock()
        mock_agent.run = Mock(side_effect=RuntimeError("broken"))
        inst._agent = mock_agent

        assert spawner.health_check(inst.instance_id) is False
        assert inst.status == AgentStatus.UNRESPONSIVE

    def test_health_check_all(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", side_effect=[9006, 9007]):
            inst1 = spawner.spawn(AgentRole(name="a1"))
            inst2 = spawner.spawn(AgentRole(name="a2"))

        mock_agent = Mock()
        mock_agent.run = Mock(return_value="ok")
        inst1._agent = mock_agent
        inst2._agent = mock_agent

        results = spawner.health_check_all()
        assert len(results) == 2
        assert all(results.values())

    def test_restart_unhealthy(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", return_value=9008):
            inst = spawner.spawn(AgentRole(name="sick"))

        inst.status = AgentStatus.UNRESPONSIVE
        restarted = spawner.restart_unhealthy()
        assert len(restarted) == 1
        assert restarted[0] == inst.instance_id


class TestTaskQueue:
    def test_submit_task(self):
        spawner = MetaSpawner()
        tid = spawner.submit_task("test task", priority=1)
        assert tid.startswith("task_")
        assert spawner.get_queue_status()["queued"] == 1

    def test_submit_with_capabilities(self):
        spawner = MetaSpawner()
        tid = spawner.submit_task("code review", required_capabilities=["python"])
        with spawner._lock:
            task = next(t for t in spawner._task_queue if t.task_id == tid)
        assert task.required_capabilities == ["python"]

    def test_route_tasks_no_agents(self):
        spawner = MetaSpawner()
        spawner.submit_task("test")
        assigned = spawner.route_tasks()
        assert assigned == []

    def test_route_tasks_with_agent(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", return_value=9009):
            spawner.spawn(AgentRole(name="worker"))

        tid = spawner.submit_task("test task")
        assigned = spawner.route_tasks()
        assert len(assigned) == 1
        assert assigned[0] == tid

    def test_route_tasks_capability_match(self):
        spawner = MetaSpawner()
        spawner.registry.register("py", AgentRole(name="py"), ["python"])
        spawner.registry.register("gen", AgentRole(name="gen"), ["generic"])

        with patch.object(spawner, "_next_port", side_effect=[9010, 9011]):
            spawner.spawn(AgentRole(name="py"))
            spawner.spawn(AgentRole(name="gen"))

        tid1 = spawner.submit_task("generic", required_capabilities=["generic"])
        tid2 = spawner.submit_task("python task", required_capabilities=["python"])

        assigned = spawner.route_tasks()
        # Both tasks get assigned to their capability-matched agents
        assert tid1 in assigned
        assert tid2 in assigned

    def test_run_task(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", return_value=9011):
            inst = spawner.spawn(AgentRole(name="exec"))

        mock_agent = Mock()
        mock_agent.run = Mock(return_value="done")
        inst._agent = mock_agent

        # Manually assign task
        task = TaskItem(
            priority=1,
            task_id="task_x",
            task_text="do thing",
            assigned_agent=inst.instance_id,
            status=TaskStatus.ASSIGNED,
        )
        with spawner._lock:
            spawner._task_queue.append(task)

        result = spawner.run_task("task_x")
        assert result == "done"
        assert inst.total_tasks == 1

    def test_run_task_not_found(self):
        spawner = MetaSpawner()
        with pytest.raises(ValueError):
            spawner.run_task("nonexistent")

    def test_run_task_failure(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", return_value=9012):
            inst = spawner.spawn(AgentRole(name="failer"))

        mock_agent = Mock()
        mock_agent.run = Mock(side_effect=RuntimeError("boom"))
        inst._agent = mock_agent

        task = TaskItem(
            priority=1,
            task_id="task_fail",
            task_text="fail",
            assigned_agent=inst.instance_id,
            status=TaskStatus.ASSIGNED,
        )
        with spawner._lock:
            spawner._task_queue.append(task)

        result = spawner.run_task("task_fail")
        assert "Error" in result or "boom" in result
        assert inst.failed_tasks == 1
        assert inst.status == AgentStatus.UNRESPONSIVE


class TestRunParallel:
    def test_run_parallel(self):
        spawner = MetaSpawner()
        roles = [AgentRole(name=f"w{i}") for i in range(3)]
        tasks = [(role, f"task {i}") for i, role in enumerate(roles)]

        with patch.object(spawner, "_next_port", side_effect=range(9013, 9016)):
            with patch.object(spawner._executor, "submit") as mock_submit:
                future = Mock()
                future.result = Mock(return_value="result")
                mock_submit.return_value = future

                results = spawner.run_parallel(tasks)

        assert len(results) == 3
        assert all(r == "result" for r in results)


class TestA2ABridge:
    def test_a2a_not_available_when_load_fails(self):
        spawner = MetaSpawner()
        spawner.a2a._client_class = None
        spawner.a2a._make_tools = None
        assert spawner.a2a.is_available() is False

    def test_a2a_discover_fallback(self):
        spawner = MetaSpawner()
        spawner.a2a._client_class = None
        result = spawner.a2a_discover("http://localhost:8100")
        # Should gracefully return empty/error without crashing
        assert isinstance(result, dict)

    def test_a2a_route_task_fallback(self):
        spawner = MetaSpawner()
        spawner.a2a._client_class = None
        result = spawner.a2a_route_task("http://localhost:8100", "test")
        assert "A2A unavailable" in result


class TestDecompose:
    def test_run_decompose(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "spawn") as mock_spawn:
            planner = Mock()
            planner._agent.run = Mock(return_value="1. Subtask A\n2. Subtask B")

            worker1 = Mock()
            worker1._agent.run = Mock(return_value="result A")
            worker2 = Mock()
            worker2._agent.run = Mock(return_value="result B")

            aggregator = Mock()
            aggregator._agent.run = Mock(return_value="final summary")

            mock_spawn.side_effect = [planner, worker1, worker2, aggregator, worker1, worker2]

            result = spawner.run_decompose("big task")

            assert result == "final summary"
            assert mock_spawn.call_count == 4
            planner._agent.run.assert_called_once()
            aggregator._agent.run.assert_called_once()

    def test_run_decompose_no_subtasks(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "spawn") as mock_spawn:
            planner = Mock()
            planner._agent.run = Mock(return_value="no numbered list here")

            worker = Mock()
            worker._agent.run = Mock(return_value="fallback result")

            mock_spawn.side_effect = [planner, worker]

            result = spawner.run_decompose("simple task")
            assert result == "fallback result"
            assert mock_spawn.call_count == 2


class TestEventLogging:
    def test_events_logged(self):
        spawner = MetaSpawner()
        spawner._log("test_event", foo="bar")
        events = spawner.get_events()
        assert len(events) == 1
        assert events[0].event_type == "test_event"
        assert events[0].details["foo"] == "bar"

    def test_event_timestamp(self):
        spawner = MetaSpawner()
        spawner._log("timed")
        events = spawner.get_events()
        assert events[0].timestamp is not None
        assert "T" in events[0].timestamp


class TestStatusMethods:
    def test_get_agent_status(self):
        spawner = MetaSpawner()

        with patch.object(spawner, "_next_port", return_value=9016):
            inst = spawner.spawn(AgentRole(name="status_test"))

        status = spawner.get_agent_status()
        assert inst.instance_id in status
        assert status[inst.instance_id]["role"] == "status_test"
        assert "uptime" in status[inst.instance_id]

    def test_get_queue_status(self):
        spawner = MetaSpawner()
        spawner.submit_task("t1")
        spawner.submit_task("t2")
        status = spawner.get_queue_status()
        assert status["queued"] == 2
        assert status["completed"] == 0
        assert status["agents"] == 0


class TestContextManager:
    def test_context_manager(self):
        with MetaSpawner() as spawner:
            assert spawner is not None
            spawner.submit_task("test")

    def test_shutdown_on_exit(self):
        spawner = MetaSpawner()
        with patch.object(spawner, "shutdown_all") as mock_shutdown:
            with spawner:
                pass
            mock_shutdown.assert_called_once()

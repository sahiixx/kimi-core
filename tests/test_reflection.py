import os
import tempfile
from pathlib import Path

from kimi_core.reflection import DecisionOutcome, DecisionTracker


def test_track_decision():
    with tempfile.TemporaryDirectory() as td:
        path = Path(os.path.join(td, "reflections.jsonl"))
        tracker = DecisionTracker()
        tracker.log.log_path = path
        tracker.start_decision("d1")
        tracker.record_decision(
            decision_id="d1",
            task_type="QUESTION",
            strategy="direct",
            confidence="HIGH",
            input_summary="hello",
            tools_used=[],
            outcome=DecisionOutcome.SUCCESS,
        )
        assert len(tracker.log._entries) == 1
        stats = tracker.log.get_stats()
        assert stats["total"] == 1
        assert stats["success_rate"] == 1.0

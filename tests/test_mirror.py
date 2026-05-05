"""Tests for kimi-mirror behavior clone."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kimi_core.mirror import (
    CodeProfiler,
    GitProfiler,
    MirrorAgent,
    MirrorProfile,
    MirrorTrainer,
    PreferenceModel,
    build_mirror,
)


class TestMirrorProfile:
    def test_defaults(self):
        p = MirrorProfile()
        assert p.user_name == "sahiix"
        assert p.naming_style == "snake_case"
        assert p.architecture_pref == "modular"
        assert p.approval_mode == "omega"
        assert p.parallel_pref == "aggressive"
        assert p.interaction_count == 0
        assert p.decision_accuracy == 0.0

    def test_serialization_roundtrip(self):
        p = MirrorProfile(user_name="test", interaction_count=5)
        data = json.dumps(p.__dict__, default=str)
        loaded = json.loads(data)
        assert loaded["user_name"] == "test"
        assert loaded["interaction_count"] == 5


class TestGitProfiler:
    def test_empty_no_crash(self):
        gp = GitProfiler([])
        assert gp.prefix_distribution() == {}
        assert gp.infer_style() == {}

    def test_prefix_distribution(self):
        gp = GitProfiler([])
        gp._commits = [
            {"msg": "feat: add foo", "hash": "abc", "date": "2024-01-01", "repo": "/a"},
            {"msg": "fix: bar", "hash": "def", "date": "2024-01-02", "repo": "/a"},
            {"msg": "feat: add baz", "hash": "ghi", "date": "2024-01-03", "repo": "/a"},
            {"msg": "freeform message", "hash": "jkl", "date": "2024-01-04", "repo": "/a"},
        ]
        dist = gp.prefix_distribution()
        assert dist["feat"] == 2
        assert dist["fix"] == 1
        assert dist["freeform"] == 1

    def test_infer_style(self):
        gp = GitProfiler([])
        gp._commits = [
            {"msg": "feat: add user_login", "hash": "a", "date": "2024-01-01", "repo": "/a"},
            {"msg": "fix: handle error", "hash": "b", "date": "2024-01-02", "repo": "/a"},
        ]
        style = gp.infer_style()
        assert style["commit_prefixes"][0] == "feat"
        assert style["naming_style"] == "snake_case"

    def test_scan_skips_non_git(self, tmp_path):
        gp = GitProfiler([str(tmp_path)])
        gp.scan()
        assert len(gp._commits) == 0


class TestCodeProfiler:
    def test_empty(self):
        cp = CodeProfiler([])
        assert cp.analyze() == {}

    def test_analyzes_python(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            'from dataclasses import dataclass\n\n@dataclass\nclass Config:\n    """A config."""\n    x: int\n\n# comment\nasync def fetch():\n    pass\n\ndef test_foo():\n    assert True\n',
            encoding="utf-8",
        )
        cp = CodeProfiler([str(tmp_path)])
        result = cp.analyze()
        assert result["architecture_pref"] == "dataclass_driven"
        assert result["async_style"] == "async_first"
        assert result["docstring_level"] == "full"
        assert result["test_style"] == "pytest"
        assert result["comment_density"] == "medium"

    def test_analyzes_typescript(self, tmp_path):
        (tmp_path / "sample.ts").write_text(
            "// comment\nimport { foo } from './bar';\n\nfunction add(x: number, y: number): number {\n  return x + y;\n}\n",
            encoding="utf-8",
        )
        cp = CodeProfiler([str(tmp_path)])
        result = cp.analyze()
        assert result["architecture_pref"] == "class_driven"
        assert result["import_style"] == "from_import"


class TestPreferenceModel:
    def test_load_from_memory(self, tmp_path):
        mem_file = tmp_path / "user_prefs.md"
        mem_file.write_text(
            "User prefers aggressive parallelization and omega mode approvals.\n"
            "Architecture should be modular.\n"
            "Naming style: snake_case.\n"
            "Minimal comments please.\n",
            encoding="utf-8",
        )
        pm = PreferenceModel(str(tmp_path))
        pm.load()
        assert pm.get("parallel_pref") == "aggressive"
        assert pm.get("approval_mode") == "omega"
        assert pm.get("architecture_pref") == "modular"
        assert pm.get("naming_style") == "snake_case"
        assert pm.get("comment_density") == "low"

    def test_load_skills(self, tmp_path):
        mem_file = tmp_path / "skills.md"
        mem_file.write_text(
            "| Skill | Status |\n| brainstorming | LOADED |\n| graphify | LOADED |\n",
            encoding="utf-8",
        )
        pm = PreferenceModel(str(tmp_path))
        pm.load()
        skills = pm.get("active_skills", [])
        assert "brainstorming" in skills
        assert "graphify" in skills

    def test_empty_dir(self, tmp_path):
        pm = PreferenceModel(str(tmp_path))
        pm.load()
        assert pm.to_dict() == {}


class TestMirrorTrainer:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))
        trainer.profile.user_name = "test_user"
        trainer.save()

        trainer2 = MirrorTrainer(str(path))
        assert trainer2.profile.user_name == "test_user"

    def test_record_decision_updates_accuracy(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))

        trainer.record_decision("ctx1", "A", "A")
        assert trainer.profile.interaction_count == 1
        assert trainer.profile.decision_accuracy == 1.0

        trainer.record_decision("ctx2", "B", "C")
        assert trainer.profile.interaction_count == 2
        assert abs(trainer.profile.decision_accuracy - 0.5) < 0.01

    def test_update_from_analysis(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))
        trainer.update_from_analysis(
            git_analysis={"naming_style": "camelCase", "async_style": "async_first"},
            code_analysis={"architecture_pref": "class_driven"},
            prefs={"parallel_pref": "moderate"},
        )
        assert trainer.profile.naming_style == "camelCase"
        assert trainer.profile.async_style == "async_first"
        assert trainer.profile.architecture_pref == "class_driven"
        assert trainer.profile.parallel_pref == "moderate"


class TestMirrorAgent:
    def test_predict_commit_prefix(self, tmp_path):
        path = tmp_path / "profile.json"
        agent = MirrorAgent(MirrorTrainer(str(path)))

        assert agent.predict("commit_prefix", change="add new login feature") == "feat"
        assert agent.predict("commit_prefix", change="fix crash in parser") == "fix"
        assert agent.predict("commit_prefix", change="update dependencies") == "chore"
        assert agent.predict("commit_prefix", change="add tests for router") == "test"

    def test_predict_architecture(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))
        trainer.profile.architecture_pref = "modular"
        agent = MirrorAgent(trainer)

        # Modular preference should pick option with "module" or "bridge"
        choice = agent.predict(
            "architecture_choice",
            option_a="One big file with everything",
            option_b="Separate modules connected by bridge",
        )
        assert choice == "B"

        trainer.profile.architecture_pref = "monolithic"
        choice = agent.predict(
            "architecture_choice",
            option_a="One big file with everything",
            option_b="Separate modules connected by bridge",
        )
        assert choice == "A"

    def test_predict_parallelization(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))
        trainer.profile.parallel_pref = "aggressive"
        agent = MirrorAgent(trainer)

        assert agent.predict("parallelization", n=2) == "parallel"
        assert agent.predict("parallelization", n=1) == "sequential"

        trainer.profile.parallel_pref = "conservative"
        assert agent.predict("parallelization", n=10) == "sequential"

        trainer.profile.parallel_pref = "moderate"
        assert agent.predict("parallelization", n=3) == "parallel"
        assert agent.predict("parallelization", n=2) == "sequential"

    def test_predict_approval(self, tmp_path):
        path = tmp_path / "profile.json"
        trainer = MirrorTrainer(str(path))
        agent = MirrorAgent(trainer)

        trainer.profile.approval_mode = "omega"
        assert agent.predict("approval_needed", action="delete file") == "just_do"

        trainer.profile.approval_mode = "inform"
        assert agent.predict("approval_needed", action="delete file") == "inform"

        trainer.profile.approval_mode = "ask"
        assert agent.predict("approval_needed", action="delete file") == "ask"

    def test_build_prompt(self, tmp_path):
        path = tmp_path / "profile.json"
        agent = MirrorAgent(MirrorTrainer(str(path)))
        prompt = agent.build_prompt("commit_prefix", change="add foo")
        assert "sahiix" in prompt
        assert "add foo" in prompt

    def test_record_outcome(self, tmp_path):
        path = tmp_path / "profile.json"
        agent = MirrorAgent(MirrorTrainer(str(path)))
        agent.record_outcome("ctx", "A", "A")
        assert agent.profile.interaction_count == 1

    def test_unknown_decision_type(self, tmp_path):
        agent = MirrorAgent(MirrorTrainer(str(tmp_path / "profile.json")))
        assert agent.predict("something_unknown") == "unknown"


class TestBuildMirror:
    def test_build_mirror_runs(self, tmp_path):
        with patch("kimi_core.mirror.load_config") as mock_config:
            cfg = Mock()
            cfg.memory_dir = str(tmp_path)
            mock_config.return_value = cfg

            agent = build_mirror()
            assert isinstance(agent, MirrorAgent)
            assert agent.profile is not None

    def test_build_mirror_with_explicit_path(self, tmp_path):
        profile_path = tmp_path / "mirror.json"
        with patch("kimi_core.mirror.load_config") as mock_config:
            cfg = Mock()
            cfg.memory_dir = str(tmp_path)
            mock_config.return_value = cfg

            agent = build_mirror(str(profile_path))
            assert agent.profile.user_name == "sahiix"

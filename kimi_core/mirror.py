"""kimi-mirror: behavior clone that learns the user's patterns and preferences.

Learns from:
- Git history (commit message style, file patterns, change frequency)
- Code patterns (naming conventions, architecture choices, style preferences)
- Memory files (explicit preferences, feedback, role context)
- Live interactions (decisions, approvals, rejections)

Provides a MirrorAgent that can predict "what would the user choose here?"
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kimi_core.config import Config, load_config


@dataclass
class MirrorProfile:
    """Learned user behavior model."""

    user_name: str = "sahiix"
    commit_prefixes: list[str] = field(default_factory=list)
    top_file_types: list[str] = field(default_factory=list)
    naming_style: str = "snake_case"  # snake_case, camelCase, PascalCase
    architecture_pref: str = "modular"  # modular, monolithic, functional
    async_style: str = "sync_first"  # sync_first, async_first, mixed
    docstring_level: str = "minimal"  # none, minimal, full
    test_style: str = "pytest_mock"  # pytest_mock, unittest, none
    comment_density: str = "low"  # low, medium, high
    import_style: str = "absolute"  # absolute, relative, mixed
    max_line_length: int = 88
    parallel_pref: str = "aggressive"  # conservative, moderate, aggressive
    approval_mode: str = "omega"  # ask, inform, omega
    voice_triggers: list[str] = field(default_factory=list)
    active_skills: list[str] = field(default_factory=list)
    swarm_mode: str = "coordinator"  # coordinator, consensus, pipeline
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interaction_count: int = 0
    decision_accuracy: float = 0.0


class GitProfiler:
    """Analyze git history for user patterns."""

    PREFIX_RE = re.compile(r"^(feat|fix|chore|docs|test|refactor|build|ci|auto|merge|init)\b")
    CONVENTIONAL_RE = re.compile(r"^(\w+)(\([^)]*\))?:\s*")

    def __init__(self, repo_paths: list[str] | None = None) -> None:
        self.repo_paths = repo_paths or []
        self._commits: list[dict[str, Any]] = []

    def scan(self, max_commits: int = 200) -> GitProfiler:
        """Scan all configured repos for commit history."""
        for repo in self.repo_paths:
            if not Path(repo).joinpath(".git").exists():
                continue
            try:
                result = subprocess.run(
                    ["git", "-C", repo, "log", "--pretty=format:%s|%h|%ad", "--date=short", "-n", str(max_commits)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if "|" in line:
                            msg, hash_, date = line.split("|", 2)
                            self._commits.append({"msg": msg.strip(), "hash": hash_, "date": date, "repo": repo})
            except Exception:
                pass
        return self

    def prefix_distribution(self) -> dict[str, int]:
        """Map commit prefixes to frequency."""
        counts: Counter[str] = Counter()
        for c in self._commits:
            m = self.CONVENTIONAL_RE.match(c["msg"])
            if m:
                counts[m.group(1)] += 1
            else:
                counts["freeform"] += 1
        return dict(counts.most_common())

    def file_type_distribution(self) -> dict[str, int]:
        """What file extensions does the user touch most?"""
        ext_counts: Counter[str] = Counter()
        seen = set()
        for c in self._commits[:100]:
            repo = c["repo"]
            h = c["hash"]
            key = f"{repo}::{h}"
            if key in seen:
                continue
            seen.add(key)
            try:
                result = subprocess.run(
                    ["git", "-C", repo, "diff-tree", "--no-commit-id", "--name-only", "-r", h],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    for fname in result.stdout.strip().split("\n"):
                        if "." in fname:
                            ext = fname.rsplit(".", 1)[-1].lower()
                            ext_counts[ext] += 1
            except Exception:
                pass
        return dict(ext_counts.most_common(10))

    def infer_style(self) -> dict[str, Any]:
        """Infer user's coding habits from commit patterns."""
        if not self._commits:
            return {}

        prefixes = self.prefix_distribution()
        file_types = self.file_type_distribution()

        # Naming style inference from commit messages
        msgs = " ".join(c["msg"] for c in self._commits[:50]).lower()
        snake_score = msgs.count("_") - msgs.count("camel")
        naming = "snake_case" if snake_score > 0 else "camelCase" if "camel" in msgs else "snake_case"

        # Async style from file types
        async_score = sum(file_types.get(k, 0) for k in ("py", "ts", "js"))
        sync_score = file_types.get("sh", 0) + file_types.get("dockerfile", 0)
        async_pref = "async_first" if async_score > sync_score * 2 else "sync_first"

        return {
            "commit_prefixes": list(prefixes.keys())[:5],
            "top_file_types": list(file_types.keys())[:5],
            "naming_style": naming,
            "async_style": async_pref,
            "total_commits_analyzed": len(self._commits),
        }


class CodeProfiler:
    """Analyze codebase for style and architectural patterns."""

    def __init__(self, scan_paths: list[str]) -> None:
        self.scan_paths = scan_paths
        self._file_samples: list[tuple[str, str]] = []  # (path, content)

    def _collect_files(self, max_per_ext: int = 20) -> None:
        """Gather representative source files."""
        ext_limits: Counter[str] = Counter()
        for root in self.scan_paths:
            for path in Path(root).rglob("*"):
                if path.is_file() and path.stat().st_size < 500_000:
                    ext = path.suffix.lstrip(".").lower()
                    if ext in ("py", "ts", "js", "tsx", "jsx", "go", "rs", "java"):
                        if ext_limits[ext] < max_per_ext:
                            try:
                                content = path.read_text(encoding="utf-8", errors="ignore")
                                self._file_samples.append((str(path), content))
                                ext_limits[ext] += 1
                            except Exception:
                                pass

    def analyze(self) -> dict[str, Any]:
        """Return style and architecture fingerprints."""
        self._collect_files()
        if not self._file_samples:
            return {}

        total_lines = 0
        docstring_count = 0
        comment_count = 0
        class_count = 0
        dataclass_count = 0
        test_count = 0
        async_count = 0
        type_hint_count = 0
        import_from_count = 0
        import_direct_count = 0

        for _, content in self._file_samples:
            lines = content.split("\n")
            total_lines += len(lines)
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    comment_count += 1
                if 'async def ' in line:
                    async_count += 1
                if 'import ' in line and 'from ' in line:
                    import_from_count += 1
                elif 'import ' in line:
                    import_direct_count += 1
            if '"""' in content or "'''" in content:
                docstring_count += 1
            if 'class ' in content:
                class_count += content.count('class ')
            if 'dataclass' in content:
                dataclass_count += 1
            if 'def test_' in content or 'class Test' in content:
                test_count += content.count('def test_')
            if 'typing.' in content or ': ' in content or '-> ' in content:
                type_hint_count += 1

        return {
            "architecture_pref": "dataclass_driven" if dataclass_count > class_count // 2 else "class_driven",
            "async_style": "async_first" if async_count > len(self._file_samples) // 2 else "sync_first",
            "docstring_level": "full" if docstring_count > len(self._file_samples) // 2 else "minimal",
            "comment_density": "medium" if comment_count > total_lines * 0.05 else "low",
            "test_style": "pytest" if test_count > 0 else "none",
            "import_style": "from_import" if import_from_count > import_direct_count else "direct_import",
            "type_hinting": "heavy" if type_hint_count > 10 else "light",
            "files_analyzed": len(self._file_samples),
        }


class PreferenceModel:
    """Load and query explicit preferences from memory and instruction files."""

    def __init__(self, memory_dir: str, instruction_files: list[str] | None = None) -> None:
        self.memory_dir = Path(memory_dir)
        self.instruction_files = instruction_files or []
        self._preferences: dict[str, Any] = {}

    def load(self) -> PreferenceModel:
        """Scan memory files and instruction files for preference directives."""
        for fpath in self.memory_dir.glob("*.md"):
            try:
                text = fpath.read_text(encoding="utf-8")
                self._parse_preferences(text)
            except Exception:
                pass

        for fpath in self.instruction_files:
            p = Path(fpath)
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8")
                    self._parse_preferences(text)
                except Exception:
                    pass
        return self

    KEYWORD_PATTERNS = {
        "parallel_pref": [
            (r"aggressive(?:ly)?\s+parallel", "aggressive"),
            (r"parallelize\s+aggressively", "aggressive"),
            (r"moderate\s+parallel", "moderate"),
            (r"conservative\s+parallel", "conservative"),
        ],
        "approval_mode": [
            (r"omega\s*mode", "omega"),
            (r"approvals?\s+bypassed", "omega"),
            (r"skip\s+prompts?", "omega"),
            (r"ask\s+before", "ask"),
            (r"inform\s+(?:me|user)", "inform"),
        ],
        "architecture_pref": [
            (r"modular\s+(?:architecture|design|style)", "modular"),
            (r"(?:architecture|design|style)\s+(?:should\s+be\s+|is\s+)?modular", "modular"),
            (r"monolithic", "monolithic"),
            (r"functional\s+style", "functional"),
        ],
        "naming_style": [
            (r"snake_case", "snake_case"),
            (r"camelCase", "camelCase"),
            (r"PascalCase", "PascalCase"),
        ],
        "comment_density": [
            (r"no\s+comment", "none"),
            (r"minimal\s+comment", "low"),
            (r"(?:extensive|heavy)\s+comment", "high"),
        ],
    }

    def _parse_preferences(self, text: str) -> None:
        """Extract preference keywords from text."""
        lower = text.lower()
        for pref_key, patterns in self.KEYWORD_PATTERNS.items():
            for pattern, value in patterns:
                if re.search(pattern, lower, re.IGNORECASE):
                    self._preferences[pref_key] = value
                    break

        # Extract lists (skills, triggers)
        lines = text.split("\n")
        skills = []
        for line in lines:
            # Match table rows where last column is LOADED
            m = re.search(r"\|\s*([^|]+?)\s*\|\s*LOADED\s*\|", line)
            if m:
                name = m.group(1).strip()
                if name and name.lower() not in ("skill", "skills", "---", "name", "status"):
                    skills.append(name)
        if skills:
            self._preferences["active_skills"] = skills

        trigger_match = re.search(r"wake\s+word|voice\s+mode|trigger", lower)
        if trigger_match:
            triggers = re.findall(r'"([^"]+)"', text)
            if triggers:
                self._preferences["voice_triggers"] = triggers[:10]

    def get(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._preferences)


class MirrorTrainer:
    """Continuously refine the mirror profile from interactions."""

    def __init__(self, profile_path: str) -> None:
        self.profile_path = Path(profile_path)
        self.profile = self._load()

    def _load(self) -> MirrorProfile:
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
                return MirrorProfile(**data)
            except Exception:
                pass
        return MirrorProfile()

    def save(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(self.profile.__dict__, indent=2, default=str),
            encoding="utf-8",
        )

    def record_decision(self, context: str, predicted: str, actual: str) -> None:
        """Log a decision and update accuracy."""
        self.profile.interaction_count += 1
        correct = int(predicted.strip().lower() == actual.strip().lower())
        n = self.profile.interaction_count
        self.profile.decision_accuracy = (
            self.profile.decision_accuracy * (n - 1) + correct
        ) / n
        self.profile.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()

    def update_from_analysis(self, git_analysis: dict[str, Any], code_analysis: dict[str, Any], prefs: dict[str, Any]) -> None:
        """Merge new analysis into the profile."""
        for source in (git_analysis, code_analysis, prefs):
            for k, v in source.items():
                if hasattr(self.profile, k):
                    setattr(self.profile, k, v)
        self.profile.last_updated = datetime.now(timezone.utc).isoformat()
        self.save()


class MirrorAgent:
    """Agent wrapper that makes decisions aligned with the user's learned profile."""

    DECISION_TEMPLATES = {
        "commit_prefix": (
            "Given the change described below, what conventional commit prefix would {user} use?\n"
            "Options: feat, fix, chore, docs, test, refactor, build, ci\n\n"
            "Change: {change}\n\n"
            "Prefix:"
        ),
        "architecture_choice": (
            "{user} is choosing between these two approaches. Based on their preference for "
            "{arch_pref} architecture with {async_style} async style, which would they pick?\n"
            "A: {option_a}\nB: {option_b}\n\nChoice (A/B):"
        ),
        "parallelization": (
            "{user} has {n} independent sub-tasks. Their parallel preference is {parallel_pref}.\n"
            "Should they run sequentially, or in parallel?\n\nDecision:"
        ),
        "approval_needed": (
            "{user}'s approval mode is {approval_mode}.\n"
            "Action: {action}\n"
            "Should the assistant ask for explicit approval, inform after doing it, or just do it?\n"
            "Decision (ask/inform/just_do):"
        ),
    }

    def __init__(self, trainer: MirrorTrainer | None = None) -> None:
        self.trainer = trainer or MirrorTrainer(self._default_profile_path())
        self.profile = self.trainer.profile

    @staticmethod
    def _default_profile_path() -> str:
        config = load_config()
        return str(Path(config.memory_dir) / "mirror_profile.json")

    def predict(self, decision_type: str, **context: Any) -> str:
        """Predict what the user would decide. Uses profile heuristics, no LLM call."""
        if decision_type == "commit_prefix":
            return self._predict_commit_prefix(context.get("change", ""))
        if decision_type == "architecture_choice":
            return self._predict_architecture(context.get("option_a", ""), context.get("option_b", ""))
        if decision_type == "parallelization":
            return self._predict_parallelization(context.get("n", 1))
        if decision_type == "approval_needed":
            return self._predict_approval(context.get("action", ""))
        return "unknown"

    def _predict_commit_prefix(self, change: str) -> str:
        change_lower = change.lower()
        prefixes = self.profile.commit_prefixes or ["feat", "fix", "chore", "docs", "test"]
        scores: dict[str, int] = {}
        for p in prefixes:
            scores[p] = 0

        keyword_map = {
            "feat": ["add ", "implement ", "new feature", "support", "introduce"],
            "fix": ["bug", "broken", "error", "crash", "resolve", "repair"],
            "chore": ["update", "bump", "clean", "ignore", "format", "lint"],
            "docs": ["document", "readme", "comment", "guide", "tutorial"],
            "test": ["test", "tests", "spec", "coverage", "pytest", "assert"],
            "refactor": ["refactor", "rewrite", "restructure", "simplify"],
            "build": ["docker", "deploy", "package", "install", "build "],
            "ci": ["github action", "workflow", "pipeline", "ci/cd"],
        }

        for prefix, keywords in keyword_map.items():
            if prefix in scores:
                scores[prefix] = sum(1 for kw in keywords if kw in change_lower)

        return max(scores, key=scores.get) if scores else "chore"

    def _predict_architecture(self, option_a: str, option_b: str) -> str:
        arch = self.profile.architecture_pref or "modular"
        a_lower = option_a.lower()
        b_lower = option_b.lower()
        a_modular = sum(1 for w in ("module", "plugin", "separate", "bridge", "interface") if w in a_lower)
        b_modular = sum(1 for w in ("module", "plugin", "separate", "bridge", "interface") if w in b_lower)
        if arch == "modular":
            return "A" if a_modular >= b_modular else "B"
        if arch == "monolithic":
            return "A" if a_modular <= b_modular else "B"
        return "A"

    def _predict_parallelization(self, n: int) -> str:
        pref = self.profile.parallel_pref or "moderate"
        if pref == "aggressive" and n > 1:
            return "parallel"
        if pref == "conservative" or n <= 1:
            return "sequential"
        return "parallel" if n > 2 else "sequential"

    def _predict_approval(self, action: str) -> str:
        mode = self.profile.approval_mode or "inform"
        if mode == "omega":
            return "just_do"
        if mode == "inform":
            return "inform"
        return "ask"

    def build_prompt(self, decision_type: str, **context: Any) -> str:
        """Build a decision prompt string (useful for external LLM queries)."""
        template = self.DECISION_TEMPLATES.get(decision_type, "")
        if not template:
            return ""
        fmt = {
            "user": self.profile.user_name,
            "arch_pref": self.profile.architecture_pref,
            "async_style": self.profile.async_style,
            "parallel_pref": self.profile.parallel_pref,
            "approval_mode": self.profile.approval_mode,
        }
        fmt.update(context)
        return template.format(**fmt)

    def record_outcome(self, context: str, predicted: str, actual: str) -> None:
        """Feed back the actual user decision to improve accuracy."""
        self.trainer.record_decision(context, predicted, actual)
        self.profile = self.trainer.profile


def build_mirror(profile_path: str | None = None) -> MirrorAgent:
    """Factory: scan repos, analyze code, load preferences, and return a trained MirrorAgent."""
    config = load_config()
    ppath = profile_path or str(Path(config.memory_dir) / "mirror_profile.json")

    home = Path.home()
    repo_candidates = [
        home / "kimi-core",
        home / "agency-agents",
        home / "friday-os",
        home / "goose-aios",
        home / "sovereign-swarm-v2",
    ]
    repo_paths = [str(p) for p in repo_candidates if p.joinpath(".git").exists()]

    git = GitProfiler(repo_paths).scan()
    git_analysis = git.infer_style()

    code = CodeProfiler([str(p) for p in repo_candidates if p.exists()])
    code_analysis = code.analyze()

    mem_dir = config.memory_dir
    instruction_files = [
        home / ".claude" / "CLAUDE.md",
        home / "AGENTS.md",
    ]
    prefs = PreferenceModel(mem_dir, [str(f) for f in instruction_files if f.exists()]).load().to_dict()

    trainer = MirrorTrainer(ppath)
    trainer.update_from_analysis(git_analysis, code_analysis, prefs)
    return MirrorAgent(trainer)


if __name__ == "__main__":
    agent = build_mirror()
    print(json.dumps(agent.profile.__dict__, indent=2, default=str))

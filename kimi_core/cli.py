"""CLI entry point for Kimi-Core."""

from __future__ import annotations

import argparse
import sys

from kimi_core.agent import KimiCore
from kimi_core.swarm import Swarm


def _cmd_run(args: argparse.Namespace) -> int:
    agent = KimiCore()

    if args.task:
        print(f">>> {args.task}")
        result = agent.run(args.task)
        print(result)
        return 0

    if args.repl:
        print("Kimi-Core REPL. Type 'exit' to quit.")
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() in ("exit", "quit"):
                    break
                result = agent.run(user_input)
                print(result)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        return 0

    return 1


def _cmd_swarm(args: argparse.Namespace) -> int:
    swarm = Swarm()

    if args.decompose:
        print(f"[swarm] decomposing: {args.decompose}")
        result = swarm.run_decompose(args.decompose)
        print(result)
        return 0

    if args.parallel:
        tasks = [t.strip() for t in args.parallel.split(",") if t.strip()]
        print(f"[swarm] parallel tasks: {tasks}")
        from kimi_core.swarm import AgentRole, WORKER_ROLE
        results = swarm.run_parallel([(WORKER_ROLE, t) for t in tasks])
        for t, r in zip(tasks, results):
            print(f"--- {t} ---")
            print(r)
        return 0

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-Core: Self-hosted AI assistant")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run single agent (default)")
    run_parser.add_argument("--task", type=str, help="Single task to execute")
    run_parser.add_argument("--repl", action="store_true", help="Interactive REPL mode")
    run_parser.add_argument("--config", type=str, help="Path to agent-config.yaml")

    swarm_parser = subparsers.add_parser("swarm", help="Run multi-agent swarm")
    swarm_parser.add_argument("--decompose", type=str, help="Task to decompose into subtasks")
    swarm_parser.add_argument("--parallel", type=str, help="Comma-separated tasks to run in parallel")

    args = parser.parse_args()

    if args.command == "swarm":
        return _cmd_swarm(args)

    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())

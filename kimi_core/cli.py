"""CLI entry point for Kimi-Core."""

from __future__ import annotations

import argparse
import sys

from kimi_core.agent import KimiCore


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimi-Core: Self-hosted AI assistant")
    parser.add_argument("--task", type=str, help="Single task to execute")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--config", type=str, help="Path to agent-config.yaml")
    args = parser.parse_args()

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

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

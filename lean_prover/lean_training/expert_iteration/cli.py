"""CLI for full or single-stage expert iteration execution."""

from __future__ import annotations

import argparse
import json
import signal

from .config import load_expert_iteration_config
from .orchestrator import ExpertIterationOrchestrator
from .schemas import IterationStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Lean expert iteration.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in IterationStage],
        default=None,
    )
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_expert_iteration_config(args.config)
    orchestrator = ExpertIterationOrchestrator(config)
    previous_handlers = {}

    def request_shutdown(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            target = getattr(signal, name)
            previous_handlers[target] = signal.getsignal(target)
            signal.signal(target, request_shutdown)
    try:
        if args.dry_run:
            result = orchestrator.dry_run()
        else:
            stage = IterationStage(args.stage) if args.stage else None
            if (
                stage is not None
                and stage is not IterationStage.BENCHMARK
                and args.iteration is None
            ):
                raise ValueError("single iteration stage requires --iteration N")
            result = orchestrator.run(
                stage=stage,
                iteration=args.iteration,
                resume=args.resume,
                force=args.force,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        orchestrator.close()
        for target, previous in previous_handlers.items():
            signal.signal(target, previous)


if __name__ == "__main__":
    main()

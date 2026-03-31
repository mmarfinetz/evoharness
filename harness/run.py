#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from harness.adapters import get_adapter
from harness.db import init_db, record_run_meta
from harness.evolve import evolve_loop
from harness.utils import default_claude_bin, default_codex_bin, detect_gpu_count, log
from harness.workspaces import ensure_repo_root, make_run_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND_REPO = ROOT / "examples" / "command_repo_demo" / "repo"
DEFAULT_COMMAND_REPO_CONFIG = ROOT / "examples" / "command_repo_demo" / "adapter.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Population-based evolutionary harness for adapter-driven code optimization.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help=(
            "Path to the target repo. When omitted, command_repo uses the bundled "
            "demo if present."
        ),
    )
    parser.add_argument(
        "--adapter",
        "--objective",
        default="command_repo",
        help="Adapter/objective implementation to run. Defaults to command_repo.",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=None,
        help=(
            "Optional adapter-specific config path. command_repo uses the bundled "
            "demo config when omitted from a source checkout."
        ),
    )
    parser.add_argument(
        "--population-size",
        "--agents",
        dest="population_size",
        type=int,
        default=None,
        help="Target population size. `--agents` remains as a compatibility alias.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum concurrent candidate sessions and benchmarks.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs available to GPU-requiring adapters. CPU-only adapters may use 0.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=None,
        help="Number of elite survivors carried forward unchanged each generation.",
    )
    parser.add_argument(
        "--selection-strategy",
        default="tournament",
        choices=("tournament", "rank"),
        help="Parent selection strategy for offspring generation.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.8,
        help="Probability of a non-crossover child using mutate mode instead of repair mode.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.3,
        help="Probability that a child is created from two parents in crossover mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed for parent selection and operator planning.",
    )
    parser.add_argument(
        "--agent-runner",
        default="claude",
        choices=("claude", "codex"),
        help="Agent CLI used for candidate sessions.",
    )
    parser.add_argument(
        "--claude-bin",
        "--agent-bin",
        dest="agent_bin",
        default=None,
        help="Agent executable. Defaults to the selected runner binary.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional runner model alias or full model name.",
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=("low", "medium", "high", "xhigh"),
        help="Reasoning effort level for runners that support it.",
    )
    parser.add_argument(
        "--agent-timeout-minutes",
        type=int,
        default=120,
        help="Maximum runtime for each agent session before the harness stops it.",
    )
    parser.add_argument(
        "--benchmark-timeout-minutes",
        "--train-timeout-minutes",
        dest="benchmark_timeout_minutes",
        type=int,
        default=10,
        help="Timeout for the adapter benchmark command of each finalized candidate state.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Optional cap on generations. Defaults to running until interrupted.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the timestamped run folder name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the run directory, DB, baseline snapshot, and generation-one workspaces, then stop.",
    )
    return parser.parse_args()


def setup_args(args: argparse.Namespace) -> tuple[argparse.Namespace, object]:
    adapter = get_adapter(args.adapter)

    if args.repo is None:
        if args.adapter == "command_repo" and DEFAULT_COMMAND_REPO.exists():
            args.repo = DEFAULT_COMMAND_REPO
        else:
            raise SystemExit(f"--repo is required with --adapter {args.adapter}")

    if args.adapter_config is None and args.adapter == "command_repo":
        if DEFAULT_COMMAND_REPO_CONFIG.exists():
            args.adapter_config = DEFAULT_COMMAND_REPO_CONFIG

    if args.agent_bin is None:
        if args.agent_runner == "codex":
            args.agent_bin = default_codex_bin()
        else:
            args.agent_bin = default_claude_bin()

    args.repo = args.repo.resolve()
    if not args.repo.exists():
        raise SystemExit(f"--repo does not exist: {args.repo}")
    if args.repo.is_file():
        raise SystemExit(f"--repo must point to a directory: {args.repo}")

    if args.adapter_config is not None:
        args.adapter_config = args.adapter_config.resolve()
        if not args.adapter_config.exists():
            raise SystemExit(f"--adapter-config does not exist: {args.adapter_config}")
        if args.adapter_config.is_dir():
            raise SystemExit(f"--adapter-config must point to a file: {args.adapter_config}")

    adapter.validate_args(args)

    detected_gpus = detect_gpu_count()
    if adapter.resources.requires_gpu:
        if args.gpus is None:
            if detected_gpus < 1:
                raise SystemExit(
                    f"adapter '{adapter.name}' requires at least one GPU, but none were detected"
                )
            args.gpus = detected_gpus
        if args.gpus < 1:
            raise SystemExit(
                f"adapter '{adapter.name}' requires --gpus >= 1, received {args.gpus}"
            )
        if detected_gpus and args.gpus > detected_gpus:
            log(
                f"--gpus={args.gpus} exceeds detected GPUs ({detected_gpus}), setting --gpus to detected"
            )
            args.gpus = detected_gpus
    else:
        if args.gpus is None:
            args.gpus = 0
        if args.gpus < 0:
            raise SystemExit("--gpus must be zero or positive")

    if args.workers is None:
        args.workers = args.gpus if adapter.resources.requires_gpu else 1
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if adapter.resources.requires_gpu and args.workers > args.gpus:
        raise SystemExit(
            f"adapter '{adapter.name}' requires one GPU per active worker; "
            f"received --workers={args.workers} with --gpus={args.gpus}"
        )

    if args.population_size is None:
        args.population_size = args.workers
    if args.population_size < 1:
        raise SystemExit("--population-size must be at least 1")

    if args.elite_count is None:
        args.elite_count = 0 if args.population_size == 1 else 1
    if args.elite_count < 0:
        raise SystemExit("--elite-count must be zero or positive")
    if args.elite_count >= args.population_size:
        raise SystemExit("--elite-count must be smaller than --population-size")

    if not 0.0 <= args.mutation_rate <= 1.0:
        raise SystemExit("--mutation-rate must be between 0 and 1")
    if not 0.0 <= args.crossover_rate <= 1.0:
        raise SystemExit("--crossover-rate must be between 0 and 1")
    return args, adapter


def main() -> int:
    args, adapter = setup_args(parse_args())
    repo = args.repo.resolve()
    ensure_repo_root(repo)
    adapter.validate_repo(repo, args)

    run_name, run_dir, workspaces_root = make_run_dir(args)

    db_path = run_dir / "results.db"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        init_db(conn)
        record_run_meta(conn, run_name, repo, args, adapter.objective)
        return evolve_loop(conn, repo, workspaces_root, run_name, run_dir, args, adapter)


if __name__ == "__main__":
    raise SystemExit(main())

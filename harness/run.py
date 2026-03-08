#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from harness.db import init_db, record_run_meta
from harness.evolve import evolve_loop
from harness.utils import default_claude_bin, detect_gpu_count, log
from harness.workspaces import ensure_repo, make_run_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "autoresearch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal harness for running Claude-driven autoresearch experiments.",
    )
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_REPO, help="Path to the autoresearch repo."
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Number of Claude candidates per generation. Defaults to the detected GPU count.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to assign across workers. Defaults to the detected GPU count.",
    )
    parser.add_argument(
        "--claude-bin", default=default_claude_bin(), help="Claude Code executable."
    )
    parser.add_argument(
        "--model", default=None, help="Optional Claude model alias or full model name."
    )
    parser.add_argument(
        "--effort",
        default="medium",
        choices=("low", "medium", "high"),
        help="Claude reasoning effort level.",
    )
    parser.add_argument(
        "--agent-timeout-minutes",
        type=int,
        default=120,
        help="Maximum runtime for each Claude agent before the harness stops it.",
    )
    parser.add_argument(
        "--train-timeout-minutes",
        type=int,
        default=10,
        help="Timeout for the harness-run benchmark of each final candidate state.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Optional cap on generations. Defaults to running until interrupted.",
    )
    parser.add_argument(
        "--run-name", default=None, help="Override the timestamped run folder name."
    )
    parser.add_argument(
        "--baseline-train-py",
        type=Path,
        default=None,
        help="Optional train.py file to use as the generation-1 baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the run directory, database, and generation-one worktrees, then stop.",
    )
    return parser.parse_args()


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.baseline_train_py is not None:
        args.baseline_train_py = args.baseline_train_py.resolve()
        if not args.baseline_train_py.exists():
            raise SystemExit(
                f"--baseline-train-py does not exist: {args.baseline_train_py}"
            )
        if args.baseline_train_py.is_dir():
            raise SystemExit(
                f"--baseline-train-py must point to a file: {args.baseline_train_py}"
            )
    detected_gpus = detect_gpu_count()
    if args.gpus is None:
        if detected_gpus < 1:
            raise SystemExit(
                "No GPUs detected. Make sure NVIDIA GPUs are visible or pass --gpus explicitly."
            )
        args.gpus = detected_gpus
    if detected_gpus and args.gpus > detected_gpus:
        log(
            f"--gpus={args.gpus} exceeds detected GPUs ({detected_gpus}), setting --gpus to detected"
        )
        args.gpus = detected_gpus
    if args.agents is None:
        args.agents = args.gpus
    if args.agents < 1:
        raise SystemExit("--agents must be at least 1")
    return args


def main() -> int:
    args = setup_args(parse_args())
    repo = args.repo.resolve()
    ensure_repo(repo)

    run_name, run_dir, workspaces_root = make_run_dir(args)

    db_path = run_dir / "results.db"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        init_db(conn)
        record_run_meta(conn, run_name, repo, args)
        return evolve_loop(conn, repo, workspaces_root, run_name, run_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())

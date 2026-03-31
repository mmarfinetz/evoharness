#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "harness" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete older harness run directories while keeping the newest entries."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Runs directory to prune. Default: {DEFAULT_RUNS_ROOT}",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=8,
        help="Number of newest run directories to keep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the directories that would be removed without deleting them.",
    )
    return parser.parse_args()


def iter_run_dirs(runs_root: Path) -> list[Path]:
    return sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def main() -> int:
    args = parse_args()
    if args.keep < 0:
        raise SystemExit("--keep must be >= 0")
    if not args.runs_root.exists():
        raise SystemExit(f"runs root does not exist: {args.runs_root}")
    if not args.runs_root.is_dir():
        raise SystemExit(f"runs root is not a directory: {args.runs_root}")

    run_dirs = iter_run_dirs(args.runs_root)
    victims = run_dirs[args.keep :]

    print(f"runs_root={args.runs_root}")
    print(f"total_runs={len(run_dirs)} keep={args.keep} prune={len(victims)}")
    for path in victims:
        print(path)

    if args.dry_run:
        return 0

    for path in victims:
        try:
            shutil.rmtree(path, onerror=ignore_missing)
        except FileNotFoundError:
            continue
    return 0


def ignore_missing(function, path, excinfo) -> None:  # type: ignore[no-untyped-def]
    _, error, _ = excinfo
    if isinstance(error, FileNotFoundError):
        return
    raise error


if __name__ == "__main__":
    raise SystemExit(main())

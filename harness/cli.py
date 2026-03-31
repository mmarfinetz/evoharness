from __future__ import annotations

import sys
from contextlib import contextmanager

from harness import __version__
from harness.run import DEFAULT_COMMAND_REPO, DEFAULT_COMMAND_REPO_CONFIG


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _run([])

    command = args[0]
    if command == "run":
        return _run(args[1:])
    if command == "demo":
        return _show_demo()
    if command in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    if command in {"-V", "--version", "version"}:
        print(__version__)
        return 0

    return _run(args)


def _run(args: list[str]) -> int:
    from harness.run import main as run_main

    with patched_argv(["evo-harness", *args]):
        return run_main()


def _show_demo() -> int:
    if DEFAULT_COMMAND_REPO.exists() and DEFAULT_COMMAND_REPO_CONFIG.exists():
        print("Bundled command_repo demo:")
        print(f"  repo:   {DEFAULT_COMMAND_REPO}")
        print(f"  config: {DEFAULT_COMMAND_REPO_CONFIG}")
        print()
        print("Example:")
        print("  evo-harness run --dry-run --population-size 2 --workers 1 --gpus 0")
        return 0

    print(
        "No bundled command_repo demo is available in this installation.\n"
        "Pass --repo and --adapter-config explicitly to evo-harness run."
    )
    return 1


def _usage() -> str:
    return (
        "Usage:\n"
        "  evo-harness run [args...]\n"
        "  evo-harness demo\n"
        "  evo-harness --version\n"
        "\n"
        "If you omit the subcommand, arguments are forwarded to `run`."
    )


@contextmanager
def patched_argv(argv: list[str]):
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous

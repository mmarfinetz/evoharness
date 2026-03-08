from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from harness.claude import (
    CandidateSession,
    ClaudeOptions,
    finalize_session_processes,
    run_candidate_sessions,
)
from harness.db import (
    finish_generation,
    get_candidate_agent_status,
    mark_generation_winner,
    record_generation,
    select_generation_winner,
    update_candidate,
)
from harness.utils import build_gpu_env, default_uv_bin, format_log_line, log
from harness.utils import terminate_gpu_processes, terminate_workspace_processes, wait_for_gpu_idle
from harness.workspaces import WorktreeSpec, create_generation_worktrees

METRIC_NAMES = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)
TIMEOUT_SUMMARY = "Harness stopped the agent at the time limit."


def evolve_loop(
    conn: sqlite3.Connection,
    repo: Path,
    worktrees_root: Path,
    run_name: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> int:
    state_dir = run_dir / "state"
    state_dir.mkdir(exist_ok=True)
    parent_train_path = initialize_parent_train(repo, state_dir, args.baseline_train_py)

    generation = 1
    while args.max_generations is None or generation <= args.max_generations:
        specs = create_generation_worktrees(
            repo,
            worktrees_root,
            parent_train_path,
            generation,
            args.agents,
            args.gpus,
        )
        record_generation(conn, generation, parent_train_path, specs)

        log(f"run_dir={run_dir}")
        log(f"generation={generation}")
        log(f"parent_train={parent_train_path}")
        for spec in specs:
            log(f"workspace {spec.name} gpu={spec.gpu_id}: {spec.path}")

        if args.dry_run:
            return 0

        run_generation_agents(conn, specs, args)
        benchmark_generation(conn, specs, args)
        winner = promote_winner(conn, generation, parent_train_path, state_dir)
        finish_generation(conn, generation, winner)
        log(
            f"generation={generation} promoted={winner['name']} "
            f"train={winner['winner_train_path']} val_bpb={winner['val_bpb']:.6f}"
        )
        parent_train_path = Path(winner["winner_train_path"])
        generation += 1

    return 0


def initialize_parent_train(
    repo: Path, state_dir: Path, baseline_train_path: Path | None
) -> Path:
    source_path = baseline_train_path or (repo / "train.py")
    parent_train_path = state_dir / "g0000-parent-train.py"
    shutil.copyfile(source_path, parent_train_path)
    return parent_train_path


def run_generation_agents(
    conn: sqlite3.Connection, specs: list[WorktreeSpec], args: argparse.Namespace
) -> None:
    candidates = [spec for spec in specs if spec.kind == "candidate"]
    options = ClaudeOptions(
        claude_bin=args.claude_bin,
        effort=args.effort,
        timeout_minutes=args.agent_timeout_minutes,
        model=args.model,
    )

    sessions: list[CandidateSession] = []
    try:
        sessions = run_candidate_sessions(candidates, options)
    finally:
        finalize_candidate_sessions(conn, sessions)


def finalize_candidate_sessions(
    conn: sqlite3.Connection, sessions: list[CandidateSession]
) -> None:
    finalize_session_processes(sessions)
    for session in sessions:
        outcome = evaluate_candidate_session(session)
        update_candidate(conn, session.spec.generation, session.spec.name, **outcome)
        log(
            f"agent generation={session.spec.generation} "
            f"{session.spec.name}: {outcome['agent_status']}"
        )


def evaluate_candidate_session(session: CandidateSession) -> dict[str, object]:
    spec = session.spec
    summary = first_tail_text(
        spec.claude_log_path,
        spec.claude_stderr_path,
        spec.claude_debug_log_path,
        spec.claude_status_path,
    )
    if not has_benchmark_candidate_state(spec):
        status = "timed_out_no_change" if session.timed_out else "no_change"
        return {"agent_status": status, "agent_summary": summary}
    return {
        "agent_status": "ready",
        "agent_summary": timeout_summary(session, summary),
        "train_sha256": train_sha256(spec.path / "train.py"),
    }


def timeout_summary(session: CandidateSession, summary: str) -> str:
    if session.timed_out and not summary:
        return TIMEOUT_SUMMARY
    return summary


def benchmark_generation(
    conn: sqlite3.Connection, specs: list[WorktreeSpec], args: argparse.Namespace
) -> None:
    drain_candidate_runtime(specs)
    for spec in specs:
        agent_status = get_candidate_agent_status(conn, spec.generation, spec.name)
        if spec.kind == "candidate" and agent_status != "ready":
            update_candidate(conn, spec.generation, spec.name, benchmark_status="skipped")
            continue

        outcome = benchmark_worktree(spec, args)
        update_candidate(conn, spec.generation, spec.name, **outcome)
        suffix = ""
        if outcome.get("val_bpb") is not None:
            suffix = f" val_bpb={outcome['val_bpb']:.6f}"
        log(
            f"benchmark generation={spec.generation} "
            f"{spec.name}: {outcome['benchmark_status']}{suffix}"
        )


def benchmark_worktree(spec: WorktreeSpec, args: argparse.Namespace) -> dict[str, object]:
    wait_for_workspace_gpu(spec)
    benchmark_source = materialize_benchmark_source(spec)
    compile_check = subprocess.run(
        [sys.executable, "-m", "py_compile", "train.py"],
        cwd=spec.path,
        capture_output=True,
        check=False,
        text=True,
    )
    if compile_check.returncode != 0:
        append_line(
            spec.train_log_path,
            format_log_line("[HARNESS] py_compile failed") + "\n",
        )
        append_line(spec.train_log_path, compile_check.stdout + compile_check.stderr)
        return benchmark_outcome("compile_failed", spec)

    with spec.train_log_path.open("w") as log_file:
        log_file.write(
            format_log_line(f"[HARNESS] benchmark start gpu={spec.gpu_id}") + "\n"
        )
        if benchmark_source != "train.py":
            log_file.write(
                format_log_line(f"[HARNESS] evaluating {benchmark_source} via train.py")
                + "\n"
            )
        try:
            result = subprocess.run(
                [default_uv_bin(), "run", "train.py"],
                cwd=spec.path,
                env=build_gpu_env(spec.gpu_id),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.train_timeout_minutes * 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            append_line(
                spec.train_log_path,
                format_log_line("[HARNESS] benchmark timed out") + "\n",
            )
            terminate_workspace_processes(spec.path)
            wait_for_workspace_gpu(spec)
            return benchmark_outcome("timeout", spec)

    metrics = parse_metrics(spec.train_log_path)
    terminate_workspace_processes(spec.path)
    wait_for_workspace_gpu(spec)
    if result.returncode != 0 or metrics["val_bpb"] is None:
        return benchmark_outcome("failed", spec, metrics)
    return benchmark_outcome("ok", spec, metrics)


def benchmark_outcome(
    benchmark_status: str,
    spec: WorktreeSpec,
    metrics: dict[str, float | None] | None = None,
) -> dict[str, object]:
    metrics = metrics or {}
    return {
        "benchmark_status": benchmark_status,
        "training_seconds": metrics.get("training_seconds"),
        "total_seconds": metrics.get("total_seconds"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "val_bpb": metrics.get("val_bpb"),
        "train_sha256": train_sha256(spec.path / "train.py"),
    }


def promote_winner(
    conn: sqlite3.Connection,
    generation: int,
    parent_train_path: Path,
    state_dir: Path,
) -> dict[str, object]:
    winner = select_generation_winner(conn, generation)
    if winner is None:
        raise SystemExit(
            f"generation {generation}: no successful benchmark completed; nothing to promote"
        )

    winner_train_path = Path(winner["worktree_path"]) / "train.py"
    next_parent_train_path = state_dir / f"g{generation:04d}-parent-train.py"
    shutil.copyfile(winner_train_path, next_parent_train_path)
    mark_generation_winner(conn, generation, winner["name"])
    return {
        "name": winner["name"],
        "val_bpb": winner["val_bpb"],
        "winner_train_path": str(next_parent_train_path),
        "previous_parent_train_path": str(parent_train_path),
    }


def parse_metrics(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text() if log_path.exists() else ""
    metrics: dict[str, float | None] = {name: None for name in METRIC_NAMES}
    for name in METRIC_NAMES:
        match = re.search(
            rf"^{re.escape(name)}:\s+([0-9.]+)$",
            text,
            flags=re.MULTILINE,
        )
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def first_tail_text(*paths: Path, limit: int = 20) -> str:
    for path in paths:
        text = tail_text(path, limit=limit)
        if text:
            return text
    return ""


def tail_text(path: Path, limit: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    for line in reversed(lines[-limit:]):
        text = line.strip()
        if text:
            return text[:300]
    return ""


def append_line(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text)


def materialize_benchmark_source(spec: WorktreeSpec) -> str:
    baseline_bytes = (spec.path / "baseline.py").read_bytes()
    best_path = spec.path / "best.py"
    if not best_path.exists():
        return "train.py"

    best_bytes = best_path.read_bytes()
    if best_bytes == baseline_bytes:
        return "train.py"

    train_path = spec.path / "train.py"
    if not train_path.exists() or train_path.read_bytes() != best_bytes:
        shutil.copyfile(best_path, train_path)
    return "best.py"


def has_benchmark_candidate_state(spec: WorktreeSpec) -> bool:
    baseline_bytes = (spec.path / "baseline.py").read_bytes()
    train_path = spec.path / "train.py"
    best_path = spec.path / "best.py"
    train_changed = train_path.exists() and train_path.read_bytes() != baseline_bytes
    best_changed = best_path.exists() and best_path.read_bytes() != baseline_bytes
    return train_changed or best_changed


def drain_candidate_runtime(specs: list[WorktreeSpec]) -> None:
    for spec in specs:
        if spec.kind != "candidate":
            continue
        terminate_workspace_processes(spec.path)
        terminate_gpu_processes(spec.gpu_id)
    waited_gpu_ids: set[int] = set()
    for spec in specs:
        if spec.gpu_id in waited_gpu_ids:
            continue
        wait_for_workspace_gpu(spec)
        waited_gpu_ids.add(spec.gpu_id)


def wait_for_workspace_gpu(spec: WorktreeSpec) -> None:
    if not wait_for_gpu_idle(spec.gpu_id):
        append_line(
            spec.train_log_path,
            format_log_line(
                f"[HARNESS] GPU {spec.gpu_id} still busy after cleanup; proceeding anyway"
            )
            + "\n",
        )


def train_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

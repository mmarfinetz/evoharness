from __future__ import annotations

import argparse
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, TypeVar

from harness.adapters.base import ObjectiveAdapter
from harness.claude import (
    CandidateSession,
    ClaudeOptions,
    finalize_session_processes,
    run_candidate_sessions,
)
from harness.db import (
    finish_generation,
    insert_candidate,
    record_generation_start,
    record_population_membership,
    update_candidate,
)
from harness.models import (
    BenchmarkResult,
    CandidateSnapshot,
    CandidateStateDetection,
    PopulationMember,
    ValidationResult,
    WorkspaceSpec,
)
from harness.selection import rank_population, select_parent, select_parent_pair, select_survivors
from harness.snapshots import create_snapshot_from_paths
from harness.utils import log
from harness.workspaces import generation_workspace_dir, initialize_workspace

TIMEOUT_SUMMARY = "Harness stopped the agent at the time limit."


def evolve_loop(
    conn: sqlite3.Connection,
    repo: Path,
    workspaces_root: Path,
    run_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    adapter: ObjectiveAdapter,
) -> int:
    del run_name
    state_dir = run_dir / "state"
    snapshots_root = state_dir / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)

    seed_snapshot = adapter.create_initial_snapshot(repo, snapshots_root, args)
    seed_member = PopulationMember(
        candidate_id="seed-baseline",
        generation=0,
        name="seed-baseline",
        snapshot=seed_snapshot,
        operator="seed",
    )
    insert_candidate(
        conn,
        candidate_id=seed_member.candidate_id,
        generation=0,
        name=seed_member.name,
        mode="seed",
        operator="seed",
        worktree_path=None,
        worker_slot=0,
        gpu_id=0 if args.gpus else None,
        parent_a=None,
        parent_b=None,
        source_snapshot_path=seed_snapshot.path,
        claude_log_path=None,
        validation_log_path=None,
        benchmark_log_path=None,
        notes_path=seed_snapshot.notes_path,
    )
    update_candidate(
        conn,
        seed_member.candidate_id,
        snapshot_path=seed_snapshot.path,
        snapshot_hash=seed_snapshot.artifact_hash,
        artifact_paths_json=seed_snapshot.artifact_paths,
    )

    current_population = [seed_member]
    if not args.dry_run:
        current_population = [benchmark_seed_member(conn, repo, workspaces_root, args, adapter, seed_member)]
        if not current_population[0].valid:
            raise SystemExit(
                f"baseline seed benchmark failed with status {current_population[0].benchmark_status}; "
                "cannot enter the evolutionary loop without one valid parent"
            )

    rng = random.Random(args.seed)
    generation = 1
    while args.max_generations is None or generation <= args.max_generations:
        record_generation_start(conn, generation, args, adapter.objective)
        elite_parents = select_elite_parents(current_population, args.elite_count)
        record_population_membership(
            conn,
            generation,
            "start",
            [member.candidate_id for member in current_population],
            elite_ids={member.candidate_id for member in elite_parents},
        )

        generation_summary = population_summary(current_population, adapter.objective.name)
        planned_candidates = plan_generation(
            generation=generation,
            current_population=current_population,
            elite_parents=elite_parents,
            workspaces_root=workspaces_root,
            repo=repo,
            adapter=adapter,
            generation_summary=generation_summary,
            args=args,
            rng=rng,
        )
        register_planned_candidates(conn, planned_candidates)

        log(f"run_dir={run_dir}")
        log(f"generation={generation}")
        log(f"population={len(current_population)} elites={len(elite_parents)} offspring={len(planned_candidates)}")
        for candidate_id, spec, parent_a, parent_b in planned_candidates:
            parent_text = parent_a.candidate_id
            if parent_b is not None:
                parent_text = f"{parent_text}+{parent_b.candidate_id}"
            log(
                f"workspace {candidate_id} mode={spec.mode} worker={spec.worker_slot} "
                f"gpu={spec.gpu_id} parents={parent_text}: {spec.path}"
            )

        if args.dry_run:
            return 0

        offspring_members = run_generation_candidates(
            conn,
            planned_candidates,
            snapshots_root,
            adapter,
            args,
        )
        combined_population = current_population + offspring_members
        elite_ids = {member.candidate_id for member in elite_parents}
        survivors = select_survivors(
            combined_population,
            target_size=args.population_size,
            elite_ids=elite_ids,
        )
        if not survivors:
            raise SystemExit(
                f"generation {generation}: no valid candidates survived constraints and objective ranking"
            )

        for survivor in survivors:
            update_candidate(
                conn,
                survivor.candidate_id,
                promoted=True,
                elite=survivor.candidate_id in elite_ids,
            )

        best_candidate = rank_population(survivors)[0]
        record_population_membership(
            conn,
            generation,
            "end",
            [member.candidate_id for member in survivors],
            elite_ids={member.candidate_id for member in survivors if member.candidate_id in elite_ids},
        )
        finish_generation(
            conn,
            generation,
            best_candidate_id=best_candidate.candidate_id,
            best_objective_value=best_candidate.objective_value,
            best_snapshot_path=best_candidate.snapshot.path,
        )
        log(
            f"generation={generation} best={best_candidate.candidate_id} "
            f"{adapter.objective.name}={best_candidate.objective_value:.6f} "
            f"survivors={len(survivors)}"
        )
        current_population = survivors
        generation += 1

    return 0


def benchmark_seed_member(
    conn: sqlite3.Connection,
    repo: Path,
    workspaces_root: Path,
    args: argparse.Namespace,
    adapter: ObjectiveAdapter,
    seed_member: PopulationMember,
) -> PopulationMember:
    record_generation_start(conn, 0, args, adapter.objective)
    record_population_membership(conn, 0, "start", [seed_member.candidate_id], elite_ids=set())
    generation_dir = generation_workspace_dir(workspaces_root, 0)
    spec = WorkspaceSpec(
        generation=0,
        index=0,
        name="seed-baseline",
        mode="seed",
        operator="seed",
        path=generation_dir / "seed-baseline",
        worker_slot=0,
        gpu_id=0 if args.gpus else None,
    )
    initialize_workspace(
        repo=repo,
        spec=spec,
        adapter=adapter,
        seed_parent=seed_member,
        crossover_parent=None,
        generation_summary="Seed baseline benchmark workspace.",
        args=args,
    )
    update_candidate(
        conn,
        seed_member.candidate_id,
        worktree_path=spec.path,
        worker_slot=spec.worker_slot,
        gpu_id=spec.gpu_id,
        claude_log_path=spec.claude_log_path,
        validation_log_path=spec.validation_log_path,
        benchmark_log_path=spec.benchmark_log_path,
        notes_path=spec.notes_path,
    )

    validation = adapter.pre_benchmark_validate(spec, args)
    update_candidate(
        conn,
        seed_member.candidate_id,
        validation_status=validation.status,
        validation_summary=validation.summary,
    )
    if not validation.passed:
        update_candidate(
            conn,
            seed_member.candidate_id,
            benchmark_status="skipped",
            benchmark_summary="seed baseline failed pre-benchmark validation",
            valid=False,
        )
        finish_generation(conn, 0, best_candidate_id=None, best_objective_value=None, best_snapshot_path=None)
        return seed_member

    benchmark = adapter.benchmark(spec, args)
    update_candidate(
        conn,
        seed_member.candidate_id,
        benchmark_status=benchmark.status,
        benchmark_summary=benchmark.summary,
        objective_value=benchmark.objective_value,
        fitness=benchmark.fitness,
        metrics_json=benchmark.metrics,
        constraints_json=benchmark.constraints,
        valid=benchmark.valid,
        promoted=benchmark.valid,
        elite=benchmark.valid,
    )
    updated_seed = PopulationMember(
        candidate_id=seed_member.candidate_id,
        generation=0,
        name=seed_member.name,
        snapshot=seed_member.snapshot,
        operator="seed",
        objective_value=benchmark.objective_value,
        fitness=benchmark.fitness,
        metrics=benchmark.metrics,
        constraints=benchmark.constraints,
        valid=benchmark.valid,
        benchmark_status=benchmark.status,
        validation_status=validation.status,
        benchmark_summary=benchmark.summary,
    )
    record_population_membership(
        conn,
        0,
        "end",
        [updated_seed.candidate_id] if updated_seed.valid else [],
        elite_ids={updated_seed.candidate_id} if updated_seed.valid else set(),
    )
    finish_generation(
        conn,
        0,
        best_candidate_id=updated_seed.candidate_id if updated_seed.valid else None,
        best_objective_value=updated_seed.objective_value,
        best_snapshot_path=updated_seed.snapshot.path if updated_seed.valid else None,
    )
    return updated_seed


def plan_generation(
    *,
    generation: int,
    current_population: list[PopulationMember],
    elite_parents: list[PopulationMember],
    workspaces_root: Path,
    repo: Path,
    adapter: ObjectiveAdapter,
    generation_summary: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None]]:
    generation_dir = generation_workspace_dir(workspaces_root, generation)
    offspring_target = max(1, args.population_size - len(elite_parents))
    candidates: list[tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None]] = []

    for index in range(1, offspring_target + 1):
        mode, operator, parent_a, parent_b = plan_candidate_operator(
            current_population=current_population,
            args=args,
            rng=rng,
        )
        spec = WorkspaceSpec(
            generation=generation,
            index=index,
            name=f"candidate-{index:03d}",
            mode=mode,
            operator=operator,
            path=generation_dir / f"candidate-{index:03d}",
            worker_slot=(index - 1) % args.workers,
            gpu_id=((index - 1) % args.gpus) if args.gpus else None,
            parent_a=parent_a.candidate_id,
            parent_b=None if parent_b is None else parent_b.candidate_id,
        )
        initialize_workspace(
            repo=repo,
            spec=spec,
            adapter=adapter,
            seed_parent=parent_a,
            crossover_parent=parent_b,
            generation_summary=generation_summary,
            args=args,
        )
        candidate_id = f"g{generation:04d}-{spec.name}"
        candidates.append((candidate_id, spec, parent_a, parent_b))
    return candidates


def register_planned_candidates(
    conn: sqlite3.Connection,
    planned_candidates: list[tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None]],
) -> None:
    for candidate_id, spec, parent_a, parent_b in planned_candidates:
        insert_candidate(
            conn,
            candidate_id=candidate_id,
            generation=spec.generation,
            name=spec.name,
            mode=spec.mode,
            operator=spec.operator,
            worktree_path=spec.path,
            worker_slot=spec.worker_slot,
            gpu_id=spec.gpu_id,
            parent_a=parent_a.candidate_id,
            parent_b=None if parent_b is None else parent_b.candidate_id,
            source_snapshot_path=parent_a.snapshot.path,
            claude_log_path=spec.claude_log_path,
            validation_log_path=spec.validation_log_path,
            benchmark_log_path=spec.benchmark_log_path,
            notes_path=spec.notes_path,
        )


def run_generation_candidates(
    conn: sqlite3.Connection,
    planned_candidates: list[tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None]],
    snapshots_root: Path,
    adapter: ObjectiveAdapter,
    args: argparse.Namespace,
) -> list[PopulationMember]:
    options = ClaudeOptions(
        runner=args.agent_runner,
        agent_bin=args.agent_bin,
        effort=args.effort,
        timeout_minutes=args.agent_timeout_minutes,
        model=args.model,
    )
    offspring_members: list[PopulationMember] = []

    for batch in batched(planned_candidates, args.workers):
        batch_specs = [spec for _, spec, _, _ in batch]
        sessions: list[CandidateSession] = []
        try:
            sessions = run_candidate_sessions(batch_specs, options)
        finally:
            finalize_session_processes(sessions)

        session_by_path = {session.spec.path: session for session in sessions}
        ready_batch: list[tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None]] = []
        for candidate_id, spec, parent_a, parent_b in batch:
            session = session_by_path[spec.path]
            agent_status, agent_summary = evaluate_candidate_session(session, adapter)
            update_candidate(
                conn,
                candidate_id,
                agent_status=agent_status,
                agent_summary=agent_summary,
            )
            log(f"agent generation={spec.generation} {candidate_id}: {agent_status}")
            if agent_status == "ready":
                ready_batch.append((candidate_id, spec, parent_a, parent_b))
            else:
                update_candidate(conn, candidate_id, benchmark_status="skipped", valid=False)

        if ready_batch:
            with ThreadPoolExecutor(max_workers=len(ready_batch)) as executor:
                results = list(
                    executor.map(
                        lambda item: finalize_candidate_workspace(
                            item,
                            adapter=adapter,
                            args=args,
                            snapshots_root=snapshots_root,
                        ),
                        ready_batch,
                    )
                )
            for result in results:
                candidate_id = result["candidate_id"]
                snapshot = result["snapshot"]
                validation: ValidationResult = result["validation"]
                benchmark: BenchmarkResult | None = result["benchmark"]
                update_candidate(
                    conn,
                    candidate_id,
                    snapshot_path=None if snapshot is None else snapshot.path,
                    snapshot_hash=None if snapshot is None else snapshot.artifact_hash,
                    artifact_paths_json=None if snapshot is None else snapshot.artifact_paths,
                    validation_status=validation.status,
                    validation_summary=validation.summary,
                )
                if benchmark is None:
                    update_candidate(
                        conn,
                        candidate_id,
                        benchmark_status="skipped",
                        benchmark_summary="pre-benchmark validation failed",
                        valid=False,
                    )
                    log(f"benchmark generation={result['generation']} {candidate_id}: skipped")
                    continue

                update_candidate(
                    conn,
                    candidate_id,
                    benchmark_status=benchmark.status,
                    benchmark_summary=benchmark.summary,
                    objective_value=benchmark.objective_value,
                    fitness=benchmark.fitness,
                    metrics_json=benchmark.metrics,
                    constraints_json=benchmark.constraints,
                    valid=benchmark.valid,
                )
                suffix = ""
                if benchmark.objective_value is not None:
                    suffix = f" {adapter.objective.name}={benchmark.objective_value:.6f}"
                log(
                    f"benchmark generation={result['generation']} "
                    f"{candidate_id}: {benchmark.status}{suffix}"
                )
                member = result["member"]
                if member is not None:
                    offspring_members.append(member)

    return offspring_members


def finalize_candidate_workspace(
    planned_candidate: tuple[str, WorkspaceSpec, PopulationMember, PopulationMember | None],
    *,
    adapter: ObjectiveAdapter,
    args: argparse.Namespace,
    snapshots_root: Path,
) -> dict[str, object]:
    candidate_id, spec, parent_a, parent_b = planned_candidate
    detected_state = adapter.detect_candidate_state(spec)
    if not detected_state.has_state:
        return {
            "candidate_id": candidate_id,
            "generation": spec.generation,
            "snapshot": None,
            "validation": ValidationResult(
                status="no_state",
                passed=False,
                summary=detected_state.summary,
            ),
            "benchmark": None,
            "member": None,
        }

    artifact_paths = adapter.prepare_candidate_for_benchmark(spec, detected_state)
    snapshot = create_snapshot_from_paths(
        snapshots_root=snapshots_root,
        candidate_id=candidate_id,
        adapter_name=adapter.name,
        label=spec.name,
        artifact_sources=[(spec.path / path, path) for path in artifact_paths],
        notes_source=spec.notes_path,
        metadata={
            "generation": spec.generation,
            "operator": spec.operator,
            "mode": spec.mode,
            "parent_a": parent_a.candidate_id,
            "parent_b": None if parent_b is None else parent_b.candidate_id,
        },
    )
    validation = adapter.pre_benchmark_validate(spec, args)
    if not validation.passed:
        return {
            "candidate_id": candidate_id,
            "generation": spec.generation,
            "snapshot": snapshot,
            "validation": validation,
            "benchmark": None,
            "member": None,
        }

    benchmark = adapter.benchmark(spec, args)
    member = PopulationMember(
        candidate_id=candidate_id,
        generation=spec.generation,
        name=spec.name,
        snapshot=snapshot,
        operator=spec.operator,
        parent_a=parent_a.candidate_id,
        parent_b=None if parent_b is None else parent_b.candidate_id,
        objective_value=benchmark.objective_value,
        fitness=benchmark.fitness,
        metrics=benchmark.metrics,
        constraints=benchmark.constraints,
        valid=benchmark.valid,
        benchmark_status=benchmark.status,
        validation_status=validation.status,
        benchmark_summary=benchmark.summary,
    )
    return {
        "candidate_id": candidate_id,
        "generation": spec.generation,
        "snapshot": snapshot,
        "validation": validation,
        "benchmark": benchmark,
        "member": member if benchmark.valid else None,
    }


def plan_candidate_operator(
    *,
    current_population: list[PopulationMember],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[str, str, PopulationMember, PopulationMember | None]:
    valid_population = [member for member in current_population if member.valid]
    parent_pool = valid_population or current_population
    if not parent_pool:
        raise SystemExit("cannot create offspring without at least one valid parent")

    if len(parent_pool) >= 2 and rng.random() < args.crossover_rate:
        parent_a, parent_b = select_parent_pair(
            parent_pool,
            args.selection_strategy,
            rng,
        )
        return "crossover", "crossover", parent_a, parent_b

    if valid_population:
        parent_a = select_parent(valid_population, args.selection_strategy, rng)
    else:
        parent_a = parent_pool[0]
    if rng.random() < args.mutation_rate:
        return "mutate", "mutate", parent_a, None
    return "repair", "repair", parent_a, None


def select_elite_parents(
    current_population: list[PopulationMember],
    elite_count: int,
) -> list[PopulationMember]:
    valid_population = [member for member in current_population if member.valid]
    if elite_count <= 0:
        return []
    return rank_population(valid_population)[:elite_count]


def evaluate_candidate_session(
    session: CandidateSession,
    adapter: ObjectiveAdapter,
) -> tuple[str, str]:
    spec = session.spec
    summary = first_tail_text(
        spec.claude_log_path,
        spec.claude_stderr_path,
        spec.claude_debug_log_path,
        spec.claude_status_path,
    )
    detected_state = adapter.detect_candidate_state(spec)
    if not detected_state.has_state:
        status = "timed_out_no_state" if session.timed_out else "no_state"
        return status, summary or detected_state.summary
    return "ready", timeout_summary(session, summary or detected_state.summary)


def timeout_summary(session: CandidateSession, summary: str) -> str:
    if session.timed_out and not summary:
        return TIMEOUT_SUMMARY
    return summary


def population_summary(population: list[PopulationMember], objective_name: str) -> str:
    if not population:
        return "No prior population members are available."
    lines = [
        f"Current population size: {len(population)}",
        f"Objective: {objective_name}",
        "",
        "Top candidates:",
    ]
    for member in rank_population(population)[: min(5, len(population))]:
        value = "n/a" if member.objective_value is None else f"{member.objective_value:.6f}"
        lines.append(
            f"- {member.candidate_id}: valid={member.valid} {objective_name}={value} "
            f"operator={member.operator} hash={member.snapshot.artifact_hash}"
        )
    return "\n".join(lines)


T = TypeVar("T")


def batched(items: list[T], size: int) -> Iterable[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


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

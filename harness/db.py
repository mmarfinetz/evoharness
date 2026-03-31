from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from harness.models import ObjectiveSpec, WorkspaceSpec


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists run_meta (
            key text primary key,
            value text not null
        );

        create table if not exists generations (
            generation integer primary key,
            started_at text not null,
            completed_at text,
            population_size integer not null,
            worker_count integer not null,
            elite_count integer not null,
            selection_strategy text not null,
            objective_name text not null,
            objective_direction text not null,
            best_candidate_id text,
            best_objective_value real,
            best_snapshot_path text
        );

        create table if not exists candidates (
            candidate_id text primary key,
            generation integer not null,
            name text not null,
            mode text not null,
            operator text not null,
            worktree_path text,
            worker_slot integer,
            gpu_id integer,
            parent_a text,
            parent_b text,
            source_snapshot_path text,
            snapshot_path text,
            snapshot_hash text,
            artifact_paths_json text,
            setup_status text not null,
            agent_status text,
            agent_summary text,
            validation_status text,
            validation_summary text,
            benchmark_status text,
            benchmark_summary text,
            objective_value real,
            fitness real,
            valid integer not null default 0,
            promoted integer not null default 0,
            elite integer not null default 0,
            metrics_json text,
            constraints_json text,
            claude_log_path text,
            validation_log_path text,
            benchmark_log_path text,
            notes_path text,
            created_at text not null
        );

        create table if not exists population_membership (
            generation integer not null,
            stage text not null,
            slot integer not null,
            candidate_id text not null,
            role text not null,
            primary key (generation, stage, slot)
        );
        """
    )
    conn.commit()


def record_run_meta(
    conn: sqlite3.Connection,
    run_name: str,
    repo: Path,
    args: Any,
    objective: ObjectiveSpec,
) -> None:
    meta = {
        "run_name": run_name,
        "repo": str(repo),
        "adapter": args.adapter,
        "objective_name": objective.name,
        "objective_direction": objective.direction,
        "agent_runner": args.agent_runner,
        "agent_bin": args.agent_bin,
        "population_size": str(args.population_size),
        "workers": str(args.workers),
        "gpus": str(args.gpus),
        "elite_count": str(args.elite_count),
        "selection_strategy": args.selection_strategy,
        "mutation_rate": str(args.mutation_rate),
        "crossover_rate": str(args.crossover_rate),
        "effort": args.effort,
        "agent_timeout_minutes": str(args.agent_timeout_minutes),
        "benchmark_timeout_minutes": str(args.benchmark_timeout_minutes),
        "max_generations": "" if args.max_generations is None else str(args.max_generations),
    }
    baseline_train_py = getattr(args, "baseline_train_py", None)
    if baseline_train_py:
        meta["baseline_train_py"] = str(baseline_train_py)
    if args.model:
        meta["model"] = args.model
    if args.adapter_config:
        meta["adapter_config"] = str(args.adapter_config)
    conn.executemany(
        "insert or replace into run_meta(key, value) values(?, ?)",
        meta.items(),
    )
    conn.commit()


def record_generation_start(
    conn: sqlite3.Connection,
    generation: int,
    args: Any,
    objective: ObjectiveSpec,
) -> None:
    conn.execute(
        """
        insert into generations(
            generation, started_at, population_size, worker_count, elite_count,
            selection_strategy, objective_name, objective_direction
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            generation,
            datetime.now().isoformat(timespec="seconds"),
            args.population_size,
            args.workers,
            args.elite_count,
            args.selection_strategy,
            objective.name,
            objective.direction,
        ),
    )
    conn.commit()


def record_population_membership(
    conn: sqlite3.Connection,
    generation: int,
    stage: str,
    candidate_ids: Iterable[str],
    *,
    elite_ids: set[str] | None = None,
) -> None:
    elite_ids = elite_ids or set()
    conn.execute(
        "delete from population_membership where generation = ? and stage = ?",
        (generation, stage),
    )
    for slot, candidate_id in enumerate(candidate_ids):
        role = "elite" if candidate_id in elite_ids else "survivor"
        conn.execute(
            """
            insert into population_membership(generation, stage, slot, candidate_id, role)
            values (?, ?, ?, ?, ?)
            """,
            (generation, stage, slot, candidate_id, role),
        )
    conn.commit()


def insert_candidate(
    conn: sqlite3.Connection,
    *,
    candidate_id: str,
    generation: int,
    name: str,
    mode: str,
    operator: str,
    worktree_path: Path | None,
    worker_slot: int | None,
    gpu_id: int | None,
    parent_a: str | None,
    parent_b: str | None,
    source_snapshot_path: Path | None,
    claude_log_path: Path | None,
    validation_log_path: Path | None,
    benchmark_log_path: Path | None,
    notes_path: Path | None,
    setup_status: str = "ready",
) -> None:
    conn.execute(
        """
        insert into candidates(
            candidate_id, generation, name, mode, operator, worktree_path, worker_slot,
            gpu_id, parent_a, parent_b, source_snapshot_path, setup_status, claude_log_path,
            validation_log_path, benchmark_log_path, notes_path, created_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            generation,
            name,
            mode,
            operator,
            None if worktree_path is None else str(worktree_path),
            worker_slot,
            gpu_id,
            parent_a,
            parent_b,
            None if source_snapshot_path is None else str(source_snapshot_path),
            setup_status,
            None if claude_log_path is None else str(claude_log_path),
            None if validation_log_path is None else str(validation_log_path),
            None if benchmark_log_path is None else str(benchmark_log_path),
            None if notes_path is None else str(notes_path),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def update_candidate(conn: sqlite3.Connection, candidate_id: str, **fields: object) -> None:
    clean_fields = {key: serialize_field(value) for key, value in fields.items() if value is not None}
    if not clean_fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in clean_fields)
    conn.execute(
        f"update candidates set {assignments} where candidate_id = ?",
        (*clean_fields.values(), candidate_id),
    )
    conn.commit()


def finish_generation(
    conn: sqlite3.Connection,
    generation: int,
    *,
    best_candidate_id: str | None,
    best_objective_value: float | None,
    best_snapshot_path: Path | None,
) -> None:
    conn.execute(
        """
        update generations
        set completed_at = ?, best_candidate_id = ?, best_objective_value = ?, best_snapshot_path = ?
        where generation = ?
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            best_candidate_id,
            best_objective_value,
            None if best_snapshot_path is None else str(best_snapshot_path),
            generation,
        ),
    )
    conn.commit()


def list_ranked_generation_candidates(conn: sqlite3.Connection, generation: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        select *
        from candidates
        where generation = ?
        order by
            valid desc,
            fitness desc,
            objective_value asc,
            snapshot_hash asc,
            candidate_id asc
        """,
        (generation,),
    ).fetchall()
    return list(rows)


def serialize_field(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    return value

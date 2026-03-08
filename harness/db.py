from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any


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
            baseline_train_path text not null,
            winner_name text,
            winner_train_path text,
            winner_val_bpb real
        );

        create table if not exists candidates (
            generation integer not null,
            name text not null,
            kind text not null,
            worktree_path text not null,
            baseline_train_path text not null,
            gpu_id integer not null,
            setup_status text not null,
            agent_status text,
            agent_summary text,
            train_sha256 text,
            benchmark_status text,
            val_bpb real,
            training_seconds real,
            total_seconds real,
            peak_vram_mb real,
            promoted integer not null default 0,
            claude_log_path text not null,
            train_log_path text not null,
            notes_path text not null,
            primary key (generation, name)
        );
        """
    )
    conn.commit()


def record_run_meta(conn: sqlite3.Connection, run_name: str, repo: Path, args: Any) -> None:
    meta = {
        "run_name": run_name,
        "repo": str(repo),
        "claude_bin": args.claude_bin,
        "agents": str(args.agents),
        "gpus": str(args.gpus),
        "effort": args.effort,
        "agent_timeout_minutes": str(args.agent_timeout_minutes),
        "train_timeout_minutes": str(args.train_timeout_minutes),
        "max_generations": "" if args.max_generations is None else str(args.max_generations),
    }
    if args.baseline_train_py:
        meta["baseline_train_py"] = str(args.baseline_train_py)
    if args.model:
        meta["model"] = args.model
    conn.executemany(
        "insert or replace into run_meta(key, value) values(?, ?)",
        meta.items(),
    )
    conn.commit()


def record_generation(
    conn: sqlite3.Connection,
    generation: int,
    baseline_train_path: Path,
    specs: list[Any],
) -> None:
    conn.execute(
        """
        insert into generations(generation, started_at, baseline_train_path)
        values (?, ?, ?)
        """,
        (
            generation,
            datetime.now().isoformat(timespec="seconds"),
            str(baseline_train_path),
        ),
    )
    for spec in specs:
        conn.execute(
            textwrap.dedent(
                """
                insert into candidates(
                    generation, name, kind, worktree_path, baseline_train_path, gpu_id,
                    setup_status, claude_log_path, train_log_path, notes_path
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            ),
            (
                spec.generation,
                spec.name,
                spec.kind,
                str(spec.path),
                str(spec.baseline_train_path),
                spec.gpu_id,
                "ready",
                str(spec.claude_log_path),
                str(spec.train_log_path),
                str(spec.notes_path),
            ),
        )
    conn.commit()


def finish_generation(conn: sqlite3.Connection, generation: int, winner: dict[str, object]) -> None:
    conn.execute(
        """
        update generations
        set completed_at = ?, winner_name = ?, winner_train_path = ?, winner_val_bpb = ?
        where generation = ?
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            winner["name"],
            winner["winner_train_path"],
            winner["val_bpb"],
            generation,
        ),
    )
    conn.commit()


def get_candidate_agent_status(conn: sqlite3.Connection, generation: int, name: str) -> str | None:
    row = conn.execute(
        "select agent_status from candidates where generation = ? and name = ?",
        (generation, name),
    ).fetchone()
    if row is None:
        return None
    return row["agent_status"]


def update_candidate(conn: sqlite3.Connection, generation: int, name: str, **fields: object) -> None:
    clean_fields = {key: value for key, value in fields.items() if value is not None}
    if not clean_fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in clean_fields)
    conn.execute(
        f"update candidates set {assignments} where generation = ? and name = ?",
        (*clean_fields.values(), generation, name),
    )
    conn.commit()


def select_generation_winner(conn: sqlite3.Connection, generation: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        select name, worktree_path, val_bpb
        from candidates
        where generation = ? and benchmark_status = 'ok'
        order by val_bpb asc, name asc
        limit 1
        """,
        (generation,),
    ).fetchone()


def mark_generation_winner(conn: sqlite3.Connection, generation: int, winner_name: str) -> None:
    conn.execute("update candidates set promoted = 0 where generation = ?", (generation,))
    conn.execute(
        "update candidates set promoted = 1 where generation = ? and name = ?",
        (generation, winner_name),
    )
    conn.commit()

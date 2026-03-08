from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.utils import build_gpu_env, terminate_gpu_processes

CLAUDE_ALLOWED_TOOLS = "Bash Edit Read MultiEdit Write Glob Grep"


@dataclass(frozen=True)
class ClaudeOptions:
    claude_bin: str
    effort: str
    timeout_minutes: int
    model: str | None = None


@dataclass
class CandidateSession:
    spec: Any
    process: subprocess.Popen[str]
    command: list[str]
    timed_out: bool = False


def run_candidate_sessions(specs: list[Any], options: ClaudeOptions) -> list[CandidateSession]:
    sessions = [start_candidate_session(spec, options) for spec in specs]
    active = list(sessions)
    deadline = time.monotonic() + options.timeout_minutes * 60

    try:
        while active and time.monotonic() < deadline:
            active = [session for session in active if session.process.poll() is None]
            if not active:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        for session in active:
            session.timed_out = True
            terminate_process_group(session.process)
            terminate_gpu_processes(session.spec.gpu_id)
        raise

    for session in active:
        if session.process.poll() is None:
            session.timed_out = True
            terminate_process_group(session.process)
            terminate_gpu_processes(session.spec.gpu_id)

    return sessions


def start_candidate_session(spec: Any, options: ClaudeOptions) -> CandidateSession:
    system_prompt = Path(spec.path / "system.md").read_text()
    stdout_path = spec.claude_log_path
    stderr_path = spec.claude_stderr_path
    debug_log_path = spec.claude_debug_log_path
    status_path = spec.claude_status_path
    command = [
        options.claude_bin,
        "--print",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--debug-file",
        str(debug_log_path),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        CLAUDE_ALLOWED_TOOLS,
        "--system-prompt",
        system_prompt,
        "--effort",
        options.effort,
    ]
    if options.model:
        command.extend(["--model", options.model])
    command.append("Start now. Work continuously until the harness interrupts you.")
    env = build_gpu_env(spec.gpu_id)

    write_json(
        status_path,
        {
            "status": "launching",
            "command": command,
            **session_file_fields(spec),
        },
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab", buffering=0) as stdout_handle, stderr_path.open(
        "ab", buffering=0
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=spec.path,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=False,
            start_new_session=True,
        )
    write_json(
        status_path,
        {
            "status": "running",
            "pid": process.pid,
            "command": command,
            **session_file_fields(spec),
        },
    )
    return CandidateSession(spec=spec, process=process, command=command)


def finalize_session_processes(sessions: list[CandidateSession]) -> None:
    for session in sessions:
        wait_for_exit(session.process)
        output = session_output(session)
        write_json(
            session.spec.claude_status_path,
            {
                "status": "timed_out" if session.timed_out else "completed",
                "command": session.command,
                **output,
            },
        )
        write_json(session.spec.claude_output_path, output)


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        process.terminate()
        return

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def wait_for_exit(process: subprocess.Popen[str]) -> None:
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        process.wait(timeout=5)


def session_file_fields(spec: Any) -> dict[str, object]:
    return {
        "cwd": str(spec.path),
        "stdout_path": str(spec.claude_log_path),
        "stderr_path": str(spec.claude_stderr_path),
        "debug_log_path": str(spec.claude_debug_log_path),
    }


def session_output(session: CandidateSession) -> dict[str, object]:
    return {
        "pid": session.process.pid,
        "returncode": session.process.returncode,
        "timed_out": session.timed_out,
        **session_file_fields(session.spec),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

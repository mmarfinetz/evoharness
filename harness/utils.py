from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def detect_gpu_ids() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        raw_ids = [part.strip() for part in cuda_visible_devices.split(",")]
        gpu_ids = [
            gpu_id
            for gpu_id in raw_ids
            if gpu_id and gpu_id not in {"-1", "none", "None", "void", "NoDevFiles"}
        ]
        return gpu_ids

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []

    result = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def detect_gpu_count() -> int:
    return len(detect_gpu_ids())


def selected_gpu_id(gpu_id: int) -> str:
    detected_gpu_ids = detect_gpu_ids()
    if gpu_id < len(detected_gpu_ids):
        return detected_gpu_ids[gpu_id]
    return str(gpu_id)


def _default_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    return str(Path.home() / ".local" / "bin" / name)


def build_gpu_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = selected_gpu_id(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def terminate_workspace_processes(workspace: Path) -> None:
    pattern = str(workspace)
    if not pattern:
        return
    _signal_matching_processes(pattern, "-TERM")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not list_workspace_processes(workspace):
            return
        time.sleep(0.5)
    _signal_matching_processes(pattern, "-KILL")


def list_workspace_processes(workspace: Path) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", str(workspace)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        return []
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def wait_for_gpu_idle(gpu_id: int, timeout_seconds: int = 30) -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return True

    target_gpu = selected_gpu_id(gpu_id)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not gpu_has_compute_processes(nvidia_smi, target_gpu):
            return True
        time.sleep(1)
    return not gpu_has_compute_processes(nvidia_smi, target_gpu)


def terminate_gpu_processes(gpu_id: int) -> None:
    for pid in gpu_process_pids(gpu_id):
        _terminate_pid(pid)


def gpu_has_compute_processes(nvidia_smi: str, target_gpu: str) -> bool:
    return bool(_gpu_process_rows(nvidia_smi, target_gpu))


def gpu_process_pids(gpu_id: int) -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    return [pid for _, pid in _gpu_process_rows(nvidia_smi, selected_gpu_id(gpu_id))]


def _gpu_process_rows(nvidia_smi: str, target_gpu: str) -> list[tuple[str, int]]:
    gpu_result = subprocess.run(
        [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_result.returncode != 0:
        return []

    target_uuids: set[str] = set()
    for line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        index, uuid = parts
        if target_gpu in {index, uuid}:
            target_uuids.add(uuid)
    if not target_uuids:
        return []

    process_result = subprocess.run(
        [nvidia_smi, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process_result.returncode not in (0, 1):
        return []

    rows: list[tuple[str, int]] = []
    for line in process_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_uuid, pid = parts
        if gpu_uuid in target_uuids and pid.isdigit():
            rows.append((gpu_uuid, int(pid)))
    return rows


def _signal_matching_processes(pattern: str, signal_name: str) -> None:
    subprocess.run(
        ["pkill", signal_name, "-f", pattern],
        capture_output=True,
        text=True,
        check=False,
    )


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def default_claude_bin() -> str:
    return _default_bin("claude")


def default_uv_bin() -> str:
    return _default_bin("uv")


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_log_line(message: str) -> str:
    return f"[{timestamp()}] {message}"


def log(message: str) -> None:
    print(format_log_line(message), flush=True)


def run_command(
    command: list[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=text, check=False)
    if check and result.returncode != 0:
        raise SystemExit(
            "command failed:\n"
            + " ".join(command)
            + "\n"
            + _command_output(result.stdout)
            + _command_output(result.stderr)
        )
    return result


def _command_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output

from __future__ import annotations

import os
import shutil
import subprocess
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


def _default_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    return str(Path.home() / ".local" / "bin" / name)


def build_gpu_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    detected_gpu_ids = detect_gpu_ids()
    selected_gpu = (
        detected_gpu_ids[gpu_id] if gpu_id < len(detected_gpu_ids) else str(gpu_id)
    )
    env["CUDA_VISIBLE_DEVICES"] = selected_gpu
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


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

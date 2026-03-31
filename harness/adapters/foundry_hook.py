from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from harness.adapters.base import ObjectiveAdapter
from harness.models import (
    ArtifactMapping,
    BenchmarkResult,
    CandidateSnapshot,
    CandidateStateDetection,
    ObjectiveSpec,
    PopulationMember,
    ResourceRequirements,
    ValidationResult,
    WorkspaceSpec,
)
from harness.snapshots import create_snapshot_from_paths, materialize_snapshot
from harness.utils import (
    build_worker_env,
    format_log_line,
    terminate_workspace_processes,
    wait_for_gpu_idle,
)


SUPPORT_ROOT = PurePosixPath(".harness/foundry_hook")
BASELINE_ROOT = SUPPORT_ROOT / "baseline"
PARENTS_ROOT = SUPPORT_ROOT / "parents"
CONFIG_COPY_PATH = SUPPORT_ROOT / "adapter_config.json"
NVIDIA_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SupportFileSpec:
    source_path: Path
    destination_path: str
    include_as_context: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source_path),
            "destination": self.destination_path,
            "context": self.include_as_context,
        }


@dataclass(frozen=True)
class FoundryHookConfig:
    objective: ObjectiveSpec
    editable_artifacts: tuple[str, ...]
    context_files: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    support_files: tuple[SupportFileSpec, ...]
    validation_commands: tuple[tuple[str, ...], ...]
    benchmark_command: tuple[str, ...]
    benchmark_output_path: str | None
    required_metrics: tuple[str, ...]
    repo_required_paths: tuple[str, ...]
    requires_gpu: bool

    def to_json_dict(self) -> dict[str, object]:
        return {
            "objective": {
                "name": self.objective.name,
                "direction": self.objective.direction,
                "description": self.objective.description,
                "primary_metric": self.objective.primary_metric,
            },
            "editable_artifacts": list(self.editable_artifacts),
            "context_files": list(self.context_files),
            "forbidden_paths": list(self.forbidden_paths),
            "support_files": [item.to_json_dict() for item in self.support_files],
            "validation_commands": [list(command) for command in self.validation_commands],
            "benchmark_command": list(self.benchmark_command),
            "benchmark_output_path": self.benchmark_output_path,
            "required_metrics": list(self.required_metrics),
            "repo_required_paths": list(self.repo_required_paths),
            "requires_gpu": self.requires_gpu,
        }


class FoundryHookAdapter(ObjectiveAdapter):
    name = "foundry_hook"

    def __init__(self) -> None:
        self._loaded_config_path: Path | None = None
        self._config: FoundryHookConfig | None = None

    @property
    def objective(self) -> ObjectiveSpec:
        return self._require_config().objective

    @property
    def resources(self) -> ResourceRequirements:
        if self._config is None:
            cli_config_path = discover_cli_adapter_config_path()
            if cli_config_path is not None and cli_config_path.exists():
                self._load_config(cli_config_path.resolve())
        config = self._config
        return ResourceRequirements(
            requires_gpu=False if config is None else config.requires_gpu,
            supports_pre_benchmark_validation=True,
        )

    def validate_args(self, args: argparse.Namespace) -> None:
        if args.adapter_config is None:
            raise SystemExit("--adapter-config is required with --adapter foundry_hook")
        self._load_config(args.adapter_config.resolve())

    def validate_repo(self, repo: Path, args: argparse.Namespace) -> None:
        config = self._config_from_args(args)
        missing = [repo / relative_path for relative_path in config.repo_required_paths if not (repo / relative_path).exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise SystemExit(
                f"foundry_hook repo validation failed; missing required paths: {joined}"
            )

    def create_initial_snapshot(
        self,
        repo: Path,
        snapshots_root: Path,
        args: argparse.Namespace,
    ) -> CandidateSnapshot:
        config = self._config_from_args(args)
        return create_snapshot_from_paths(
            snapshots_root=snapshots_root,
            candidate_id="seed-baseline",
            adapter_name=self.name,
            label="baseline",
            artifact_sources=[
                (repo / relative_path, relative_path)
                for relative_path in config.editable_artifacts
            ],
            metadata={
                "repo": str(repo),
                "adapter_config": str(args.adapter_config),
            },
        )

    def editable_artifacts(self) -> tuple[str, ...]:
        return self._require_config().editable_artifacts

    def forbidden_paths(self) -> tuple[str, ...]:
        config = self._require_config()
        support_paths = (
            SUPPORT_ROOT.as_posix(),
            BASELINE_ROOT.as_posix(),
            PARENTS_ROOT.as_posix(),
        )
        support_destinations = tuple(item.destination_path for item in config.support_files)
        return tuple(dict.fromkeys(config.forbidden_paths + support_paths + support_destinations))

    def materialize_workspace(
        self,
        repo: Path,
        spec: WorkspaceSpec,
        seed_snapshot: CandidateSnapshot,
        parent_a: PopulationMember,
        parent_b: PopulationMember | None,
        generation_summary: str,
        args: argparse.Namespace,
    ) -> dict[str, str]:
        del repo
        config = self._config_from_args(args)
        materialize_snapshot(seed_snapshot, spec.path)
        materialize_snapshot(seed_snapshot, spec.path / BASELINE_ROOT)
        materialize_snapshot(parent_a.snapshot, spec.path / (PARENTS_ROOT / "parent_a"))
        if parent_b is not None:
            materialize_snapshot(parent_b.snapshot, spec.path / (PARENTS_ROOT / "parent_b"))
        for support_file in config.support_files:
            destination_path = spec.path / support_file.destination_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(support_file.source_path, destination_path)

        config_copy_path = spec.path / CONFIG_COPY_PATH
        config_copy_path.parent.mkdir(parents=True, exist_ok=True)
        config_copy_path.write_text(json.dumps(config.to_json_dict(), indent=2, sort_keys=True) + "\n")

        context_files = list(config.context_files)
        context_files.extend(config.editable_artifacts)
        context_files.extend(
            support_file.destination_path
            for support_file in config.support_files
            if support_file.include_as_context
        )
        context_files.extend(self._support_artifact_paths("baseline"))
        context_files.extend(self._support_artifact_paths("parents/parent_a"))
        context_files.extend(["notes.md", "lineage.md", CONFIG_COPY_PATH.as_posix()])

        parent_a_notes = spec.path / (SUPPORT_ROOT / "parent_a_notes.md")
        if parent_a.snapshot.notes_path is not None and parent_a.snapshot.notes_path.exists():
            shutil.copyfile(parent_a.snapshot.notes_path, parent_a_notes)
            context_files.append((SUPPORT_ROOT / "parent_a_notes.md").as_posix())

        if parent_b is not None:
            context_files.extend(self._support_artifact_paths("parents/parent_b"))
            parent_b_notes = spec.path / (SUPPORT_ROOT / "parent_b_notes.md")
            if parent_b.snapshot.notes_path is not None and parent_b.snapshot.notes_path.exists():
                shutil.copyfile(parent_b.snapshot.notes_path, parent_b_notes)
                context_files.append((SUPPORT_ROOT / "parent_b_notes.md").as_posix())

        validation_commands = "; ".join(format_command(command) for command in config.validation_commands)
        benchmark_command = format_command(config.benchmark_command)
        if config.benchmark_output_path is not None:
            benchmark_command = f"{benchmark_command} (expects JSON at `{config.benchmark_output_path}`)"

        return {
            "adapter_name": self.name,
            "objective_name": self.objective.name,
            "objective_direction": self.objective.direction,
            "objective_description": self.objective.description,
            "primary_metric": self.objective.primary_metric,
            "direct_edit_artifacts": ", ".join(
                format_paths((*config.editable_artifacts, "notes.md"))
            ),
            "checkpoint_artifacts": ", ".join(format_paths(config.editable_artifacts)),
            "forbidden_paths": ", ".join(format_paths(self.forbidden_paths())),
            "benchmark_command": benchmark_command,
            "validation_command": validation_commands,
            "resource_instructions": self._resource_instructions(spec.gpu_id, config.requires_gpu),
            "checkpoint_instructions": (
                "The current live source files are the promoted candidate state. "
                f"To recover from a regression, restore from `{BASELINE_ROOT.as_posix()}` "
                "or the parent copies under `.harness/foundry_hook/parents/`."
            ),
            "context_files": ", ".join(format_paths(tuple(dict.fromkeys(context_files)))),
            "generation_summary": generation_summary,
            "additional_rules": (
                "Treat `.harness/foundry_hook/` as read-only reference material. "
                "Run the configured validation commands before claiming a valid contract state, "
                "and only use the configured evaluator command to produce benchmark metrics."
            ),
        }

    def detect_candidate_state(self, spec: WorkspaceSpec) -> CandidateStateDetection:
        config = self._require_config()
        changed = False
        for relative_path in config.editable_artifacts:
            live_path = spec.path / relative_path
            baseline_path = spec.path / BASELINE_ROOT / relative_path
            if not live_path.exists():
                raise SystemExit(f"workspace is missing editable artifact: {live_path}")
            if not baseline_path.exists():
                raise SystemExit(f"workspace is missing baseline artifact: {baseline_path}")
            if live_path.read_bytes() != baseline_path.read_bytes():
                changed = True

        if changed:
            return CandidateStateDetection(
                status="ready",
                summary="benchmarking modified live contract artifacts",
                has_state=True,
                artifact_mappings=tuple(
                    ArtifactMapping(relative_path, relative_path)
                    for relative_path in config.editable_artifacts
                ),
            )
        return CandidateStateDetection(
            status="no_state",
            summary="no modified contract artifacts detected",
            has_state=False,
        )

    def prepare_candidate_for_benchmark(
        self,
        spec: WorkspaceSpec,
        detected_state: CandidateStateDetection,
    ) -> tuple[str, ...]:
        if not detected_state.has_state:
            raise SystemExit(
                f"cannot prepare candidate {spec.name} for benchmark without a detected state"
            )
        for relative_path in self._require_config().editable_artifacts:
            artifact_path = spec.path / relative_path
            if not artifact_path.exists():
                raise SystemExit(
                    f"candidate {spec.name} is missing benchmark artifact: {artifact_path}"
                )
        return self._require_config().editable_artifacts

    def pre_benchmark_validate(
        self,
        spec: WorkspaceSpec,
        args: argparse.Namespace,
    ) -> ValidationResult:
        config = self._config_from_args(args)
        timeout_seconds = max(1, args.benchmark_timeout_minutes * 60)
        with spec.validation_log_path.open("w") as handle:
            for index, command in enumerate(config.validation_commands, start=1):
                handle.write(
                    format_log_line(
                        f"[HARNESS] validation {index}/{len(config.validation_commands)}: {render_plain_command(command)}"
                    )
                    + "\n"
                )
                handle.flush()
                try:
                    result = subprocess.run(
                        list(command),
                        cwd=spec.path,
                        env=build_worker_env(spec.gpu_id),
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    handle.write(format_log_line("[HARNESS] validation timed out") + "\n")
                    terminate_workspace_processes(spec.path)
                    if spec.gpu_id is not None:
                        wait_for_gpu_idle(spec.gpu_id, timeout_seconds=NVIDIA_TIMEOUT_SECONDS)
                    return ValidationResult(
                        status="timeout",
                        passed=False,
                        summary="validation timed out",
                        details={
                            "command": list(command),
                            "command_index": index,
                        },
                    )

                if result.stdout:
                    handle.write(result.stdout)
                if result.stderr:
                    handle.write(result.stderr)
                if result.returncode != 0:
                    return ValidationResult(
                        status="validation_failed",
                        passed=False,
                        summary=tail_text(spec.validation_log_path)
                        or f"validation command {index} failed",
                        details={
                            "command": list(command),
                            "command_index": index,
                            "returncode": result.returncode,
                        },
                    )

        return ValidationResult(
            status="ok",
            passed=True,
            summary=f"{len(config.validation_commands)} validation commands passed",
            details={"command_count": len(config.validation_commands)},
        )

    def benchmark(
        self,
        spec: WorkspaceSpec,
        args: argparse.Namespace,
    ) -> BenchmarkResult:
        config = self._config_from_args(args)
        result: subprocess.CompletedProcess[str] | None = None
        with spec.benchmark_log_path.open("w") as log_file:
            if spec.gpu_id is not None and not wait_for_gpu_idle(
                spec.gpu_id, timeout_seconds=NVIDIA_TIMEOUT_SECONDS
            ):
                log_file.write(
                    format_log_line(
                        f"[HARNESS] GPU {spec.gpu_id} remained busy after cleanup; proceeding"
                    )
                    + "\n"
                )
            log_file.write(format_log_line(f"[HARNESS] benchmark start gpu={spec.gpu_id}") + "\n")
            log_file.write(
                format_log_line(
                    f"[HARNESS] benchmark command: {render_plain_command(config.benchmark_command)}"
                )
                + "\n"
            )
            log_file.flush()
            try:
                result = subprocess.run(
                    list(config.benchmark_command),
                    cwd=spec.path,
                    env=build_worker_env(spec.gpu_id),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=max(1, args.benchmark_timeout_minutes * 60),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                append_line(
                    spec.benchmark_log_path,
                    format_log_line("[HARNESS] benchmark timed out") + "\n",
                )
                terminate_workspace_processes(spec.path)
                if spec.gpu_id is not None:
                    wait_for_gpu_idle(spec.gpu_id, timeout_seconds=NVIDIA_TIMEOUT_SECONDS)
                return BenchmarkResult(
                    status="timeout",
                    summary="benchmark timed out",
                    objective_value=None,
                    fitness=None,
                    metrics={},
                    constraints={
                        "process_exit_zero": {
                            "passed": False,
                            "detail": "benchmark timed out",
                        }
                    },
                    valid=False,
                )

        terminate_workspace_processes(spec.path)
        if spec.gpu_id is not None:
            wait_for_gpu_idle(spec.gpu_id, timeout_seconds=NVIDIA_TIMEOUT_SECONDS)

        metrics, parse_error = self._load_benchmark_metrics(spec, config)
        missing_metrics = [name for name in config.required_metrics if metrics.get(name) is None]
        primary_metric = metrics.get(config.objective.primary_metric)
        primary_metric_ok, objective_value = coerce_metric_value(primary_metric)

        constraints: dict[str, dict[str, object]] = {
            "process_exit_zero": {
                "passed": result is not None and result.returncode == 0,
                "detail": f"returncode={None if result is None else result.returncode}",
            },
            "metrics_complete": {
                "passed": not missing_metrics,
                "missing_metrics": missing_metrics,
            },
            "metrics_json_parseable": {
                "passed": parse_error is None,
                "detail": "ok" if parse_error is None else parse_error,
            },
            "primary_metric_numeric": {
                "passed": primary_metric_ok,
                "detail": (
                    f"{config.objective.primary_metric}={primary_metric!r}"
                    if primary_metric_ok
                    else f"{config.objective.primary_metric} is not numeric: {primary_metric!r}"
                ),
            },
        }
        valid = all(item["passed"] for item in constraints.values())
        if not valid or objective_value is None:
            status = "parse_failed" if parse_error is not None or not primary_metric_ok or missing_metrics else "failed"
            return BenchmarkResult(
                status=status,
                summary=tail_text(spec.benchmark_log_path)
                or parse_error
                or "benchmark failed",
                objective_value=None,
                fitness=None,
                metrics=metrics,
                constraints=constraints,
                valid=False,
            )

        fitness = objective_value if config.objective.direction == "maximize" else -objective_value
        return BenchmarkResult(
            status="ok",
            summary=f"{config.objective.primary_metric}={objective_value:.6f}",
            objective_value=objective_value,
            fitness=fitness,
            metrics=metrics,
            constraints=constraints,
            valid=True,
        )

    def _config_from_args(self, args: argparse.Namespace) -> FoundryHookConfig:
        if args.adapter_config is None:
            raise SystemExit("--adapter-config is required with --adapter foundry_hook")
        return self._load_config(args.adapter_config.resolve())

    def _load_config(self, path: Path) -> FoundryHookConfig:
        if self._loaded_config_path == path and self._config is not None:
            return self._config
        config = load_foundry_hook_config(path)
        self._loaded_config_path = path
        self._config = config
        return config

    def _require_config(self) -> FoundryHookConfig:
        if self._config is None:
            raise SystemExit(
                "foundry_hook adapter config is not loaded yet; pass --adapter-config and run setup first"
            )
        return self._config

    def _resource_instructions(self, gpu_id: int | None, requires_gpu: bool) -> str:
        if not requires_gpu:
            return "No GPU is required for this workspace."
        if gpu_id is None:
            return "This workspace requires a GPU, but none has been assigned."
        return (
            "You have been assigned GPU "
            f"`{gpu_id}`. Any manual evaluator runs must use `CUDA_VISIBLE_DEVICES={gpu_id}`."
        )

    def _support_artifact_paths(self, prefix: str) -> list[str]:
        base = PurePosixPath(SUPPORT_ROOT.as_posix()) / prefix
        return [(base / relative_path).as_posix() for relative_path in self._require_config().editable_artifacts]

    def _load_benchmark_metrics(
        self,
        spec: WorkspaceSpec,
        config: FoundryHookConfig,
    ) -> tuple[dict[str, Any], str | None]:
        if config.benchmark_output_path is not None:
            metrics_path = spec.path / config.benchmark_output_path
            if not metrics_path.exists():
                return {}, f"benchmark output JSON not found: {metrics_path}"
            try:
                payload = json.loads(metrics_path.read_text())
            except json.JSONDecodeError as exc:
                return {}, f"benchmark output JSON parse failed: {exc}"
        else:
            log_text = spec.benchmark_log_path.read_text() if spec.benchmark_log_path.exists() else ""
            payload, error = parse_json_payload_from_text(log_text)
            if error is not None:
                return {}, error
        try:
            metrics_payload = extract_metrics_payload(payload)
        except ValueError as exc:
            return {}, str(exc)
        return metrics_payload, None


def load_foundry_hook_config(path: Path) -> FoundryHookConfig:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid foundry_hook adapter config JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"foundry_hook adapter config must be a JSON object: {path}")

    objective_payload = expect_mapping(payload.get("objective"), "objective", path)
    objective = ObjectiveSpec(
        name=expect_string(objective_payload.get("name"), "objective.name", path),
        direction=expect_direction(objective_payload.get("direction"), "objective.direction", path),
        description=expect_string(
            objective_payload.get("description"), "objective.description", path
        ),
        primary_metric=expect_string(
            objective_payload.get("primary_metric"), "objective.primary_metric", path
        ),
    )

    editable_artifacts = expect_path_list(
        payload.get("editable_artifacts"),
        "editable_artifacts",
        path,
        allow_empty=False,
    )
    context_files = expect_path_list(
        payload.get("context_files", []),
        "context_files",
        path,
        allow_empty=True,
    )
    forbidden_paths = expect_path_list(
        payload.get("forbidden_paths", []),
        "forbidden_paths",
        path,
        allow_empty=True,
    )
    support_files = expect_support_files(
        payload.get("support_files", []),
        "support_files",
        path,
    )
    validation_commands = expect_command_list(
        payload.get("validation_commands"),
        "validation_commands",
        path,
        allow_empty=False,
    )
    benchmark_command = expect_single_command(
        payload.get("benchmark_command"),
        "benchmark_command",
        path,
    )
    benchmark_output_path = payload.get("benchmark_output_path")
    if benchmark_output_path is not None:
        benchmark_output_path = normalize_relative_path(
            benchmark_output_path,
            "benchmark_output_path",
            path,
        )

    required_metrics = payload.get("required_metrics")
    if required_metrics is None:
        required_metric_paths = (objective.primary_metric,)
    else:
        required_metric_paths = expect_string_list(
            required_metrics,
            "required_metrics",
            path,
            allow_empty=False,
        )
        if objective.primary_metric not in required_metric_paths:
            required_metric_paths = tuple(dict.fromkeys(required_metric_paths + (objective.primary_metric,)))

    repo_required_paths = payload.get("repo_required_paths")
    if repo_required_paths is None:
        default_required = ["foundry.toml", *editable_artifacts, *context_files]
        repo_required_path_list = tuple(dict.fromkeys(default_required))
    else:
        repo_required_path_list = expect_path_list(
            repo_required_paths,
            "repo_required_paths",
            path,
            allow_empty=False,
        )

    requires_gpu = payload.get("requires_gpu", False)
    if not isinstance(requires_gpu, bool):
        raise SystemExit(f"foundry_hook config field requires_gpu must be a boolean: {path}")

    return FoundryHookConfig(
        objective=objective,
        editable_artifacts=editable_artifacts,
        context_files=context_files,
        forbidden_paths=forbidden_paths,
        support_files=support_files,
        validation_commands=validation_commands,
        benchmark_command=benchmark_command,
        benchmark_output_path=benchmark_output_path,
        required_metrics=required_metric_paths,
        repo_required_paths=repo_required_path_list,
        requires_gpu=requires_gpu,
    )


def expect_mapping(value: object, field_name: str, config_path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SystemExit(f"foundry_hook config field {field_name} must be an object: {config_path}")
    return value


def expect_string(value: object, field_name: str, config_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(
            f"foundry_hook config field {field_name} must be a non-empty string: {config_path}"
        )
    return value.strip()


def expect_direction(value: object, field_name: str, config_path: Path) -> str:
    direction = expect_string(value, field_name, config_path)
    if direction not in {"maximize", "minimize"}:
        raise SystemExit(
            f"foundry_hook config field {field_name} must be 'maximize' or 'minimize': {config_path}"
        )
    return direction


def expect_string_list(
    value: object,
    field_name: str,
    config_path: Path,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SystemExit(f"foundry_hook config field {field_name} must be a list: {config_path}")
    items = tuple(expect_string(item, f"{field_name}[]", config_path) for item in value)
    if not allow_empty and not items:
        raise SystemExit(f"foundry_hook config field {field_name} must not be empty: {config_path}")
    return tuple(dict.fromkeys(items))


def expect_path_list(
    value: object,
    field_name: str,
    config_path: Path,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    items = expect_string_list(value, field_name, config_path, allow_empty=allow_empty)
    return tuple(
        normalize_relative_path(item, f"{field_name}[]", config_path) for item in items
    )


def expect_support_files(
    value: object,
    field_name: str,
    config_path: Path,
) -> tuple[SupportFileSpec, ...]:
    if not isinstance(value, list):
        raise SystemExit(f"foundry_hook config field {field_name} must be a list: {config_path}")

    support_files: list[SupportFileSpec] = []
    for index, item in enumerate(value):
        mapping = expect_mapping(item, f"{field_name}[{index}]", config_path)
        include_as_context = mapping.get("context", True)
        if not isinstance(include_as_context, bool):
            raise SystemExit(
                f"foundry_hook config field {field_name}[{index}].context must be a boolean: {config_path}"
            )
        support_files.append(
            SupportFileSpec(
                source_path=resolve_config_relative_file(
                    mapping.get("source"),
                    f"{field_name}[{index}].source",
                    config_path,
                ),
                destination_path=normalize_relative_path(
                    mapping.get("destination"),
                    f"{field_name}[{index}].destination",
                    config_path,
                ),
                include_as_context=include_as_context,
            )
        )
    return tuple(support_files)


def expect_command_list(
    value: object,
    field_name: str,
    config_path: Path,
    *,
    allow_empty: bool,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise SystemExit(f"foundry_hook config field {field_name} must be a list: {config_path}")
    commands = tuple(
        expect_single_command(item, f"{field_name}[]", config_path) for item in value
    )
    if not allow_empty and not commands:
        raise SystemExit(f"foundry_hook config field {field_name} must not be empty: {config_path}")
    return commands


def expect_single_command(
    value: object,
    field_name: str,
    config_path: Path,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SystemExit(
            f"foundry_hook config field {field_name} must be a command array: {config_path}"
        )
    command = tuple(expect_string(item, f"{field_name}[]", config_path) for item in value)
    if not command:
        raise SystemExit(
            f"foundry_hook config field {field_name} must not be an empty command: {config_path}"
        )
    return command


def normalize_relative_path(value: object, field_name: str, config_path: Path) -> str:
    raw = expect_string(value, field_name, config_path)
    normalized = PurePosixPath(raw)
    if normalized.is_absolute() or normalized.parts[:1] == ("/",):
        raise SystemExit(
            f"foundry_hook config field {field_name} must be repo-relative, not absolute: {config_path}"
        )
    if ".." in normalized.parts or normalized == PurePosixPath("."):
        raise SystemExit(
            f"foundry_hook config field {field_name} must stay within the repo: {config_path}"
        )
    return normalized.as_posix()


def resolve_config_relative_file(value: object, field_name: str, config_path: Path) -> Path:
    raw = expect_string(value, field_name, config_path)
    normalized = PurePosixPath(raw)
    if normalized.is_absolute() or normalized.parts[:1] == ("/",):
        raise SystemExit(
            f"foundry_hook config field {field_name} must be relative to the config file: {config_path}"
        )
    if ".." in normalized.parts or normalized == PurePosixPath("."):
        raise SystemExit(
            f"foundry_hook config field {field_name} must stay within the config directory: {config_path}"
        )
    resolved = (config_path.parent / Path(normalized.as_posix())).resolve()
    if not resolved.exists():
        raise SystemExit(
            f"foundry_hook config field {field_name} references a missing file: {resolved}"
        )
    if not resolved.is_file():
        raise SystemExit(
            f"foundry_hook config field {field_name} must point to a file: {resolved}"
        )
    return resolved


def discover_cli_adapter_config_path() -> Path | None:
    args = sys.argv[1:]
    for index, token in enumerate(args):
        if token == "--adapter-config" and index + 1 < len(args):
            return Path(args[index + 1])
        if token.startswith("--adapter-config="):
            return Path(token.split("=", 1)[1])
    return None


def format_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"`{path}`" for path in paths)


def format_command(command: tuple[str, ...]) -> str:
    return f"`{render_plain_command(command)}`"


def render_plain_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def append_line(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text)


def tail_text(path: Path, limit: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    for line in reversed(lines[-limit:]):
        text = line.strip()
        if text:
            return text[:300]
    return ""


def parse_json_payload_from_text(text: str) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    stripped = text.strip()
    if not stripped:
        return None, "benchmark output log is empty"
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    lines = text.splitlines()
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:]).strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue
    return None, "benchmark output did not contain parseable JSON"


def extract_metrics_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("benchmark metrics payload must be a JSON object")
    metrics_payload = payload.get("metrics", payload)
    if not isinstance(metrics_payload, dict):
        raise ValueError("benchmark metrics field must be a JSON object")
    return metrics_payload


def coerce_metric_value(value: object) -> tuple[bool, float | None]:
    if isinstance(value, bool):
        return False, None
    if isinstance(value, (int, float)):
        return True, float(value)
    if isinstance(value, str):
        try:
            return True, float(value)
        except ValueError:
            return False, None
    return False, None

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ObjectiveDirection = Literal["minimize", "maximize"]
CandidateMode = Literal["seed", "mutate", "crossover", "repair", "elite"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: ObjectiveDirection
    description: str
    primary_metric: str


@dataclass(frozen=True)
class ResourceRequirements:
    requires_gpu: bool
    supports_pre_benchmark_validation: bool = True


@dataclass(frozen=True)
class ArtifactMapping:
    source_path: str
    destination_path: str


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str
    label: str
    path: Path
    artifact_paths: tuple[str, ...]
    artifact_hash: str
    adapter_name: str
    notes_path: Path | None = None
    metadata_path: Path | None = None


@dataclass
class PopulationMember:
    candidate_id: str
    generation: int
    name: str
    snapshot: CandidateSnapshot
    operator: str
    parent_a: str | None = None
    parent_b: str | None = None
    objective_value: float | None = None
    fitness: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    benchmark_status: str = "pending"
    agent_summary: str = ""
    validation_status: str = "pending"
    benchmark_summary: str = ""


@dataclass(frozen=True)
class ValidationResult:
    status: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    summary: str
    objective_value: float | None
    fitness: float | None
    metrics: dict[str, Any]
    constraints: dict[str, Any]
    valid: bool


@dataclass(frozen=True)
class CandidateStateDetection:
    status: str
    summary: str
    has_state: bool
    artifact_mappings: tuple[ArtifactMapping, ...] = ()


@dataclass(frozen=True)
class WorkspaceSpec:
    generation: int
    index: int
    name: str
    mode: CandidateMode
    operator: str
    path: Path
    worker_slot: int
    gpu_id: int | None
    parent_a: str | None = None
    parent_b: str | None = None

    @property
    def claude_log_path(self) -> Path:
        return self.path / "claude.log"

    @property
    def claude_stderr_path(self) -> Path:
        return self.path / "claude.stderr.log"

    @property
    def claude_debug_log_path(self) -> Path:
        return self.path / "claude.debug.log"

    @property
    def claude_output_path(self) -> Path:
        return self.path / "claude.output.json"

    @property
    def claude_status_path(self) -> Path:
        return self.path / "claude.status.json"

    @property
    def validation_log_path(self) -> Path:
        return self.path / "validation.log"

    @property
    def benchmark_log_path(self) -> Path:
        return self.path / "run.log"

    @property
    def notes_path(self) -> Path:
        return self.path / "notes.md"

    @property
    def lineage_path(self) -> Path:
        return self.path / "lineage.md"

    @property
    def metadata_path(self) -> Path:
        return self.path / "workspace.json"

    @property
    def program_prompt_path(self) -> Path:
        return self.path / "program.md"

    @property
    def system_prompt_path(self) -> Path:
        return self.path / "system.md"

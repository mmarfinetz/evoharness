from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

from harness.models import (
    BenchmarkResult,
    CandidateSnapshot,
    CandidateStateDetection,
    ObjectiveSpec,
    PopulationMember,
    ResourceRequirements,
    ValidationResult,
    WorkspaceSpec,
)


class ObjectiveAdapter(ABC):
    name: str

    @property
    @abstractmethod
    def objective(self) -> ObjectiveSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def resources(self) -> ResourceRequirements:
        raise NotImplementedError

    @abstractmethod
    def validate_repo(self, repo: Path, args: argparse.Namespace) -> None:
        raise NotImplementedError

    def validate_args(self, args: argparse.Namespace) -> None:
        if args.adapter_config is not None:
            raise SystemExit(
                f"--adapter-config is not supported by adapter '{self.name}': {args.adapter_config}"
            )

    @abstractmethod
    def create_initial_snapshot(
        self,
        repo: Path,
        snapshots_root: Path,
        args: argparse.Namespace,
    ) -> CandidateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def editable_artifacts(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def forbidden_paths(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def detect_candidate_state(self, spec: WorkspaceSpec) -> CandidateStateDetection:
        raise NotImplementedError

    @abstractmethod
    def prepare_candidate_for_benchmark(
        self, spec: WorkspaceSpec, detected_state: CandidateStateDetection
    ) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def pre_benchmark_validate(
        self,
        spec: WorkspaceSpec,
        args: argparse.Namespace,
    ) -> ValidationResult:
        raise NotImplementedError

    @abstractmethod
    def benchmark(
        self,
        spec: WorkspaceSpec,
        args: argparse.Namespace,
    ) -> BenchmarkResult:
        raise NotImplementedError

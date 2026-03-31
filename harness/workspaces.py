from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.adapters.base import ObjectiveAdapter
from harness.models import PopulationMember, WorkspaceSpec


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "harness" / "prompts"
COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "target",
    ".claude",
    ".codex",
)


def ensure_repo_root(repo: Path) -> None:
    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")
    if not repo.is_dir():
        raise SystemExit(f"repo is not a directory: {repo}")


def make_run_dir(args: argparse.Namespace) -> tuple[str, Path, Path]:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "harness" / "runs" / run_name
    workspaces_root = run_dir / "worktrees"
    workspaces_root.mkdir(parents=True, exist_ok=False)
    return run_name, run_dir, workspaces_root


def initialize_workspace(
    *,
    repo: Path,
    spec: WorkspaceSpec,
    adapter: ObjectiveAdapter,
    seed_parent: PopulationMember,
    crossover_parent: PopulationMember | None,
    generation_summary: str,
    args: argparse.Namespace,
) -> None:
    shutil.copytree(repo, spec.path, ignore=COPY_IGNORE)
    adapter_prompt_context = adapter.materialize_workspace(
        repo,
        spec,
        seed_parent.snapshot,
        seed_parent,
        crossover_parent,
        generation_summary,
        args,
    )
    prompt_context = {
        **adapter_prompt_context,
        "candidate_name": spec.name,
        "generation_number": str(spec.generation),
        "mode_name": spec.mode,
        "operator_name": spec.operator,
        "mode_instructions": mode_instructions(spec.mode),
    }
    write_lineage(spec, seed_parent, crossover_parent)
    copy_prompts(spec, prompt_context)
    write_notes_stub(
        spec,
        objective_name=adapter.objective.name,
        objective_direction=adapter.objective.direction,
    )
    write_workspace_metadata(
        spec,
        prompt_context,
        seed_parent=seed_parent,
        crossover_parent=crossover_parent,
    )


def generation_workspace_dir(workspaces_root: Path, generation: int) -> Path:
    generation_dir = workspaces_root / f"g{generation:04d}"
    generation_dir.mkdir(parents=True, exist_ok=False)
    return generation_dir


def copy_prompts(spec: WorkspaceSpec, prompt_context: dict[str, str]) -> None:
    write_prompt(PROMPTS_DIR / "program.md", spec.program_prompt_path, prompt_context)
    shutil.copyfile(PROMPTS_DIR / "system.md", spec.system_prompt_path)


def write_prompt(template_path: Path, destination_path: Path, context: dict[str, str]) -> None:
    text = template_path.read_text()
    for key, value in context.items():
        text = text.replace("{$" + key + "}", value)
    destination_path.write_text(text)


def write_notes_stub(
    spec: WorkspaceSpec,
    *,
    objective_name: str,
    objective_direction: str,
) -> None:
    if spec.notes_path.exists():
        return
    spec.notes_path.write_text(
        textwrap.dedent(
            f"""\
            # Notes

            ## Candidate

            - Name: {spec.name}
            - Generation: {spec.generation}
            - Mode: {spec.mode}
            - Objective: {objective_name} ({objective_direction})

            ## Context

            ## Hypotheses

            ## Experiments

            | # | Changes | Objective | Status | Evidence |
            |---|---------|-----------|--------|----------|

            ## Observations

            ## Best Validated State
            """
        )
    )


def write_lineage(
    spec: WorkspaceSpec,
    parent_a: PopulationMember,
    parent_b: PopulationMember | None,
) -> None:
    lines = [
        "# Lineage",
        "",
        f"- Generation: {spec.generation}",
        f"- Candidate: {spec.name}",
        f"- Mode: {spec.mode}",
        f"- Operator: {spec.operator}",
        f"- Worker slot: {spec.worker_slot}",
        f"- GPU id: {spec.gpu_id}",
        "",
        "## Parent A",
        "",
        f"- Candidate id: {parent_a.candidate_id}",
        f"- Objective: {format_metric(parent_a.objective_value)}",
        f"- Fitness: {format_metric(parent_a.fitness)}",
        f"- Valid: {parent_a.valid}",
        f"- Artifact hash: {parent_a.snapshot.artifact_hash}",
    ]
    if parent_b is not None:
        lines.extend(
            [
                "",
                "## Parent B",
                "",
                f"- Candidate id: {parent_b.candidate_id}",
                f"- Objective: {format_metric(parent_b.objective_value)}",
                f"- Fitness: {format_metric(parent_b.fitness)}",
                f"- Valid: {parent_b.valid}",
                f"- Artifact hash: {parent_b.snapshot.artifact_hash}",
            ]
        )
    spec.lineage_path.write_text("\n".join(lines) + "\n")


def write_workspace_metadata(
    spec: WorkspaceSpec,
    prompt_context: dict[str, str],
    *,
    seed_parent: PopulationMember,
    crossover_parent: PopulationMember | None,
) -> None:
    payload: dict[str, Any] = {
        "generation": spec.generation,
        "candidate_name": spec.name,
        "mode": spec.mode,
        "operator": spec.operator,
        "worker_slot": spec.worker_slot,
        "gpu_id": spec.gpu_id,
        "parent_a": seed_parent.candidate_id,
        "parent_b": None if crossover_parent is None else crossover_parent.candidate_id,
        "prompt_context": prompt_context,
    }
    spec.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def mode_instructions(mode: str) -> str:
    if mode == "mutate":
        return (
            "Start from Parent A's state and make bounded, evidence-driven changes. "
            "Prefer a small number of coherent edits over a full rewrite."
        )
    if mode == "crossover":
        return (
            "Use both parents as semantic inputs. Merge ideas at the file or subsystem level "
            "and do not do blind line-by-line splicing."
        )
    if mode == "repair":
        return (
            "Treat the parent state as mostly correct. Focus on narrow fixes that improve validity "
            "or objective quality without broad churn."
        )
    if mode == "seed":
        return "This workspace exists to benchmark the baseline seed state."
    return "Work conservatively and keep the candidate state coherent."

from __future__ import annotations

import argparse
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from harness.utils import default_uv_bin


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "harness" / "prompts"
COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
)


@dataclass(frozen=True)
class WorktreeSpec:
    generation: int
    name: str
    kind: str
    path: Path
    baseline_train_path: Path
    gpu_id: int

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
    def train_log_path(self) -> Path:
        return self.path / "run.log"

    @property
    def notes_path(self) -> Path:
        return self.path / "notes.md"


def ensure_repo(repo: Path) -> None:
    if not repo.exists():
        raise SystemExit(f"repo does not exist: {repo}")
    if not repo.is_dir():
        raise SystemExit(f"repo is not a directory: {repo}")
    train_path = repo / "train.py"
    if not train_path.exists():
        raise SystemExit(f"repo does not contain train.py: {train_path}")


def make_run_dir(args: argparse.Namespace) -> tuple[str, Path, Path]:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "harness" / "runs" / run_name
    worktrees_root = run_dir / "worktrees"
    worktrees_root.mkdir(parents=True, exist_ok=False)
    return run_name, run_dir, worktrees_root


def create_generation_worktrees(
    repo: Path,
    worktrees_root: Path,
    baseline_train_path: Path,
    generation: int,
    agents: int,
    gpus: int,
) -> list[WorktreeSpec]:
    generation_dir = worktrees_root / f"g{generation:04d}"
    generation_dir.mkdir(parents=True, exist_ok=False)

    specs: list[WorktreeSpec] = []
    parent_spec = WorktreeSpec(
        generation=generation,
        name="parent",
        kind="parent",
        path=generation_dir / "parent",
        baseline_train_path=baseline_train_path,
        gpu_id=0,
    )
    initialize_worktree(repo, parent_spec)
    specs.append(parent_spec)

    for idx in range(1, agents + 1):
        spec = WorktreeSpec(
            generation=generation,
            name=f"candidate-{idx:03d}",
            kind="candidate",
            path=generation_dir / f"candidate-{idx:03d}",
            baseline_train_path=baseline_train_path,
            gpu_id=(idx - 1) % gpus,
        )
        initialize_worktree(repo, spec)
        specs.append(spec)

    return specs


def initialize_worktree(repo: Path, spec: WorktreeSpec) -> None:
    shutil.copytree(repo, spec.path, ignore=COPY_IGNORE)
    baseline_bytes = spec.baseline_train_path.read_bytes()
    (spec.path / "train.py").write_bytes(baseline_bytes)
    (spec.path / "baseline.py").write_bytes(baseline_bytes)
    (spec.path / "best.py").write_bytes(baseline_bytes)
    copy_prompts(spec)


def copy_prompts(spec: WorktreeSpec) -> None:
    write_prompt(
        PROMPTS_DIR / "program.md",
        spec.path / "program.md",
        {
            "gpu_number": str(spec.gpu_id),
            "uv_bin": default_uv_bin(),
        },
    )
    shutil.copyfile(PROMPTS_DIR / "system.md", spec.path / "system.md")
    write_notes_stub(spec.path)


def write_prompt(template_path: Path, destination_path: Path, context: dict[str, str]) -> None:
    text = template_path.read_text()
    for key, value in context.items():
        text = text.replace("{$" + key + "}", value)
    destination_path.write_text(text)


def write_notes_stub(worktree: Path) -> None:
    notes_path = worktree / "notes.md"
    if notes_path.exists():
        return
    notes_path.write_text(
        textwrap.dedent(
            """\
            # Notes

            ## Context

            ## Hypotheses

            ## Experiments

            | # | Changes | val_bpb | Status |
            |---|---------|---------|--------|
            | 0 | Baseline (depth=8, dim=512) | 0.995554 | baseline |

            ## Experiments Observations
            # 0
            # - Success/failure/needs iteration
            # - Hypothesis retrospective
            # - Insights

            # Overall insights

            ## Best Result
            """
        )
    )

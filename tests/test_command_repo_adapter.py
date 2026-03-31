from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from harness.adapters.command_repo import CommandRepoAdapter
from harness.models import PopulationMember, WorkspaceSpec


class CommandRepoAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()
        self._snapshot_index = 0
        self._workspace_index = 0

        self.write_repo_file("README.md", "# Command Repo Demo\n")
        self.write_repo_file("candidate.py", "BIAS = 0.0\n")
        self.write_repo_file("config/weights.json", '{"weight": 1.0}\n')
        self.write_repo_file("evaluate.py", "print('placeholder')\n")
        self.write_repo_file("test_candidate.py", "import unittest\n\nclass CandidateTests(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(True)\n")
        self.write_support_file("support/reference.json", '{"target": 2.5}\n')

    def test_validate_args_requires_adapter_config(self) -> None:
        adapter = CommandRepoAdapter()
        args = argparse.Namespace(adapter_config=None, baseline_train_py=None)
        with self.assertRaises(SystemExit):
            adapter.validate_args(args)

    def test_validate_repo_rejects_missing_required_paths(self) -> None:
        config_path = self.write_config(
            {
                "repo_required_paths": [
                    "README.md",
                    "candidate.py",
                    "missing.py",
                ]
            }
        )
        adapter, args = self.make_adapter(config_path)
        with self.assertRaises(SystemExit):
            adapter.validate_repo(self.repo, args)

    def test_create_initial_snapshot_captures_multiple_artifacts(self) -> None:
        adapter, args = self.make_adapter(self.write_config())
        snapshot = adapter.create_initial_snapshot(
            self.repo,
            self.next_snapshots_root(),
            args,
        )
        self.assertEqual(
            snapshot.artifact_paths,
            ("candidate.py", "config/weights.json"),
        )
        self.assertTrue((snapshot.path / "artifacts" / "candidate.py").exists())
        self.assertTrue((snapshot.path / "artifacts" / "config/weights.json").exists())

    def test_materialize_workspace_copies_support_files(self) -> None:
        adapter, args = self.make_adapter(self.write_config())
        spec = self.make_workspace(adapter, args)

        copied_support = spec.path / ".harness/command_repo/reference.json"
        self.assertTrue(copied_support.exists())
        self.assertIn(".harness/command_repo/reference.json", adapter.forbidden_paths())

    def test_detect_candidate_state_changes_after_live_edit(self) -> None:
        adapter, args = self.make_adapter(self.write_config())
        spec = self.make_workspace(adapter, args)

        initial_state = adapter.detect_candidate_state(spec)
        self.assertFalse(initial_state.has_state)
        self.assertEqual(initial_state.status, "no_state")

        candidate_path = spec.path / "candidate.py"
        candidate_path.write_text("BIAS = 1.25\n")

        changed_state = adapter.detect_candidate_state(spec)
        self.assertTrue(changed_state.has_state)
        self.assertEqual(changed_state.status, "ready")
        self.assertEqual(
            tuple(mapping.source_path for mapping in changed_state.artifact_mappings),
            ("candidate.py", "config/weights.json"),
        )

    def test_pre_benchmark_validate_fails_fast(self) -> None:
        config_path = self.write_config(
            {
                "validation_commands": [
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('first-ran.txt').write_text('ok')",
                    ],
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('validation failed'); sys.exit(11)",
                    ],
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('should-not-run.txt').write_text('no')",
                    ],
                ]
            }
        )
        adapter, args = self.make_adapter(config_path)
        spec = self.make_workspace(adapter, args)

        result = adapter.pre_benchmark_validate(spec, args)

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "validation_failed")
        self.assertTrue((spec.path / "first-ran.txt").exists())
        self.assertFalse((spec.path / "should-not-run.txt").exists())
        self.assertEqual(result.details["returncode"], 11)

    def test_benchmark_reads_metrics_from_output_file(self) -> None:
        output_path = ".harness/command_repo/metrics.json"
        config_path = self.write_config(
            {
                "benchmark_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from pathlib import Path; "
                        "payload={'metrics': {'score': 0.91, 'rmse': 0.09, 'max_abs_error': 0.12}}; "
                        f"path=Path({output_path!r}); "
                        "path.parent.mkdir(parents=True, exist_ok=True); "
                        "path.write_text(json.dumps(payload))"
                    ),
                ],
                "benchmark_output_path": output_path,
                "required_metrics": [
                    "score",
                    "rmse",
                ],
            }
        )
        adapter, args = self.make_adapter(config_path)
        spec = self.make_workspace(adapter, args)

        result = adapter.benchmark(spec, args)

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.objective_value, 0.91)
        self.assertAlmostEqual(result.fitness, 0.91)
        self.assertAlmostEqual(result.metrics["rmse"], 0.09)

    def test_benchmark_reads_stdout_and_enforces_required_metrics(self) -> None:
        config_path = self.write_config(
            {
                "benchmark_command": [
                    sys.executable,
                    "-c",
                    "import json; print(json.dumps({'score': 0.5}))",
                ],
                "benchmark_output_path": None,
                "required_metrics": [
                    "score",
                    "rmse",
                ],
            }
        )
        adapter, args = self.make_adapter(config_path)
        spec = self.make_workspace(adapter, args)

        result = adapter.benchmark(spec, args)

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "parse_failed")
        self.assertEqual(
            result.constraints["metrics_complete"]["missing_metrics"],
            ["rmse"],
        )

    def make_adapter(self, config_path: Path) -> tuple[CommandRepoAdapter, argparse.Namespace]:
        adapter = CommandRepoAdapter()
        args = argparse.Namespace(
            adapter_config=config_path,
            benchmark_timeout_minutes=1,
            baseline_train_py=None,
        )
        adapter.validate_args(args)
        return adapter, args

    def make_workspace(
        self,
        adapter: CommandRepoAdapter,
        args: argparse.Namespace,
    ) -> WorkspaceSpec:
        snapshot = adapter.create_initial_snapshot(self.repo, self.next_snapshots_root(), args)
        parent = PopulationMember(
            candidate_id="seed-baseline",
            generation=0,
            name="seed-baseline",
            snapshot=snapshot,
            operator="seed",
        )
        self._workspace_index += 1
        spec = WorkspaceSpec(
            generation=0,
            index=self._workspace_index,
            name=f"candidate-{self._workspace_index:03d}",
            mode="mutate",
            operator="mutate",
            path=self.root / f"workspace-{self._workspace_index:03d}",
            worker_slot=0,
            gpu_id=None,
        )
        shutil.copytree(self.repo, spec.path)
        adapter.materialize_workspace(
            self.repo,
            spec,
            snapshot,
            parent,
            None,
            "test generation",
            args,
        )
        return spec

    def next_snapshots_root(self) -> Path:
        self._snapshot_index += 1
        path = self.snapshots_root / f"snapshots-{self._snapshot_index:03d}"
        path.mkdir()
        return path

    def write_repo_file(self, relative_path: str, text: str) -> None:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_support_file(self, relative_path: str, text: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def write_config(self, overrides: dict[str, object] | None = None) -> Path:
        payload: dict[str, object] = {
            "objective": {
                "name": "score",
                "direction": "maximize",
                "description": "Higher score is better.",
                "primary_metric": "score",
            },
            "editable_artifacts": [
                "candidate.py",
                "config/weights.json",
            ],
            "context_files": [
                "README.md",
                "evaluate.py",
                "test_candidate.py",
            ],
            "forbidden_paths": [
                "evaluate.py",
            ],
            "support_files": [
                {
                    "source": str(self.root / "support/reference.json"),
                    "destination": ".harness/command_repo/reference.json",
                    "context": False,
                }
            ],
            "validation_commands": [
                [
                    sys.executable,
                    "-c",
                    "print('validation ok')",
                ]
            ],
            "benchmark_command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'score': 0.5, 'rmse': 0.5, 'max_abs_error': 0.5}))",
            ],
            "required_metrics": [
                "score",
                "rmse",
            ],
            "repo_required_paths": [
                "README.md",
                "candidate.py",
                "config/weights.json",
                "evaluate.py",
                "test_candidate.py",
            ],
            "requires_gpu": False,
        }
        if overrides:
            payload.update(overrides)
        config_path = self.root / "adapter.json"
        config_path.write_text(json.dumps(payload, indent=2) + "\n")
        return config_path


if __name__ == "__main__":
    unittest.main()

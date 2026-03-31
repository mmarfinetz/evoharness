from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from harness.adapters.foundry_hook import FoundryHookAdapter
from harness.models import PopulationMember, WorkspaceSpec


class FoundryHookAdapterTests(unittest.TestCase):
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

        self.write_repo_file("README.md", "# Hook Repo\n")
        self.write_repo_file("foundry.toml", "[profile.default]\nsrc = 'src'\n")
        self.write_repo_file("src/LvrHook.sol", "contract LvrHook { function width() external pure returns (uint256) { return 18; } }\n")
        self.write_repo_file("src/LvrPolicy.sol", "contract LvrPolicy { function admission() external pure returns (bool) { return true; } }\n")
        self.write_repo_file("test/LvrHook.t.sol", "contract HookTest {}\n")
        self.write_repo_file("scripts/evaluate_hook.py", "# placeholder\n")
        self.write_support_file("support/HookEvaluator.t.sol", "contract HookEvaluator {}\n")

    def test_validate_args_requires_adapter_config(self) -> None:
        adapter = FoundryHookAdapter()
        args = argparse.Namespace(adapter_config=None, baseline_train_py=None)
        with self.assertRaises(SystemExit):
            adapter.validate_args(args)

    def test_validate_repo_rejects_missing_required_paths(self) -> None:
        config_path = self.write_config(
            {
                "repo_required_paths": [
                    "foundry.toml",
                    "src/LvrHook.sol",
                    "src/Missing.sol",
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
            ("src/LvrHook.sol", "src/LvrPolicy.sol"),
        )
        hook_path = snapshot.path / "artifacts" / "src/LvrHook.sol"
        policy_path = snapshot.path / "artifacts" / "src/LvrPolicy.sol"
        self.assertIn("contract LvrHook", hook_path.read_text())
        self.assertIn("contract LvrPolicy", policy_path.read_text())

    def test_detect_candidate_state_changes_after_live_edit(self) -> None:
        adapter, args = self.make_adapter(self.write_config())
        spec = self.make_workspace(adapter, args)

        initial_state = adapter.detect_candidate_state(spec)
        self.assertFalse(initial_state.has_state)
        self.assertEqual(initial_state.status, "no_state")

        hook_path = spec.path / "src/LvrHook.sol"
        hook_path.write_text("contract LvrHook { function width() external pure returns (uint256) { return 24; } }\n")

        changed_state = adapter.detect_candidate_state(spec)
        self.assertTrue(changed_state.has_state)
        self.assertEqual(changed_state.status, "ready")
        self.assertEqual(
            tuple(mapping.source_path for mapping in changed_state.artifact_mappings),
            ("src/LvrHook.sol", "src/LvrPolicy.sol"),
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
                        "import sys; print('forge test failed'); sys.exit(7)",
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
        self.assertEqual(result.details["returncode"], 7)

    def test_benchmark_reads_metrics_from_output_file(self) -> None:
        output_path = ".harness/foundry_hook/metrics.json"
        config_path = self.write_config(
            {
                "benchmark_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from pathlib import Path; "
                        "payload={'metrics': {'unrecaptured_lvr': 0.25, 'lp_net_from_toxic_flow': 13.5, 'recapture_ratio': 0.6}}; "
                        f"path=Path({output_path!r}); "
                        "path.parent.mkdir(parents=True, exist_ok=True); "
                        "path.write_text(json.dumps(payload))"
                    ),
                ],
                "benchmark_output_path": output_path,
                "required_metrics": [
                    "unrecaptured_lvr",
                    "lp_net_from_toxic_flow",
                    "recapture_ratio",
                ],
            }
        )
        adapter, args = self.make_adapter(config_path)
        spec = self.make_workspace(adapter, args)

        result = adapter.benchmark(spec, args)

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.objective_value, 0.25)
        self.assertAlmostEqual(result.fitness, -0.25)
        self.assertAlmostEqual(result.metrics["lp_net_from_toxic_flow"], 13.5)

    def test_benchmark_reads_stdout_and_enforces_required_metrics(self) -> None:
        config_path = self.write_config(
            {
                "benchmark_command": [
                    sys.executable,
                    "-c",
                    "import json; print(json.dumps({'unrecaptured_lvr': 0.25}))",
                ],
                "benchmark_output_path": None,
                "required_metrics": [
                    "unrecaptured_lvr",
                    "recapture_ratio",
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
            ["recapture_ratio"],
        )

    def test_materialize_workspace_copies_support_files_and_forbids_destinations(self) -> None:
        config_path = self.write_config(
            {
                "support_files": [
                    {
                        "source": "support/HookEvaluator.t.sol",
                        "destination": "test/harness/HookEvaluator.t.sol",
                        "context": True,
                    }
                ]
            }
        )
        adapter, args = self.make_adapter(config_path)
        spec = self.make_workspace(adapter, args)

        copied_support = spec.path / "test/harness/HookEvaluator.t.sol"
        self.assertTrue(copied_support.exists())
        self.assertIn("HookEvaluator", copied_support.read_text())
        self.assertIn("test/harness/HookEvaluator.t.sol", adapter.forbidden_paths())

    def make_adapter(self, config_path: Path) -> tuple[FoundryHookAdapter, argparse.Namespace]:
        adapter = FoundryHookAdapter()
        args = argparse.Namespace(
            adapter_config=config_path,
            benchmark_timeout_minutes=1,
            baseline_train_py=None,
        )
        adapter.validate_args(args)
        return adapter, args

    def make_workspace(
        self,
        adapter: FoundryHookAdapter,
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

    def write_support_file(self, relative_path: str, text: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_config(self, overrides: dict[str, object] | None = None) -> Path:
        payload: dict[str, object] = {
            "objective": {
                "name": "unrecaptured_lvr",
                "direction": "minimize",
                "description": "Lower is better",
                "primary_metric": "unrecaptured_lvr",
            },
            "editable_artifacts": [
                "src/LvrHook.sol",
                "src/LvrPolicy.sol",
            ],
            "context_files": [
                "README.md",
                "foundry.toml",
                "test/LvrHook.t.sol",
                "scripts/evaluate_hook.py",
            ],
            "forbidden_paths": [
                "out",
                "broadcast",
            ],
            "validation_commands": [
                [sys.executable, "-c", "print('forge build ok')"],
                [sys.executable, "-c", "print('forge test ok')"],
            ],
            "benchmark_command": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'unrecaptured_lvr': 0.5, 'recapture_ratio': 0.7}))",
            ],
            "required_metrics": [
                "unrecaptured_lvr",
            ],
            "repo_required_paths": [
                "foundry.toml",
                "src/LvrHook.sol",
                "src/LvrPolicy.sol",
                "test/LvrHook.t.sol",
                "scripts/evaluate_hook.py",
            ],
            "requires_gpu": False,
        }
        if overrides:
            payload.update(overrides)
        config_path = self.root / f"adapter-{len(list(self.root.glob('adapter-*.json'))):03d}.json"
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return config_path


if __name__ == "__main__":
    unittest.main()

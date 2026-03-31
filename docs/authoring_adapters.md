# Authoring adapters

`evo-harness` has two integration levels:

- `command_repo`: use this when your task can be expressed as editable files plus real validation and benchmark commands
- custom Python adapter: use this when you need custom snapshot handling, special benchmark parsing, or repo-specific workspace setup

For most research repos, start with `command_repo`. It is the shortest path from
"I have a measurable objective" to a working evolutionary run.

## Start with `command_repo`

The adapter config lives in JSON and defines:

- `objective`: metric name, direction, description, and primary metric
- `editable_artifacts`: the files the agent is allowed to change
- `context_files`: extra repo files the agent should read before editing
- `forbidden_paths`: files or directories the agent must treat as read-only
- `support_files`: optional config-relative or absolute files copied into each workspace
- `validation_commands`: hard constraints that must pass before benchmarking
- `benchmark_command`: the real evaluator command
- `benchmark_output_path`: optional path to a JSON file written by the benchmark
- `required_metrics`: metrics that must be present in the benchmark JSON
- `repo_required_paths`: required files checked before a run starts
- `requires_gpu`: whether each active worker requires a GPU slot

The benchmark must emit JSON either:

- directly to stdout, or
- to `benchmark_output_path`

Both flat and wrapped payloads are accepted:

```json
{
  "score": 0.91,
  "rmse": 0.09
}
```

```json
{
  "metrics": {
    "score": 0.91,
    "rmse": 0.09
  }
}
```

See [`examples/command_repo_demo/adapter.json`](../examples/command_repo_demo/adapter.json) for a complete minimal example.
If you want a more opinionated advanced example, see the worked Foundry folder
at [`examples/foundry_hook/README.md`](../examples/foundry_hook/README.md).
The Uni V4 folder is kept only as a filled-in case study under
[`case_studies/uni_v4_hook/README.md`](../case_studies/uni_v4_hook/README.md).

## When to write a custom adapter

Write a Python adapter when one or more of these are true:

- the candidate state is not just "these repo files changed"
- the benchmark output needs custom parsing or additional constraint logic
- each workspace needs generated files, copied baselines, or repo-specific reference material
- the benchmark target is not the same as the editable files the agent touched

The adapter interface is defined in [`harness/adapters/base.py`](../harness/adapters/base.py).

At a minimum, a custom adapter decides:

- how to validate the target repo before a run
- which artifacts form the candidate state
- how to materialize baseline and parent state into each workspace
- how to detect a promotable candidate state
- how to run validation
- how to run and parse the benchmark

## Adapter lifecycle

The harness uses adapters in this order:

1. `validate_args`
2. `validate_repo`
3. `create_initial_snapshot`
4. `materialize_workspace`
5. `detect_candidate_state`
6. `prepare_candidate_for_benchmark`
7. `pre_benchmark_validate`
8. `benchmark`

The evolutionary loop itself is generic. Selection, elitism, crossover,
mutation planning, snapshots, and SQLite lineage tracking live outside the
adapter.

## Practical guidance

- Make validation commands strict. They are your hard constraints.
- Keep benchmark commands real. Do not let adapters invent metrics.
- Keep editable artifacts small when possible. Tighter search spaces help.
- Put expensive reference data in `support_files` instead of asking the agent to recreate it.
- Prefer deterministic benchmarks for early runs so improvements are attributable.

## Recommended path

1. Get a baseline benchmark working by hand.
2. Wrap that repo with `command_repo`.
3. Run `--max-generations 0` to verify baseline parsing and artifacts.
4. Run `--dry-run` with a larger population to inspect workspaces and prompts.
5. Only write a custom adapter if `command_repo` cannot express the task cleanly.

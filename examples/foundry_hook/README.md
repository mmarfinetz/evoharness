# Foundry Hook Templates

This folder is the reusable `foundry_hook` starting point.

These configs are patterns, not validated drop-ins. They assume:

- an external Foundry repo with a `foundry.toml`
- one or more editable hook or policy files
- real `forge` validation commands
- a benchmark command that emits JSON metrics

Template configs:

- `fee_policy.json`: optimize a hook fee policy or quoting policy
- `production_policy.json`: optimize the same policy against a stricter production objective
- `auction_design.json`: optimize an auction or fallback script against replay-style scenarios

How to use:

1. Copy the closest JSON file to your own adapter config path.
2. Replace the repo-relative paths in `editable_artifacts`, `context_files`, and `repo_required_paths`.
3. Replace the validation commands with the exact `forge` commands that must pass before benchmarking.
4. Replace the benchmark command with your real evaluator.
5. Replace the metric names and objective description with your repo's actual benchmark contract.
6. Run the harness against your Foundry repo:

```bash
evo-harness run \
  --adapter foundry_hook \
  --repo /path/to/foundry-repo \
  --adapter-config /path/to/adapter.json \
  --population-size 4 \
  --workers 2 \
  --gpus 0
```

Benchmark contract:

- `validation_commands` are hard constraints. If any of them fail, the candidate is rejected.
- `benchmark_command` must emit JSON metrics, either to stdout or to `benchmark_output_path`.
- `required_metrics` must exactly match the metric keys your evaluator produces.

Use `support_files` when the evaluator or fixtures live next to the adapter config
instead of inside the target repo. Each item is copied into the workspace before
validation and benchmarking:

```json
{
  "source": "./support/evaluate_policy.py",
  "destination": ".harness/foundry_hook/evaluate_policy.py",
  "context": true
}
```

If you want to see a filled-in real-repo case study, see
[`../../case_studies/uni_v4_hook/README.md`](../../case_studies/uni_v4_hook/README.md).

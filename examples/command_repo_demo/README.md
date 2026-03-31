# command_repo demo

This example is the public onboarding path for `evo-harness`.

The target repo under `repo/` is deliberately small and CPU-only:

- edit `candidate.py`
- validate with `py_compile` plus a fast unit test
- benchmark with `evaluate.py`, which emits JSON metrics to stdout

Run it from the repository root:

```bash
evo-harness run \
  --adapter command_repo \
  --repo examples/command_repo_demo/repo \
  --adapter-config examples/command_repo_demo/adapter.json \
  --population-size 2 \
  --workers 1 \
  --gpus 0 \
  --dry-run
```

Then benchmark the seed:

```bash
evo-harness run \
  --adapter command_repo \
  --repo examples/command_repo_demo/repo \
  --adapter-config examples/command_repo_demo/adapter.json \
  --population-size 1 \
  --workers 1 \
  --gpus 0 \
  --max-generations 0
```

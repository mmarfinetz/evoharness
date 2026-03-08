# autoharness
`autoharness` is meant to make `autoresearch` easier to run in parallel, keep running for long stretches, and require less human supervision once the machine is prepared.

The harness creates isolated git worktrees, launches one Claude agent per GPU per worktree, benchmarks each agent's results, finds the best, and starts a new round of agents to build off of the last round's best. In this way you can have several agents running parallel  experiments and choose whatever works the best to build off of.

## Requirements
- a Linux server with NVIDIA GPUs, ideally H100s
- Python 3.10+
- git
- a Claude Code install and login

`setup.sh` installs the missing pieces this repo expects, syncs the `autoresearch` environment, and prepares its data.

## Quick start

```bash
git clone --recurse-submodules <repo-url>
cd autoharness

./setup.sh
```

After running the setup script you will need to sign in to your Claude account (or use an API key).
```bash
claude
```

Now you can run the harness with:
```bash
python3 -m harness.run
```

Notes:
- `setup.sh` initializes the `autoresearch` submodule, installs `uv`, syncs dependencies, and runs `autoresearch/prepare.py`
- by default, the harness auto-detects the available GPU count and uses that for both `--gpus` and `--agents`

## Common commands

Run a single short generation:

```bash
python3 -m harness.run --max-generations 1 --agent-timeout-minutes 20
```

Use a different `train.py` than the default (e.g. to have your runs build off of each other):
```bash
python3 -m harness.run --baseline-train-py /path/to/custom-train.py
```


## Repo layout

- `harness/`: orchestration code for worktrees, agent execution, benchmarking, and result tracking
- `autoresearch/`: the training project being optimized
- `setup.sh`: machine bootstrap for a fresh server

## Related project

For the training code and the underlying research loop, see [`autoresearch/README.md`](./autoresearch/README.md).

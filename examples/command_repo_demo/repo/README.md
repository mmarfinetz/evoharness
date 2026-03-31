# Toy Curve Fit

This repo exists to demonstrate the generic `command_repo` adapter.

The candidate state is `candidate.py`. The benchmark in `evaluate.py` compares
`predict(x)` against a fixed reference curve and emits JSON metrics:

- `score`: higher is better
- `rmse`: lower is better
- `max_abs_error`: lower is better

Hard constraints:

- `candidate.py` must compile
- `python3 -m unittest -q test_candidate.py` must pass

The target curve is visible in `evaluate.py`. This is intentional. The goal of
the example is not secrecy; it is to show how to package a repo so the harness
can optimize it.

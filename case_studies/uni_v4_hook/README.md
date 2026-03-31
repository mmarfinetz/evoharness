# Foundry Case Study

This folder is a concrete case study for one external Foundry hook repo. It is
intentionally specific in file paths, tests, and evaluator support code.

For the reusable starting point, use
[`../../examples/foundry_hook/README.md`](../../examples/foundry_hook/README.md). This folder exists to
show what those generic patterns look like once they are filled in for a real
repo.

Configs:

- `fee_policy.json`: evolve a hook fee policy against a code-coupled evaluator
- `production_policy.json`: evolve the same fee policy against a stricter production-style objective
- `auction_design.json`: evolve an auction mechanism or fallback script against replay-based scenario windows

Use these as reference patterns, not as drop-in configs for unrelated repos.

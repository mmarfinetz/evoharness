from __future__ import annotations

from harness.adapters.base import ObjectiveAdapter
from harness.adapters.command_repo import CommandRepoAdapter
from harness.adapters.foundry_hook import FoundryHookAdapter


ADAPTERS: dict[str, ObjectiveAdapter] = {
    "command_repo": CommandRepoAdapter(),
    "foundry_hook": FoundryHookAdapter(),
}


def get_adapter(name: str) -> ObjectiveAdapter:
    adapter = ADAPTERS.get(name)
    if adapter is None:
        available = ", ".join(sorted(ADAPTERS))
        raise SystemExit(f"unknown adapter '{name}'. Available adapters: {available}")
    return adapter

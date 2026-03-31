from __future__ import annotations

import random
from typing import Iterable, Sequence

from harness.models import PopulationMember


def candidate_sort_key(member: PopulationMember) -> tuple[object, ...]:
    objective_value = member.objective_value if member.objective_value is not None else float("inf")
    fitness = member.fitness if member.fitness is not None else float("-inf")
    artifact_hash = member.snapshot.artifact_hash or ""
    return (
        0 if member.valid else 1,
        -fitness,
        objective_value,
        artifact_hash,
        member.candidate_id,
    )


def rank_population(population: Iterable[PopulationMember]) -> list[PopulationMember]:
    return sorted(population, key=candidate_sort_key)


def select_parent(
    population: Sequence[PopulationMember],
    strategy: str,
    rng: random.Random,
) -> PopulationMember:
    ranked = rank_population([member for member in population if member.valid])
    if not ranked:
        raise SystemExit("cannot select a parent from an empty population")
    if strategy == "rank":
        weights = list(range(len(ranked), 0, -1))
        return rng.choices(ranked, weights=weights, k=1)[0]
    if strategy == "tournament":
        competitors = rng.sample(ranked, k=min(3, len(ranked)))
        return rank_population(competitors)[0]
    raise SystemExit(f"unsupported selection strategy: {strategy}")


def select_parent_pair(
    population: Sequence[PopulationMember],
    strategy: str,
    rng: random.Random,
) -> tuple[PopulationMember, PopulationMember]:
    valid_population = [member for member in population if member.valid]
    if len(valid_population) < 2:
        raise SystemExit("crossover requires at least two population members")
    first = select_parent(valid_population, strategy, rng)
    remainder = [
        member
        for member in valid_population
        if member.candidate_id != first.candidate_id
    ]
    second = select_parent(remainder, strategy, rng)
    return first, second


def select_survivors(
    population: Sequence[PopulationMember],
    *,
    target_size: int,
    elite_ids: set[str],
) -> list[PopulationMember]:
    ranked = rank_population([member for member in population if member.valid])
    if not ranked:
        return []

    survivors: list[PopulationMember] = []
    seen_hashes: set[str] = set()

    for member in ranked:
        if member.candidate_id in elite_ids:
            survivors.append(member)
            seen_hashes.add(member.snapshot.artifact_hash)
    for member in ranked:
        if member.candidate_id in elite_ids:
            continue
        if len(survivors) >= target_size:
            break
        artifact_hash = member.snapshot.artifact_hash
        if artifact_hash in seen_hashes:
            continue
        survivors.append(member)
        seen_hashes.add(artifact_hash)

    return survivors[:target_size]

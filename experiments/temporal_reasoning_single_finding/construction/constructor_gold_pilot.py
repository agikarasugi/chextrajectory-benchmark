"""Construct Gold pilot v1 with one balanced eligible task per trajectory."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..protocol.contracts import TASKS
from ..protocol.task_selection import GOLD_PILOT_BENCHMARK_VERSION
from .benchmark_artifacts import (
    DEFAULT_AUDIT_SIDECAR, DEFAULT_BENCHMARK_ROOT,
    DEFAULT_FOUR_TP_SOURCE, DEFAULT_THREE_TP_SOURCE,
)
from .gold_common import load_gold_population, write_gold_benchmark
from .source_data import SOURCE_CONTRACT_RECONCILIATION


ASSIGNMENT_SEED = 42
ASSIGNMENT_METHOD = "most-severe-first-shuffle-v1"
TASK_QUOTAS = {
    3: dict(zip(TASKS, (84, 84, 83, 84), strict=True)),
    4: dict(zip(TASKS, (44, 44, 45, 45), strict=True)),
}


def assign_tasks(
    items: list[dict[str, Any]], quotas: dict[str, int], *, seed: int = ASSIGNMENT_SEED
) -> list[dict[str, Any]]:
    """Assign all items exactly once, without altering reference eligibility."""
    if set(quotas) != set(TASKS) or any(
        type(value) is not int or value < 0 for value in quotas.values()
    ):
        raise ValueError("Task quotas must be non-negative integers for all four tasks")
    if sum(quotas.values()) != len(items):
        raise ValueError("Task quotas must account for every valid Gold trajectory")
    if len({item["trajectory_id"] for item in items}) != len(items):
        raise ValueError("Duplicate trajectory ID in task assignment")
    if any("assigned_task" in item for item in items):
        raise ValueError("Task assignment requires previously unassigned source items")
    ordered = sorted(items, key=lambda item: item["trajectory_id"])
    shuffled = list(ordered)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    severe = "most_severe_visit_or_plateau"
    eligible = [item for item in shuffled if item["tasks"][severe]["eligible"]]
    if len(eligible) < quotas[severe]:
        raise ValueError("Insufficient eligible Most Severe trajectories for its quota")
    assignments = {item["trajectory_id"]: severe for item in eligible[:quotas[severe]]}
    remaining = [item for item in shuffled if item["trajectory_id"] not in assignments]
    rng.shuffle(remaining)
    offset = 0
    for task in TASKS:
        if task == severe:
            continue
        for item in remaining[offset:offset + quotas[task]]:
            if not item["tasks"][task]["eligible"]:
                raise ValueError(f"Cannot assign ineligible task: {task}")
            assignments[item["trajectory_id"]] = task
        offset += quotas[task]
    if Counter(assignments.values()) != Counter(quotas):
        raise ValueError("Task assignment did not fill all quotas")
    return [
        dict(item, assigned_task=assignments[item["trajectory_id"]])
        for item in ordered
    ]


def construct_gold_pilot(
    *, three_tp_source: Path = DEFAULT_THREE_TP_SOURCE,
    four_tp_source: Path = DEFAULT_FOUR_TP_SOURCE,
    audit_sidecar: Path = DEFAULT_AUDIT_SIDECAR,
    output_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, Any]:
    """Retain the complete Gold population and freeze the approved task allocation."""
    items, provenance, counts = load_gold_population(
        three_tp_source=three_tp_source, four_tp_source=four_tp_source,
        audit_sidecar=audit_sidecar, benchmark_version=GOLD_PILOT_BENCHMARK_VERSION,
    )
    assigned = {
        count: assign_tasks(items[count], TASK_QUOTAS[count]) for count in (3, 4)
    }
    return write_gold_benchmark(
        output_root=output_root, directory_name="gold_pilot_v1",
        benchmark_version=GOLD_PILOT_BENCHMARK_VERSION,
        items=assigned, provenance=provenance,
        construction={
            "kind": "gold_pilot",
            "selection_rule": "all-valid-gold-v1",
            "gold_source_counts": counts,
            "source_contract_reconciliation": SOURCE_CONTRACT_RECONCILIATION,
            "task_assignment": {
                "method": ASSIGNMENT_METHOD, "seed": ASSIGNMENT_SEED,
                "remaining_task_order": [task for task in TASKS if task != TASKS[2]],
                "quotas": {f"{count}tp": TASK_QUOTAS[count] for count in (3, 4)},
            },
        },
    )


def main() -> None:
    """Construct Gold pilot v1 without image access or inference."""
    result = construct_gold_pilot()
    print(json.dumps({"gold_pilot_manifest": result["gold_manifest_path"]}, indent=2))


if __name__ == "__main__":
    main()

"""Explicit final-task selection, separate from reference eligibility."""

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ANNOTATION_SOURCES, NO_IDENTIFIABLE, PILOT_BENCHMARK_VERSION,
    SMOKE_BENCHMARK_VERSION, TASKS,
)
from .task_rules import derive_answer


GOLD_PILOT_BENCHMARK_VERSION = "temporal-reasoning-gold-pilot-v1"
SMOKE_V2_BENCHMARK_VERSION = "temporal-reasoning-smoke-v2"
TASK_SELECTION_CONTRACT = "one-eligible-task-v1"
ALL_TASK_VERSIONS = {
    PILOT_BENCHMARK_VERSION, SMOKE_BENCHMARK_VERSION, "temporal-reasoning-gold-v1"
}


def assigned_task(item: Mapping[str, Any]) -> str | None:
    """Validate one optional assignment on a benchmark item or prepared SEQ input."""
    if "assigned_task" not in item:
        return None
    task = item["assigned_task"]
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError("assigned_task must name exactly one supported task")
    if derive_answer(task, item["reference_change_labels"]) == NO_IDENTIFIABLE:
        raise ValueError("assigned_task must be reference-eligible")
    if "tasks" in item and item["tasks"][task]["eligible"] is not True:
        raise ValueError("assigned_task conflicts with task eligibility")
    return task


def selected_tasks(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Return assigned tasks, or every eligible task for legacy benchmarks."""
    task = assigned_task(item)
    if task is not None:
        return (task,)
    return tuple(task for task in TASKS if item["tasks"][task]["eligible"])


def selection_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize selected questions without changing any eligibility counts."""
    counts = {
        f"{length}tp_{tier}": {task: 0 for task in TASKS}
        for length in (3, 4) for tier in ANNOTATION_SOURCES
    }
    for item in items:
        task = assigned_task(item)
        if task is None:
            raise ValueError("Every item must have an assigned_task")
        counts[f"{item['timepoint_count']}tp_{item['annotation_source']}"][task] += 1
    return {
        "contract": TASK_SELECTION_CONTRACT,
        "selected_final_cases": len(items),
        "selected_by_cohort": counts,
    }


def validate_task_selection(
    manifest: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> None:
    """Check explicit selection and its counts, or preserve the all-task contract."""
    version = manifest["benchmark_version"]
    enabled = "task_selection" in manifest
    if version in {GOLD_PILOT_BENCHMARK_VERSION, SMOKE_V2_BENCHMARK_VERSION} and not enabled:
        raise ValueError("Assigned benchmarks require an explicit task_selection contract")
    if enabled and version in ALL_TASK_VERSIONS:
        raise ValueError("Frozen all-task benchmarks cannot add task_selection")
    if not enabled:
        if any("assigned_task" in item for item in items):
            raise ValueError("assigned_task requires a manifest task_selection contract")
        return
    actual = manifest["task_selection"]
    expected = selection_summary(items)
    # Reject booleans masquerading as integer counts, as well as missing/extra keys.
    def same(value: Any, reference: Any) -> bool:
        """Compare exact JSON shapes and scalar types recursively."""
        if type(value) is not type(reference):
            return False
        if isinstance(reference, dict):
            return value.keys() == reference.keys() and all(
                same(value[key], child) for key, child in reference.items()
            )
        return value == reference

    if not same(actual, expected):
        raise ValueError("Manifest task_selection disagrees with assigned tasks")
    if version == SMOKE_V2_BENCHMARK_VERSION:
        quotas = {
            f"{length}tp_{tier}": {
                task: 2 if tier == "gold" else 0 for task in TASKS
            }
            for length in (3, 4) for tier in ANNOTATION_SOURCES
        }
        if expected["selected_by_cohort"] != quotas:
            raise ValueError("Smoke v2 requires two Gold items per task and length")


def unreduced_call_ceiling(items: Sequence[Mapping[str, Any]]) -> int:
    """Count calls before PAIR reuse, excluding unselected tasks for assigned items."""
    return sum(
        item["timepoint_count"] + (4 if assigned_task(item) is not None else 16)
        for item in items
    )

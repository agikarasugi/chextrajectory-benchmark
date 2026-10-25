"""Reference-answer consistency checks for frozen benchmark items."""

from collections.abc import Iterable, Mapping
from typing import Any

from ..protocol.contracts import (
    MOST_SEVERE_EXCLUSION,
    NO_IDENTIFIABLE,
    TASKS,
    answer_choices,
    require_exact_keys,
    validate_change_labels,
    validate_timepoint_count,
)
from ..protocol.task_rules import derive_answer


REFERENCE_CHECK_FIELDS = {
    "status",
    "checked_task_entries",
    "eligible_final_cases",
    "ineligible_most_severe_task_entries",
    "mismatches",
}
TASK_ENTRY_FIELDS = {"eligible", "reference_answer", "exclusion_reason"}


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return one ordinary mapping or reject the value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _check_task_entry(
    *,
    trajectory_id: str,
    task: str,
    entry: Any,
    derived_answer: str,
    timepoint_count: int,
) -> bool:
    """Validate one stored task entry and return whether it is eligible."""
    task_entry = _require_mapping(entry, f"{trajectory_id}.{task}")
    require_exact_keys(task_entry, TASK_ENTRY_FIELDS, f"{trajectory_id}.{task}")

    eligible = task_entry["eligible"]
    if type(eligible) is not bool:
        raise ValueError(f"{trajectory_id}.{task}.eligible must be a Boolean")
    reference_answer = task_entry["reference_answer"]
    exclusion_reason = task_entry["exclusion_reason"]

    if eligible:
        if exclusion_reason is not None:
            raise ValueError(
                f"{trajectory_id}.{task} is eligible but has an exclusion reason"
            )
        if not isinstance(reference_answer, str):
            raise ValueError(
                f"{trajectory_id}.{task} has no concrete reference answer"
            )
        if reference_answer not in answer_choices(task, timepoint_count):
            raise ValueError(
                f"{trajectory_id}.{task} reference answer is outside the vocabulary"
            )
        if derived_answer == NO_IDENTIFIABLE:
            raise ValueError(
                f"{trajectory_id}.{task} is eligible but fixed code finds no answer"
            )
        if reference_answer != derived_answer:
            raise ValueError(
                f"{trajectory_id}.{task} reference mismatch: "
                f"stored={reference_answer!r}, derived={derived_answer!r}"
            )
        return True

    if task != "most_severe_visit_or_plateau":
        raise ValueError(f"{trajectory_id}.{task} cannot be ineligible")
    if reference_answer is not None:
        raise ValueError(f"{trajectory_id}.{task} ineligible answer must be null")
    if exclusion_reason != MOST_SEVERE_EXCLUSION:
        raise ValueError(f"{trajectory_id}.{task} has the wrong exclusion reason")
    if derived_answer != NO_IDENTIFIABLE:
        raise ValueError(
            f"{trajectory_id}.{task} is excluded but fixed code derives an answer"
        )
    return False


def run_reference_check(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce every stored task answer and return the manifest summary."""
    checked_task_entries = 0
    eligible_final_cases = 0
    ineligible_most_severe = 0

    for item_index, raw_item in enumerate(items):
        item = _require_mapping(raw_item, f"benchmark item {item_index}")
        trajectory_id = item.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"Benchmark item {item_index} has no trajectory_id")
        timepoint_count = item.get("timepoint_count")
        validate_timepoint_count(timepoint_count)
        raw_labels = item.get("reference_change_labels")
        if not isinstance(raw_labels, list):
            raise ValueError(
                f"{trajectory_id}.reference_change_labels must be an array"
            )
        labels = validate_change_labels(
            raw_labels,
            timepoint_count,
        )
        tasks = _require_mapping(item.get("tasks"), f"{trajectory_id}.tasks")
        require_exact_keys(tasks, set(TASKS), f"{trajectory_id}.tasks")

        for task in TASKS:
            checked_task_entries += 1
            derived_answer = derive_answer(task, labels)
            if _check_task_entry(
                trajectory_id=trajectory_id,
                task=task,
                entry=tasks[task],
                derived_answer=derived_answer,
                timepoint_count=timepoint_count,
            ):
                eligible_final_cases += 1
            else:
                ineligible_most_severe += 1

    return {
        "status": "passed",
        "checked_task_entries": checked_task_entries,
        "eligible_final_cases": eligible_final_cases,
        "ineligible_most_severe_task_entries": ineligible_most_severe,
        "mismatches": 0,
    }


def validate_reference_check_summary(value: Any) -> dict[str, Any]:
    """Validate and normalize one manifest Reference Check summary."""
    summary = _require_mapping(value, "reference_check")
    require_exact_keys(summary, REFERENCE_CHECK_FIELDS, "reference_check")
    if summary["status"] != "passed":
        raise ValueError("Manifest Reference Check status is not passed")
    for field in REFERENCE_CHECK_FIELDS - {"status"}:
        count = summary[field]
        if type(count) is not int or count < 0:
            raise ValueError(f"reference_check.{field} must be a non-negative integer")
    if summary["mismatches"] != 0:
        raise ValueError("Manifest Reference Check records mismatches")
    return summary

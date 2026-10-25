"""Deterministic implementations of the four frozen final tasks."""

from collections.abc import Sequence

from .contracts import (
    NO_IDENTIFIABLE,
    answer_choices,
    validate_change_labels,
    validate_task,
)


def trajectory_shape(change_labels: Sequence[str]) -> str:
    """Classify the trajectory after ignoring unchanged intervals."""
    labels = validate_change_labels(change_labels)
    active = [label for label in labels if label != "unchanged"]
    if not active:
        return "constant"
    collapsed = [active[0]]
    for label in active[1:]:
        if label != collapsed[-1]:
            collapsed.append(label)
    if len(collapsed) == 1:
        return (
            "monotonic_worsening"
            if collapsed[0] == "deterioration"
            else "monotonic_improvement"
        )
    if len(collapsed) == 2:
        return "single_peak" if collapsed[0] == "deterioration" else "single_trough"
    return "fluctuating"


def turning_regions(change_labels: Sequence[str]) -> str:
    """Identify every switch between opposing active directions."""
    labels = validate_change_labels(change_labels)
    active = [
        (index, label)
        for index, label in enumerate(labels)
        if label != "unchanged"
    ]
    regions: list[str] = []
    for (previous_index, previous), (current_index, current) in zip(
        active,
        active[1:],
        strict=False,
    ):
        if previous == current:
            continue
        kind = "peak" if previous == "deterioration" else "trough"
        first_visit = previous_index + 2
        last_visit = current_index + 1
        visit = (
            f"T{first_visit}"
            if first_visit == last_visit
            else f"T{first_visit}-T{last_visit}"
        )
        regions.append(f"{kind}:{visit}")
    return ",".join(regions) if regions else "none"


def most_severe_visit_or_plateau(change_labels: Sequence[str]) -> str:
    """Return the uniquely identifiable maximum region or no-answer status."""
    labels = validate_change_labels(change_labels)
    shape = trajectory_shape(labels)
    if shape in {"single_trough", "fluctuating"}:
        return NO_IDENTIFIABLE
    deterioration = [
        index for index, label in enumerate(labels) if label == "deterioration"
    ]
    improvement = [
        index for index, label in enumerate(labels) if label == "improvement"
    ]
    first_visit = deterioration[-1] + 2 if deterioration else 1
    last_visit = improvement[0] + 1 if improvement else len(labels) + 1
    return (
        f"T{first_visit}"
        if first_visit == last_visit
        else f"T{first_visit}-T{last_visit}"
    )


def first_improvement_between_visits(change_labels: Sequence[str]) -> str:
    """Return the earliest improving interval, or none."""
    labels = validate_change_labels(change_labels)
    for index, label in enumerate(labels, start=1):
        if label == "improvement":
            return f"T{index}-T{index + 1}"
    return "none"


def derive_answer(task: str, change_labels: Sequence[str]) -> str:
    """Derive one task answer from a validated change-label vector."""
    validate_task(task)
    labels = validate_change_labels(change_labels)
    functions = {
        "trajectory_shape": trajectory_shape,
        "turning_regions": turning_regions,
        "most_severe_visit_or_plateau": most_severe_visit_or_plateau,
        "first_improvement_between_visits": first_improvement_between_visits,
    }
    answer = functions[task](labels)
    if (
        answer != NO_IDENTIFIABLE
        and answer not in answer_choices(task, len(labels) + 1)
    ):
        raise AssertionError(f"Derived answer outside frozen vocabulary: {answer}")
    return answer

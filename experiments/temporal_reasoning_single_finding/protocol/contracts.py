"""Frozen names, vocabularies, and structural contracts."""

from collections.abc import Sequence
from typing import Any


PILOT_BENCHMARK_VERSION = "temporal-reasoning-pilot-v1"
SMOKE_BENCHMARK_VERSION = "temporal-reasoning-smoke-v1"
PAIR_PROMPT_VERSION = "pair-change-v1"
SEQ_PROMPT_VERSION = "sequence-change-v1"
FINAL_PROMPT_VERSION = "final-tasks-v1"
JSON_SCHEMA_VERSION = "v1"
DATASET = "chexpert_plus"
IMAGE_PATH_CONTRACT = "source-jpg-to-png-final-suffix-v1"
LC_INFERENCE_POLICY = "predicted-labels-v1"
REFERENCE_SUPPORTED_LC_POLICY = "reference-supported-answer-v1"
LC_INFERENCE_POLICIES = (LC_INFERENCE_POLICY, REFERENCE_SUPPORTED_LC_POLICY)
UPSTREAM_ANSWER_MISMATCH = "upstream_answer_mismatch"

CHANGE_LABELS = ("deterioration", "unchanged", "improvement")
SOURCE_LABELS = ("New", "Worse", "Stable", "Improved", "Resolved")
ANNOTATION_SOURCES = ("silver", "gold")

TASKS = (
    "trajectory_shape",
    "turning_regions",
    "most_severe_visit_or_plateau",
    "first_improvement_between_visits",
)
SETTINGS = (
    "SEQ-DIR",
    "PAIR-FC",
    "REF-LC",
    "SEQ-FC",
    "PAIR-LC",
    "SEQ-LC",
)
MODEL_BACKED_SETTINGS = ("SEQ-DIR", "REF-LC", "PAIR-LC", "SEQ-LC")
FIXED_CODE_SETTINGS = ("PAIR-FC", "SEQ-FC")

STATUSES = (
    "valid",
    "invalid_response",
    "response_error",
    "upstream_invalid",
    "upstream_response_error",
    "no_identifiable_most_severe_visit_or_plateau",
    UPSTREAM_ANSWER_MISMATCH,
)
FAILURE_REASONS = (
    "refusal",
    "content_filter",
    "no_json_object",
    "malformed_json",
    "multiple_json_objects",
    "duplicate_json_key",
    "schema_mismatch",
)
NO_IDENTIFIABLE = "no_identifiable_most_severe_visit_or_plateau"
MOST_SEVERE_EXCLUSION = "no_unique_most_severe_visit_or_plateau"


FINDING_DISPLAY_NAMES = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "edema",
    "enlarged_cardiomediastinum": "enlarged cardiomediastinum",
    "lung_lesion": "lung lesion",
    "lung_opacity": "lung opacity",
    "pleural_effusion": "pleural effusion",
    "pleural_other": "pleural other",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}

SOURCE_TO_CHANGE_LABEL = {
    "New": "deterioration",
    "Worse": "deterioration",
    "Stable": "unchanged",
    "Improved": "improvement",
    "Resolved": "improvement",
}


def lc_skip_status(
    fc_status: str,
    fc_answer: str | None,
    reference_answer: str,
    policy: str | None,
) -> str | None:
    """Apply the saved LC policy, with absent policy retaining legacy behavior."""
    if policy is not None and policy not in LC_INFERENCE_POLICIES:
        raise ValueError("Unsupported LC inference policy")
    if fc_status == NO_IDENTIFIABLE and policy == LC_INFERENCE_POLICY:
        return None
    if fc_status != "valid":
        return fc_status
    if policy == REFERENCE_SUPPORTED_LC_POLICY and fc_answer != reference_answer:
        return UPSTREAM_ANSWER_MISMATCH
    return None


def interval_keys(timepoint_count: int) -> tuple[str, ...]:
    """Return the exact chronological SEQ response keys."""
    validate_timepoint_count(timepoint_count)
    return tuple(
        f"T{index}_to_T{index + 1}"
        for index in range(1, timepoint_count)
    )


def answer_choices(task: str, timepoint_count: int) -> tuple[str, ...]:
    """Return the frozen answer vocabulary for one task and sequence length."""
    validate_task(task)
    validate_timepoint_count(timepoint_count)
    if task == "trajectory_shape":
        choices = (
            "constant",
            "monotonic_worsening",
            "monotonic_improvement",
            "single_peak",
            "single_trough",
        )
        return choices if timepoint_count == 3 else choices + ("fluctuating",)
    if task == "turning_regions":
        if timepoint_count == 3:
            return ("none", "peak:T2", "trough:T2")
        return (
            "none",
            "peak:T2",
            "peak:T2,trough:T3",
            "peak:T2-T3",
            "peak:T3",
            "trough:T2",
            "trough:T2,peak:T3",
            "trough:T2-T3",
            "trough:T3",
        )
    if task == "most_severe_visit_or_plateau":
        if timepoint_count == 3:
            return ("T1", "T1-T2", "T1-T3", "T2", "T2-T3", "T3")
        return (
            "T1",
            "T1-T2",
            "T1-T3",
            "T1-T4",
            "T2",
            "T2-T3",
            "T2-T4",
            "T3",
            "T3-T4",
            "T4",
        )
    return tuple(
        [*(f"T{index}-T{index + 1}" for index in range(1, timepoint_count)), "none"]
    )


def validate_change_labels(
    labels: Sequence[str],
    timepoint_count: int | None = None,
) -> tuple[str, ...]:
    """Validate and normalize one chronological change-label vector."""
    if isinstance(labels, (str, bytes)):
        raise ValueError("Change labels must be a sequence of strings")
    result = tuple(labels)
    if timepoint_count is not None:
        validate_timepoint_count(timepoint_count)
        if len(result) != timepoint_count - 1:
            raise ValueError("Change-label count does not match timepoint count")
    if not result or any(label not in CHANGE_LABELS for label in result):
        raise ValueError(f"Unsupported change-label vector: {result}")
    return result


def validate_task(task: str) -> None:
    """Reject a task outside the frozen contract."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")


def validate_timepoint_count(timepoint_count: int) -> None:
    """Reject a sequence length outside the current contract."""
    if type(timepoint_count) is not int or timepoint_count not in (3, 4):
        raise ValueError(f"Unsupported timepoint count: {timepoint_count}")


def require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    """Require a mapping to contain exactly the declared fields."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")

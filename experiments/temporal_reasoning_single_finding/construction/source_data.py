"""Load and validate the frozen single-finding source trajectories."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from experiments.temporal_reasoning_single_finding.artifact_io import read_json
from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    ANNOTATION_SOURCES,
    DATASET,
    FINDING_DISPLAY_NAMES,
    MOST_SEVERE_EXCLUSION,
    NO_IDENTIFIABLE,
    SOURCE_LABELS,
    SOURCE_TO_CHANGE_LABEL,
    TASKS,
    require_exact_keys,
    validate_timepoint_count,
)
from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
    derive_answer,
)


SOURCE_RECORD_KEYS = {
    "sequence_id",
    "image_sequence_id",
    "patient_id",
    "tier",
    "finding",
    "timepoints",
    "change_labels",
}
SOURCE_TIMEPOINT_KEYS = {
    "timepoint",
    "study_id",
    "image_path",
    "patient_report_date_order_min",
    "patient_report_date_order_max",
}
AUDIT_COLUMNS = (
    "sequence_id",
    "image_sequence_id",
    "tier",
    "time_points",
    "patient_id",
    "finding_token",
    "study_ids",
    "image_paths",
    "study_order_min",
    "study_order_max",
    "transition_labels",
    "is_unambiguous",
    "all_images_match_csv",
)
QUALITY_EXCLUSION_REASONS = (
    "invalid_chronology",
    "non_frontal_images",
    "repeated_images",
    "source_image_not_listed",
    "state_inconsistent",
)
SOURCE_CONTRACT_RECONCILIATION = (
    "The two JSON artifacts are the sole source of benchmark content. The "
    "audit Parquet is used only as a read-only validation sidecar because the "
    "JSON export omits is_unambiguous and all_images_match_csv."
)

_LOWER_HEX_ID = re.compile(r"[0-9a-f]{32}")
_FRONTAL_FILENAME = re.compile(r"view[0-9]+_frontal[.]jpg")
_PATIENT_ID = re.compile(r"patient[0-9]+")
_STUDY_ID = re.compile(r"study[0-9]+")
_SOURCE_LABEL_STATES = {
    "New": (False, True),
    "Worse": (True, True),
    "Stable": (True, True),
    "Improved": (True, True),
    "Resolved": (True, False),
}


@dataclass(frozen=True, slots=True)
class SourceTimepoint:
    """One source visit with image identity and chronology metadata."""

    timepoint: str
    study_id: str
    image_path: str
    order_min: int
    order_max: int


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """One fully validated trajectory available to benchmark construction."""

    source_sequence_id: str
    image_sequence_id: str
    patient_id: str
    annotation_source: str
    timepoint_count: int
    target_finding: str
    timepoints: tuple[SourceTimepoint, ...]
    source_interval_labels: tuple[str, ...]

    @property
    def trajectory_id(self) -> str:
        """Return the approved dataset-namespaced source identity."""
        return f"{DATASET}:{self.source_sequence_id}"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Relevant fields from one row of the read-only audit sidecar."""

    sequence_id: str
    image_sequence_id: str
    annotation_source: str
    timepoint_count: int
    patient_id: str
    target_finding: str
    study_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    order_mins: tuple[int, ...]
    order_maxs: tuple[int, ...]
    transition_labels: tuple[str, ...] | None
    is_unambiguous: bool
    all_images_match_csv: bool


@dataclass(frozen=True, slots=True)
class AuditIndex:
    """Validated audit entries indexed by source sequence ID."""

    entries: dict[str, AuditEntry]
    record_count: int


@dataclass(frozen=True, slots=True)
class SourceLoadResult:
    """Eligible candidates and auditable source-filter counts."""

    candidates: tuple[SourceCandidate, ...]
    source_record_count: int
    excluded_record_count: int
    quality_exclusion_counts: dict[str, int]
    interval_references: dict[tuple[str, str, str, str], str]
    counts_by_annotation_source: dict[str, dict[str, Any]]


def _require_object(value: Any, name: str) -> dict[str, Any]:
    """Return an object or raise a contextual source error."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    """Return an array or raise a contextual source error."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _require_string(value: Any, name: str) -> str:
    """Return a non-empty string or raise a contextual source error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_integer(value: Any, name: str) -> int:
    """Return an integer while rejecting booleans and numeric coercion."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_boolean(value: Any, name: str) -> bool:
    """Return a boolean without truth-value coercion."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Return a tuple containing only non-empty strings."""
    values = _require_list(value, name)
    return tuple(
        _require_string(item, f"{name}[{index}]")
        for index, item in enumerate(values)
    )


def _require_integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Return a tuple containing only integers."""
    values = _require_list(value, name)
    return tuple(
        _require_integer(item, f"{name}[{index}]")
        for index, item in enumerate(values)
    )


def _validate_source_id(value: Any, name: str) -> str:
    """Validate one established lowercase MD5-shaped source identifier."""
    identifier = _require_string(value, name)
    if _LOWER_HEX_ID.fullmatch(identifier) is None:
        raise ValueError(f"{name} must contain 32 lowercase hexadecimal characters")
    return identifier


def load_audit_index(path: Path) -> AuditIndex:
    """Load the metadata-only audit sidecar without reading image content."""
    if not path.is_file():
        raise FileNotFoundError(f"Audit sidecar not found: {path}")
    try:
        frame = pd.read_parquet(
            path,
            columns=list(AUDIT_COLUMNS),
            dtype_backend="pyarrow",
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid audit sidecar schema: {path}") from error
    rows = frame.sort_values(
        "sequence_id", kind="mergesort"
    ).to_dict(orient="records")

    entries: dict[str, AuditEntry] = {}
    for row_number, value in enumerate(rows, start=1):
        prefix = f"audit row {row_number}"
        sequence_id = _validate_source_id(
            value["sequence_id"], f"{prefix}.sequence_id"
        )
        image_sequence_id = _validate_source_id(
            value["image_sequence_id"], f"{prefix}.image_sequence_id"
        )
        annotation_source = _require_string(
            value["tier"], f"{prefix}.tier"
        )
        if annotation_source not in ANNOTATION_SOURCES:
            raise ValueError(f"Unsupported audit tier: {annotation_source}")
        timepoint_count = _require_integer(
            value["time_points"], f"{prefix}.time_points"
        )
        validate_timepoint_count(timepoint_count)
        patient_id = _require_string(
            value["patient_id"], f"{prefix}.patient_id"
        )
        target_finding = _require_string(
            value["finding_token"], f"{prefix}.finding_token"
        )
        if target_finding not in FINDING_DISPLAY_NAMES:
            raise ValueError(f"Unsupported audit finding: {target_finding}")

        study_ids = _require_string_tuple(
            value["study_ids"], f"{prefix}.study_ids"
        )
        image_paths = _require_string_tuple(
            value["image_paths"], f"{prefix}.image_paths"
        )
        order_mins = _require_integer_tuple(
            value["study_order_min"], f"{prefix}.study_order_min"
        )
        order_maxs = _require_integer_tuple(
            value["study_order_max"], f"{prefix}.study_order_max"
        )
        expected_lengths = {
            len(study_ids),
            len(image_paths),
            len(order_mins),
            len(order_maxs),
        }
        if expected_lengths != {timepoint_count}:
            raise ValueError(f"{prefix} visit metadata has inconsistent lengths")

        raw_transition_labels = value["transition_labels"]
        transition_labels = (
            None
            if raw_transition_labels is None
            else _require_string_tuple(
                raw_transition_labels, f"{prefix}.transition_labels"
            )
        )
        if transition_labels is not None and (
            len(transition_labels) != timepoint_count - 1
            or any(label not in SOURCE_LABELS for label in transition_labels)
        ):
            raise ValueError(f"{prefix} has invalid transition labels")

        entry = AuditEntry(
            sequence_id=sequence_id,
            image_sequence_id=image_sequence_id,
            annotation_source=annotation_source,
            timepoint_count=timepoint_count,
            patient_id=patient_id,
            target_finding=target_finding,
            study_ids=study_ids,
            image_paths=image_paths,
            order_mins=order_mins,
            order_maxs=order_maxs,
            transition_labels=transition_labels,
            is_unambiguous=_require_boolean(
                value["is_unambiguous"], f"{prefix}.is_unambiguous"
            ),
            all_images_match_csv=_require_boolean(
                value["all_images_match_csv"],
                f"{prefix}.all_images_match_csv",
            ),
        )
        if sequence_id in entries:
            raise ValueError(f"Duplicate audit sequence_id: {sequence_id}")
        entries[sequence_id] = entry
    return AuditIndex(entries=entries, record_count=len(frame))


def _parse_timepoints(
    value: Any,
    *,
    timepoint_count: int,
    patient_id: str,
    record_name: str,
) -> tuple[tuple[SourceTimepoint, ...], set[str]]:
    """Parse visits and return any non-fatal structural exclusion reasons."""
    raw_points = _require_list(value, f"{record_name}.timepoints")
    if len(raw_points) != timepoint_count:
        raise ValueError(
            f"{record_name}.timepoints must contain {timepoint_count} entries"
        )

    points: list[SourceTimepoint] = []
    reasons: set[str] = set()
    for index, raw_point in enumerate(raw_points, start=1):
        point_name = f"{record_name}.timepoints[{index - 1}]"
        point = _require_object(raw_point, point_name)
        require_exact_keys(point, SOURCE_TIMEPOINT_KEYS, point_name)
        timepoint = _require_string(point["timepoint"], f"{point_name}.timepoint")
        study_id = _require_string(point["study_id"], f"{point_name}.study_id")
        image_path = _require_string(point["image_path"], f"{point_name}.image_path")
        order_min = _require_integer(
            point["patient_report_date_order_min"],
            f"{point_name}.patient_report_date_order_min",
        )
        order_max = _require_integer(
            point["patient_report_date_order_max"],
            f"{point_name}.patient_report_date_order_max",
        )

        if "\\" in image_path:
            raise ValueError(f"{point_name}.image_path must use POSIX separators")
        relative_path = PurePosixPath(image_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe source image path: {image_path}")
        if relative_path.suffix != ".jpg":
            raise ValueError(f"Source image path must end in exact .jpg: {image_path}")
        if len(relative_path.parts) != 4 or relative_path.parts[0] != "train":
            raise ValueError(f"Unsupported source image path structure: {image_path}")
        path_patient, path_study, filename = relative_path.parts[1:]
        if (
            _PATIENT_ID.fullmatch(path_patient) is None
            or _STUDY_ID.fullmatch(path_study) is None
            or path_patient != patient_id
            or path_study != study_id
        ):
            raise ValueError(f"Source image metadata conflicts with path: {image_path}")
        if _FRONTAL_FILENAME.fullmatch(filename) is None:
            reasons.add("non_frontal_images")
        if timepoint != f"T{index}" or order_min > order_max:
            reasons.add("invalid_chronology")

        points.append(
            SourceTimepoint(
                timepoint=timepoint,
                study_id=study_id,
                image_path=image_path,
                order_min=order_min,
                order_max=order_max,
            )
        )

    if len({point.image_path for point in points}) != timepoint_count:
        reasons.add("repeated_images")
    if any(
        previous.order_max >= current.order_min
        for previous, current in zip(points, points[1:], strict=False)
    ):
        reasons.add("invalid_chronology")
    return tuple(points), reasons


def _parse_source_labels(
    value: Any, *, timepoint_count: int, record_name: str
) -> tuple[str, ...]:
    """Parse one exact-length vector from the source-label vocabulary."""
    labels = _require_string_tuple(value, f"{record_name}.change_labels")
    if len(labels) != timepoint_count - 1:
        raise ValueError(
            f"{record_name}.change_labels must contain {timepoint_count - 1} entries"
        )
    unsupported = [label for label in labels if label not in SOURCE_LABELS]
    if unsupported:
        raise ValueError(f"Unsupported source labels: {unsupported}")
    return labels


def source_labels_are_state_consistent(labels: tuple[str, ...]) -> bool:
    """Return whether consecutive source labels agree on their shared visit."""
    return all(
        _SOURCE_LABEL_STATES[previous][1] == _SOURCE_LABEL_STATES[current][0]
        for previous, current in zip(labels, labels[1:], strict=False)
    )


def _require_source_id_integrity(candidate: SourceCandidate) -> None:
    """Recompute both established source IDs and reject conflicting content."""
    image_paths = [point.image_path for point in candidate.timepoints]
    sequence_values = [
        candidate.annotation_source,
        candidate.patient_id,
        candidate.target_finding,
        *image_paths,
    ]
    image_values = [
        candidate.annotation_source,
        candidate.patient_id,
        *image_paths,
    ]
    expected_sequence_id = hashlib.md5(
        "|".join(sequence_values).encode("utf-8")
    ).hexdigest()
    expected_image_sequence_id = hashlib.md5(
        "|".join(image_values).encode("utf-8")
    ).hexdigest()
    if candidate.source_sequence_id != expected_sequence_id:
        raise ValueError(
            "Source sequence_id conflicts with its identity fields: "
            f"{candidate.source_sequence_id}"
        )
    if candidate.image_sequence_id != expected_image_sequence_id:
        raise ValueError(
            "Source image_sequence_id conflicts with its identity fields: "
            f"{candidate.source_sequence_id}"
        )


def _require_audit_match(
    candidate: SourceCandidate,
    audit_entry: AuditEntry,
) -> None:
    """Reject any source content that conflicts with its audit-sidecar row."""
    expected = (
        candidate.image_sequence_id,
        candidate.annotation_source,
        candidate.timepoint_count,
        candidate.patient_id,
        candidate.target_finding,
        tuple(point.study_id for point in candidate.timepoints),
        tuple(point.image_path for point in candidate.timepoints),
        tuple(point.order_min for point in candidate.timepoints),
        tuple(point.order_max for point in candidate.timepoints),
        candidate.source_interval_labels,
    )
    actual = (
        audit_entry.image_sequence_id,
        audit_entry.annotation_source,
        audit_entry.timepoint_count,
        audit_entry.patient_id,
        audit_entry.target_finding,
        audit_entry.study_ids,
        audit_entry.image_paths,
        audit_entry.order_mins,
        audit_entry.order_maxs,
        audit_entry.transition_labels,
    )
    if actual != expected:
        raise ValueError(
            "Source JSON conflicts with audit sidecar for sequence_id: "
            f"{candidate.source_sequence_id}"
        )
    if not audit_entry.is_unambiguous:
        raise ValueError(
            "Source JSON contains an audit-ambiguous sequence_id: "
            f"{candidate.source_sequence_id}"
        )


def _check_interval_references(
    references: dict[tuple[str, str, str, str], str],
    candidate: SourceCandidate,
) -> None:
    """Reject conflicting labels for one source-specific image interval."""
    for previous, current, label in zip(
        candidate.timepoints[:-1],
        candidate.timepoints[1:],
        candidate.source_interval_labels,
        strict=True,
    ):
        key = (
            candidate.annotation_source,
            candidate.target_finding,
            previous.image_path,
            current.image_path,
        )
        existing = references.setdefault(key, label)
        if existing != label:
            raise ValueError(
                "Conflicting source interval reference for "
                f"{candidate.source_sequence_id}"
            )


def load_source_candidates(
    path: Path,
    *,
    timepoint_count: int,
    audit_index: AuditIndex,
) -> SourceLoadResult:
    """Load one source JSON and apply the frozen fatal and quality checks."""
    validate_timepoint_count(timepoint_count)
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"Source artifact must be a JSON array: {path}")

    candidates: list[SourceCandidate] = []
    seen_ids: set[str] = set()
    interval_references: dict[tuple[str, str, str, str], str] = {}
    exclusion_counts: Counter[str] = Counter(
        {reason: 0 for reason in QUALITY_EXCLUSION_REASONS}
    )
    excluded_records = 0
    counts_by_source = {
        source: {
            "source_record_count": 0,
            "eligible_record_count": 0,
            "excluded_record_count": 0,
            "quality_exclusion_counts": {
                reason: 0 for reason in QUALITY_EXCLUSION_REASONS
            },
        }
        for source in ANNOTATION_SOURCES
    }
    for record_index, raw_record in enumerate(value):
        record_name = f"{path.name}[{record_index}]"
        record = _require_object(raw_record, record_name)
        require_exact_keys(record, SOURCE_RECORD_KEYS, record_name)
        sequence_id = _validate_source_id(
            record["sequence_id"], f"{record_name}.sequence_id"
        )
        if sequence_id in seen_ids:
            raise ValueError(f"Duplicate source sequence_id: {sequence_id}")
        seen_ids.add(sequence_id)
        image_sequence_id = _validate_source_id(
            record["image_sequence_id"], f"{record_name}.image_sequence_id"
        )
        patient_id = _require_string(
            record["patient_id"], f"{record_name}.patient_id"
        )
        annotation_source = _require_string(
            record["tier"], f"{record_name}.tier"
        )
        if annotation_source not in ANNOTATION_SOURCES:
            raise ValueError(f"Unsupported annotation source: {annotation_source}")
        target_finding = _require_string(
            record["finding"], f"{record_name}.finding"
        )
        if target_finding not in FINDING_DISPLAY_NAMES:
            raise ValueError(f"Unsupported target finding: {target_finding}")
        points, reasons = _parse_timepoints(
            record["timepoints"],
            timepoint_count=timepoint_count,
            patient_id=patient_id,
            record_name=record_name,
        )
        labels = _parse_source_labels(
            record["change_labels"],
            timepoint_count=timepoint_count,
            record_name=record_name,
        )
        candidate = SourceCandidate(
            source_sequence_id=sequence_id,
            image_sequence_id=image_sequence_id,
            patient_id=patient_id,
            annotation_source=annotation_source,
            timepoint_count=timepoint_count,
            target_finding=target_finding,
            timepoints=points,
            source_interval_labels=labels,
        )
        _require_source_id_integrity(candidate)
        audit_entry = audit_index.entries.get(sequence_id)
        if audit_entry is None:
            raise ValueError(f"Source sequence_id is absent from audit: {sequence_id}")
        _require_audit_match(candidate, audit_entry)
        _check_interval_references(interval_references, candidate)

        source_counts = counts_by_source[annotation_source]
        source_counts["source_record_count"] += 1
        if not audit_entry.all_images_match_csv:
            reasons.add("source_image_not_listed")
        if not source_labels_are_state_consistent(labels):
            reasons.add("state_inconsistent")
        if reasons:
            excluded_records += 1
            exclusion_counts.update(reasons)
            source_counts["excluded_record_count"] += 1
            for reason in reasons:
                source_counts["quality_exclusion_counts"][reason] += 1
            continue
        candidates.append(candidate)
        source_counts["eligible_record_count"] += 1

    candidates.sort(key=lambda candidate: candidate.source_sequence_id)
    return SourceLoadResult(
        candidates=tuple(candidates),
        source_record_count=len(value),
        excluded_record_count=excluded_records,
        quality_exclusion_counts=dict(sorted(exclusion_counts.items())),
        interval_references=dict(sorted(interval_references.items())),
        counts_by_annotation_source=counts_by_source,
    )


def build_benchmark_item(
    candidate: SourceCandidate,
    *,
    benchmark_version: str,
) -> dict[str, Any]:
    """Convert one internal source candidate to the frozen public item schema."""
    if not isinstance(benchmark_version, str) or not benchmark_version:
        raise ValueError("benchmark_version must be a non-empty string")
    reference_labels = tuple(
        SOURCE_TO_CHANGE_LABEL[label]
        for label in candidate.source_interval_labels
    )
    tasks: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        answer = derive_answer(task, reference_labels)
        if task == "most_severe_visit_or_plateau" and answer == NO_IDENTIFIABLE:
            tasks[task] = {
                "eligible": False,
                "reference_answer": None,
                "exclusion_reason": MOST_SEVERE_EXCLUSION,
            }
        else:
            tasks[task] = {
                "eligible": True,
                "reference_answer": answer,
                "exclusion_reason": None,
            }

    return {
        "benchmark_version": benchmark_version,
        "trajectory_id": candidate.trajectory_id,
        "dataset": DATASET,
        "annotation_source": candidate.annotation_source,
        "source_sequence_id": candidate.source_sequence_id,
        "patient_id": candidate.patient_id,
        "timepoint_count": candidate.timepoint_count,
        "target_finding": {
            "canonical_name": candidate.target_finding,
            "display_name": FINDING_DISPLAY_NAMES[candidate.target_finding],
        },
        "images": [
            {
                "timepoint": point.timepoint,
                "image_id": point.image_path,
                "relative_path": str(
                    PurePosixPath(point.image_path).with_suffix(".png")
                ),
            }
            for point in candidate.timepoints
        ],
        "source_interval_labels": list(candidate.source_interval_labels),
        "reference_change_labels": list(reference_labels),
        "tasks": tasks,
    }

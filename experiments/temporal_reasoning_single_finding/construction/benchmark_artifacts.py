"""Shared validation, provenance, and manifests for benchmark constructors."""

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..artifact_io import assert_no_image_payload, repository_root, sha256_file
from ..protocol.contracts import (
    ANNOTATION_SOURCES,
    DATASET,
    FINAL_PROMPT_VERSION,
    FINDING_DISPLAY_NAMES,
    IMAGE_PATH_CONTRACT,
    JSON_SCHEMA_VERSION,
    MOST_SEVERE_EXCLUSION,
    NO_IDENTIFIABLE,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
    SOURCE_LABELS,
    SOURCE_TO_CHANGE_LABEL,
    TASKS,
    require_exact_keys,
    validate_change_labels,
)
from ..protocol.task_rules import derive_answer
from ..protocol.task_selection import (
    assigned_task, selection_summary, validate_task_selection,
)
from .source_data import SourceLoadResult


ROOT = repository_root()
DEFAULT_THREE_TP_SOURCE = (
    ROOT / "artifacts/chexpert_strict_order_finding_sequences_3tp.json"
)
DEFAULT_FOUR_TP_SOURCE = (
    ROOT / "artifacts/chexpert_strict_order_finding_sequences_4tp.json"
)
DEFAULT_AUDIT_SIDECAR = (
    ROOT / "artifacts/chexpert_strict_order_finding_sequences.parquet"
)
DEFAULT_BENCHMARK_ROOT = (
    ROOT / "artifacts/temporal_reasoning_single_finding/benchmarks"
)

BENCHMARK_ITEM_KEYS = {
    "benchmark_version",
    "trajectory_id",
    "dataset",
    "annotation_source",
    "source_sequence_id",
    "patient_id",
    "timepoint_count",
    "target_finding",
    "images",
    "source_interval_labels",
    "reference_change_labels",
    "tasks",
}
TARGET_FINDING_KEYS = {"canonical_name", "display_name"}
IMAGE_KEYS = {"timepoint", "image_id", "relative_path"}
TASK_ENTRY_KEYS = {"eligible", "reference_answer", "exclusion_reason"}
MANIFEST_KEYS = {
    "benchmark_version",
    "pair_prompt_version",
    "seq_prompt_version",
    "final_prompt_version",
    "json_schema_version",
    "dataset",
    "image_path_contract",
    "tasks",
    "files",
    "source_files",
    "construction",
    "counts",
    "reference_check",
    "validation",
}


def _stable_source_path(path: Path) -> str:
    """Return a deterministic repository-relative path or fixture filename."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _require_nonempty_string(value: Any, name: str) -> str:
    """Return a non-empty string or raise a benchmark schema error."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_benchmark_item(
    item: dict[str, Any],
    *,
    benchmark_version: str,
) -> None:
    """Validate one constructed item against the exact frozen schema."""
    fields = BENCHMARK_ITEM_KEYS | ({"assigned_task"} if "assigned_task" in item else set())
    require_exact_keys(item, fields, "benchmark item")
    if item["benchmark_version"] != benchmark_version:
        raise ValueError("Benchmark item version does not match its manifest")
    if item["dataset"] != DATASET:
        raise ValueError("Benchmark item has an unsupported dataset")
    annotation_source = item["annotation_source"]
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError("Benchmark item has an unsupported annotation source")
    source_sequence_id = _require_nonempty_string(
        item["source_sequence_id"], "source_sequence_id"
    )
    if item["trajectory_id"] != f"{DATASET}:{source_sequence_id}":
        raise ValueError("Benchmark item trajectory_id is not source-derived")
    _require_nonempty_string(item["patient_id"], "patient_id")
    timepoint_count = item["timepoint_count"]
    if type(timepoint_count) is not int or timepoint_count not in (3, 4):
        raise ValueError("Benchmark item has an invalid timepoint_count")

    target_finding = item["target_finding"]
    if not isinstance(target_finding, dict):
        raise ValueError("target_finding must be an object")
    require_exact_keys(target_finding, TARGET_FINDING_KEYS, "target_finding")
    canonical_name = target_finding["canonical_name"]
    if canonical_name not in FINDING_DISPLAY_NAMES:
        raise ValueError("Benchmark item has an unsupported target finding")
    if target_finding["display_name"] != FINDING_DISPLAY_NAMES[canonical_name]:
        raise ValueError("Target finding display name is not canonical")

    images = item["images"]
    if not isinstance(images, list) or len(images) != timepoint_count:
        raise ValueError("Benchmark item has an invalid images array")
    image_ids: list[str] = []
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            raise ValueError("Benchmark image entry must be an object")
        require_exact_keys(image, IMAGE_KEYS, "benchmark image")
        if image["timepoint"] != f"T{index}":
            raise ValueError("Benchmark image timepoints are not chronological")
        image_id = _require_nonempty_string(image["image_id"], "image_id")
        expected_relative = str(PurePosixPath(image_id).with_suffix(".png"))
        if image["relative_path"] != expected_relative:
            raise ValueError("Benchmark image PNG path is not suffix-derived")
        image_ids.append(image_id)
    if len(set(image_ids)) != timepoint_count:
        raise ValueError("Benchmark item repeats an image")

    source_labels = item["source_interval_labels"]
    if not isinstance(source_labels, list) or (
        len(source_labels) != timepoint_count - 1
        or any(label not in SOURCE_LABELS for label in source_labels)
    ):
        raise ValueError("Benchmark item has invalid source interval labels")
    reference_labels = validate_change_labels(
        item["reference_change_labels"], timepoint_count
    )
    if reference_labels != tuple(
        SOURCE_TO_CHANGE_LABEL[label] for label in source_labels
    ):
        raise ValueError("Benchmark item reference labels conflict with source labels")

    tasks = item["tasks"]
    if not isinstance(tasks, dict):
        raise ValueError("Benchmark item tasks must be an object")
    require_exact_keys(tasks, set(TASKS), "benchmark tasks")
    for task in TASKS:
        entry = tasks[task]
        if not isinstance(entry, dict):
            raise ValueError(f"Task entry must be an object: {task}")
        require_exact_keys(entry, TASK_ENTRY_KEYS, f"task {task}")
        if type(entry["eligible"]) is not bool:
            raise ValueError(f"Task eligibility must be boolean: {task}")
        expected_answer = derive_answer(task, reference_labels)
        if entry["eligible"]:
            if (
                entry["reference_answer"] != expected_answer
                or entry["exclusion_reason"] is not None
                or expected_answer == NO_IDENTIFIABLE
            ):
                raise ValueError(f"Eligible task reference is invalid: {task}")
        elif (
            task != "most_severe_visit_or_plateau"
            or expected_answer != NO_IDENTIFIABLE
            or entry["reference_answer"] is not None
            or entry["exclusion_reason"] != MOST_SEVERE_EXCLUSION
        ):
            raise ValueError(f"Ineligible task entry is invalid: {task}")
    assigned_task(item)
    assert_no_image_payload(item)


def _reference_check(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce every stored task answer and return the frozen summary."""
    checked = 0
    eligible = 0
    ineligible_most_severe = 0
    mismatches = 0
    for item in items:
        reference_labels = tuple(item["reference_change_labels"])
        for task in TASKS:
            checked += 1
            entry = item["tasks"][task]
            derived = derive_answer(task, reference_labels)
            if entry["eligible"]:
                eligible += 1
                if entry["reference_answer"] != derived:
                    mismatches += 1
            else:
                ineligible_most_severe += 1
                if (
                    task != "most_severe_visit_or_plateau"
                    or derived != NO_IDENTIFIABLE
                    or entry["reference_answer"] is not None
                    or entry["exclusion_reason"] != MOST_SEVERE_EXCLUSION
                ):
                    mismatches += 1
    if mismatches:
        raise ValueError(f"Reference Check found {mismatches} mismatches")
    return {
        "status": "passed",
        "checked_task_entries": checked,
        "eligible_final_cases": eligible,
        "ineligible_most_severe_task_entries": ineligible_most_severe,
        "mismatches": mismatches,
    }


def _annotation_source_counts(
    items: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Count items while retaining explicit zeroes for both source tiers."""
    counts = Counter(item["annotation_source"] for item in items)
    return {source: counts[source] for source in ANNOTATION_SOURCES}


def _file_entry(
    path: Path,
    items: Sequence[dict[str, Any]],
    *,
    timepoint_count: int,
) -> dict[str, Any]:
    """Describe one completed deterministic benchmark JSON file."""
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "item_count": len(items),
        "timepoint_count": timepoint_count,
        "annotation_source_counts": _annotation_source_counts(items),
    }


def _patient_count_summaries(
    items: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return distinct-patient and maximum-per-patient summaries."""
    overall = Counter((item["dataset"], item["patient_id"]) for item in items)
    by_source: dict[str, Counter[tuple[str, str]]] = {
        source: Counter() for source in ANNOTATION_SOURCES
    }
    by_group: dict[str, dict[str, Counter[tuple[str, str]]]] = {
        f"{timepoint_count}tp": {
            source: Counter() for source in ANNOTATION_SOURCES
        }
        for timepoint_count in (3, 4)
    }
    for item in items:
        patient = (item["dataset"], item["patient_id"])
        source = item["annotation_source"]
        by_source[source][patient] += 1
        by_group[f"{item['timepoint_count']}tp"][source][patient] += 1

    distinct = {
        "overall": len(overall),
        "by_annotation_source": {
            source: len(by_source[source]) for source in ANNOTATION_SOURCES
        },
        "by_timepoint_and_annotation_source": {
            group: {
                source: len(by_group[group][source])
                for source in ANNOTATION_SOURCES
            }
            for group in ("3tp", "4tp")
        },
    }
    maximum = {
        "overall": max(overall.values(), default=0),
        "by_annotation_source": {
            source: max(by_source[source].values(), default=0)
            for source in ANNOTATION_SOURCES
        },
        "by_timepoint_and_annotation_source": {
            group: {
                source: max(by_group[group][source].values(), default=0)
                for source in ANNOTATION_SOURCES
            }
            for group in ("3tp", "4tp")
        },
    }
    return distinct, maximum


def _cross_length_overlap(
    items_by_timepoint: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Count shared image inputs and ordered PAIR inputs across 3TP and 4TP."""
    result: dict[str, dict[str, int]] = {}
    for source in ANNOTATION_SOURCES:
        by_timepoint = {
            timepoint_count: [
                item
                for item in items_by_timepoint[timepoint_count]
                if item["annotation_source"] == source
            ]
            for timepoint_count in (3, 4)
        }
        image_sets = {
            timepoint_count: {
                image["image_id"]
                for item in by_timepoint[timepoint_count]
                for image in item["images"]
            }
            for timepoint_count in (3, 4)
        }
        pair_sets = {
            timepoint_count: {
                (earlier["image_id"], later["image_id"])
                for item in by_timepoint[timepoint_count]
                for earlier, later in zip(
                    item["images"], item["images"][1:], strict=False
                )
            }
            for timepoint_count in (3, 4)
        }
        shared_images = image_sets[3] & image_sets[4]
        result[source] = {
            "shared_image_ids": len(shared_images),
            "shared_ordered_adjacent_image_pairs": len(
                pair_sets[3] & pair_sets[4]
            ),
            "three_tp_trajectories_with_shared_images": sum(
                bool(
                    {image["image_id"] for image in item["images"]}
                    & shared_images
                )
                for item in by_timepoint[3]
            ),
            "four_tp_trajectories_with_shared_images": sum(
                bool(
                    {image["image_id"] for image in item["images"]}
                    & shared_images
                )
                for item in by_timepoint[4]
            ),
        }
    return result


def _item_counts(
    items_by_timepoint: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Build every population count required by the frozen manifest."""
    items = [
        item
        for timepoint_count in (3, 4)
        for item in items_by_timepoint[timepoint_count]
    ]
    distinct, maximum = _patient_count_summaries(items)
    source_counts = Counter(item["annotation_source"] for item in items)
    finding_counts = Counter(
        item["target_finding"]["canonical_name"] for item in items
    )
    eligible_by_task = {
        task: sum(item["tasks"][task]["eligible"] for item in items)
        for task in TASKS
    }
    return {
        "total_items": len(items),
        "by_dataset": {DATASET: len(items)},
        "by_annotation_source": {
            source: source_counts[source] for source in ANNOTATION_SOURCES
        },
        "by_timepoint_count": {
            str(timepoint_count): len(items_by_timepoint[timepoint_count])
            for timepoint_count in (3, 4)
        },
        "by_target_finding": dict(sorted(finding_counts.items())),
        "eligible_by_task": eligible_by_task,
        "distinct_patients": distinct,
        "maximum_selected_trajectories_per_patient": maximum,
        "cross_length_input_overlap_by_annotation_source": (
            _cross_length_overlap(items_by_timepoint)
        ),
    }


def source_files(
    *,
    three_tp_source: Path,
    four_tp_source: Path,
    audit_sidecar: Path,
    audit_record_count: int,
    source_results: Mapping[int, SourceLoadResult],
) -> dict[str, Any]:
    """Describe the content sources and the audit-only reconciliation sidecar."""
    result: dict[str, Any] = {}
    for timepoint_count, path in (
        (3, three_tp_source),
        (4, four_tp_source),
    ):
        source_result = source_results[timepoint_count]
        result[f"{timepoint_count}tp"] = {
            "path": _stable_source_path(path),
            "sha256": sha256_file(path),
            "item_count": source_result.source_record_count,
            "eligible_item_count": len(source_result.candidates),
            "excluded_item_count": source_result.excluded_record_count,
            "quality_exclusion_counts": (
                source_result.quality_exclusion_counts
            ),
        }
    result["audit_sidecar"] = {
        "path": _stable_source_path(audit_sidecar),
        "sha256": sha256_file(audit_sidecar),
        "record_count": audit_record_count,
        "role": "read_only_validation_sidecar",
    }
    return result


def validate_cross_source_references(
    source_results: Mapping[int, SourceLoadResult],
) -> None:
    """Reject reference conflicts repeated across the 3TP and 4TP sources."""
    combined: dict[tuple[str, str, str, str], str] = {}
    for timepoint_count in (3, 4):
        for key, label in source_results[timepoint_count].interval_references.items():
            existing = combined.setdefault(key, label)
            if existing != label:
                raise ValueError(
                    "Conflicting interval reference across source artifacts: "
                    f"{key}"
                )


def build_manifest(
    *,
    benchmark_version: str,
    items_by_timepoint: Mapping[int, Sequence[dict[str, Any]]],
    file_paths: Mapping[int, Path],
    source_files: dict[str, Any],
    construction: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate one deterministic benchmark manifest."""
    all_items: list[dict[str, Any]] = []
    for timepoint_count in (3, 4):
        items = items_by_timepoint[timepoint_count]
        if [item["trajectory_id"] for item in items] != sorted(
            item["trajectory_id"] for item in items
        ):
            raise ValueError("Benchmark items must be sorted by trajectory_id")
        for item in items:
            _validate_benchmark_item(
                item,
                benchmark_version=benchmark_version,
            )
        if len({item["trajectory_id"] for item in items}) != len(items):
            raise ValueError("Benchmark file contains duplicate trajectory IDs")
        all_items.extend(items)
    if len({item["trajectory_id"] for item in all_items}) != len(all_items):
        raise ValueError("Benchmark files contain a cross-length trajectory ID")

    manifest = {
        "benchmark_version": benchmark_version,
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "seq_prompt_version": SEQ_PROMPT_VERSION,
        "final_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "dataset": DATASET,
        "image_path_contract": IMAGE_PATH_CONTRACT,
        "tasks": list(TASKS),
        "files": {
            f"{timepoint_count}tp": _file_entry(
                file_paths[timepoint_count],
                items_by_timepoint[timepoint_count],
                timepoint_count=timepoint_count,
            )
            for timepoint_count in (3, 4)
        },
        "source_files": source_files,
        "construction": construction,
        "counts": _item_counts(items_by_timepoint),
        "reference_check": _reference_check(all_items),
        "validation": {
            "status": "passed",
            "benchmark_item_schema": "passed",
            "output_checksums": "passed",
            "reference_check": "passed",
            "source_audit_reconciliation": "passed",
            "source_schema": "passed",
        },
    }
    if any("assigned_task" in item for item in all_items):
        manifest["task_selection"] = selection_summary(all_items)
    validate_task_selection(manifest, all_items)
    fields = MANIFEST_KEYS | ({"task_selection"} if "task_selection" in manifest else set())
    require_exact_keys(manifest, fields, "benchmark manifest")
    assert_no_image_payload(manifest)
    return manifest

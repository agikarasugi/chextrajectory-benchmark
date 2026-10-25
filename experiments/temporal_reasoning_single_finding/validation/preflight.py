"""Read-only benchmark and model-run preflight validation."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..artifact_io import (
    assert_no_image_payload,
    read_json,
    repository_root,
    safe_relative_path,
    sha256_file,
)
from ..inference.experiment_config import experiment_resume_identity
from ..protocol.prompt_compatibility import normalize_prompt_fixture
from ..protocol.contracts import (
    ANNOTATION_SOURCES,
    DATASET,
    FINAL_PROMPT_VERSION,
    FINDING_DISPLAY_NAMES,
    IMAGE_PATH_CONTRACT,
    JSON_SCHEMA_VERSION,
    LC_INFERENCE_POLICY,
    PAIR_PROMPT_VERSION,
    PILOT_BENCHMARK_VERSION,
    SEQ_PROMPT_VERSION,
    SETTINGS,
    SMOKE_BENCHMARK_VERSION,
    SOURCE_LABELS,
    SOURCE_TO_CHANGE_LABEL,
    TASKS,
    answer_choices,
    require_exact_keys,
    validate_change_labels,
    validate_timepoint_count,
)
from ..protocol.prompts import (
    render_lc_messages,
    render_pair_messages,
    render_seq_dir_messages,
    render_seq_messages,
)
from ..protocol.task_selection import (
    GOLD_PILOT_BENCHMARK_VERSION,
    SMOKE_V2_BENCHMARK_VERSION,
    assigned_task,
    unreduced_call_ceiling,
    validate_task_selection,
)
from .reference_check import (
    run_reference_check,
    validate_reference_check_summary,
)


MANIFEST_FIELDS = {
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
FILE_FIELDS = {
    "path",
    "sha256",
    "item_count",
    "timepoint_count",
    "annotation_source_counts",
}
ITEM_FIELDS = {
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
IMAGE_FIELDS = {"timepoint", "image_id", "relative_path"}
TARGET_FINDING_FIELDS = {"canonical_name", "display_name"}
COUNT_FIELDS = {
    "total_items",
    "by_dataset",
    "by_annotation_source",
    "by_timepoint_count",
    "by_target_finding",
    "eligible_by_task",
    "distinct_patients",
    "maximum_selected_trajectories_per_patient",
    "cross_length_input_overlap_by_annotation_source",
}
VALIDATION_FIELDS = {
    "status",
    "benchmark_item_schema",
    "output_checksums",
    "reference_check",
    "source_audit_reconciliation",
    "source_schema",
}
CALL_PLAN_FIELDS = {
    "pair_predictions",
    "seq_predictions",
    "seq_dir",
    "ref_lc",
    "pair_lc_potential",
    "seq_lc_potential",
    "potential_logical_calls_before_retries",
    "unreduced_maximum_calls_before_retries",
    "maximum_transport_attempts",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_STATES = {
    "New": ("absent", "present"),
    "Worse": ("present", "present"),
    "Stable": ("present", "present"),
    "Improved": ("present", "present"),
    "Resolved": ("present", "absent"),
}
_TASK_TITLES = {
    "trajectory_shape": "Trajectory Shape",
    "turning_regions": "Turning Regions",
    "most_severe_visit_or_plateau": "Most Severe Visit or Plateau",
    "first_improvement_between_visits": "First Improvement Between Visits",
}


@dataclass(frozen=True)
class ValidatedBenchmark:
    """A benchmark whose manifest, files, items, and references all agree."""

    manifest_path: Path
    manifest: dict[str, Any]
    items_by_timepoint: dict[int, list[dict[str, Any]]]
    items: list[dict[str, Any]]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a string-keyed mapping or reject the value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _list(value: Any, name: str) -> list[Any]:
    """Return a JSON-style list or reject the value."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _non_empty_string(value: Any, name: str) -> str:
    """Return a non-empty string or reject the value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    """Return a non-negative integer while rejecting Booleans."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_count_tree(actual: Any, expected: Any, name: str) -> None:
    """Compare a nested count declaration with strict integer types."""
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(
                f"{name} count differs: expected={expected}, actual={actual}"
            )
        return
    if not isinstance(expected, Mapping):
        if actual != expected:
            raise ValueError(f"{name} differs")
        return
    value = _mapping(actual, name)
    require_exact_keys(value, set(expected), name)
    for key, expected_child in expected.items():
        _require_count_tree(value[key], expected_child, f"{name}.{key}")


def _sha256(value: Any, name: str) -> str:
    """Validate one lowercase SHA-256 hexadecimal digest."""
    digest = _non_empty_string(value, name)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validate_relative_image_path(value: Any, suffix: str, name: str) -> str:
    """Validate one safe dataset-relative POSIX image identifier."""
    path_text = _non_empty_string(value, name)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} is not a safe relative path")
    if path.suffix != suffix:
        raise ValueError(f"{name} must end in exact {suffix}")
    return path_text


def _validate_item(
    raw_item: Any,
    *,
    benchmark_version: str,
    timepoint_count: int,
    index: int,
) -> dict[str, Any]:
    """Validate one complete frozen single-finding benchmark item."""
    item = _mapping(raw_item, f"{timepoint_count}TP item {index}")
    fields = ITEM_FIELDS | ({"assigned_task"} if "assigned_task" in item else set())
    require_exact_keys(item, fields, f"{timepoint_count}TP item {index}")
    prefix = f"{timepoint_count}TP item {index}"
    if item["benchmark_version"] != benchmark_version:
        raise ValueError(f"{prefix} benchmark_version conflicts with manifest")
    if item["dataset"] != DATASET:
        raise ValueError(f"{prefix} uses an unsupported dataset")
    annotation_source = item["annotation_source"]
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(f"{prefix} uses an unsupported annotation source")
    validate_timepoint_count(item["timepoint_count"])
    if item["timepoint_count"] != timepoint_count:
        raise ValueError(f"{prefix} is stored in the wrong benchmark file")

    source_sequence_id = _non_empty_string(
        item["source_sequence_id"], f"{prefix}.source_sequence_id"
    )
    trajectory_id = _non_empty_string(item["trajectory_id"], f"{prefix}.trajectory_id")
    if trajectory_id != f"{DATASET}:{source_sequence_id}":
        raise ValueError(f"{prefix} trajectory_id is not the namespaced source ID")
    _non_empty_string(item["patient_id"], f"{prefix}.patient_id")

    finding = _mapping(item["target_finding"], f"{prefix}.target_finding")
    require_exact_keys(finding, TARGET_FINDING_FIELDS, f"{prefix}.target_finding")
    canonical = _non_empty_string(
        finding["canonical_name"], f"{prefix}.target_finding.canonical_name"
    )
    if FINDING_DISPLAY_NAMES.get(canonical) != finding["display_name"]:
        raise ValueError(f"{prefix} target finding does not use the frozen mapping")

    images = _list(item["images"], f"{prefix}.images")
    if len(images) != timepoint_count:
        raise ValueError(f"{prefix} has the wrong image count")
    image_ids: list[str] = []
    for image_index, raw_image in enumerate(images, start=1):
        image = _mapping(raw_image, f"{prefix}.images[{image_index - 1}]")
        require_exact_keys(
            image,
            IMAGE_FIELDS,
            f"{prefix}.images[{image_index - 1}]",
        )
        if image["timepoint"] != f"T{image_index}":
            raise ValueError(f"{prefix} image timepoints are not chronological")
        image_id = _validate_relative_image_path(
            image["image_id"], ".jpg", f"{prefix}.images[{image_index - 1}].image_id"
        )
        relative_path = _validate_relative_image_path(
            image["relative_path"],
            ".png",
            f"{prefix}.images[{image_index - 1}].relative_path",
        )
        if relative_path != image_id[:-4] + ".png":
            raise ValueError(f"{prefix} PNG path is not the final-suffix conversion")
        image_ids.append(image_id)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{prefix} repeats an image")

    source_labels = _list(
        item["source_interval_labels"], f"{prefix}.source_interval_labels"
    )
    if len(source_labels) != timepoint_count - 1:
        raise ValueError(f"{prefix} has the wrong source-label count")
    if any(
        type(label) is not str or label not in SOURCE_LABELS
        for label in source_labels
    ):
        raise ValueError(f"{prefix} contains an unsupported source label")
    for earlier, later in zip(source_labels, source_labels[1:], strict=False):
        if _SOURCE_STATES[earlier][1] != _SOURCE_STATES[later][0]:
            raise ValueError(f"{prefix} has state-inconsistent source labels")
    raw_reference_labels = _list(
        item["reference_change_labels"], f"{prefix}.reference_change_labels"
    )
    reference_labels = validate_change_labels(raw_reference_labels, timepoint_count)
    expected_labels = tuple(SOURCE_TO_CHANGE_LABEL[label] for label in source_labels)
    if reference_labels != expected_labels:
        raise ValueError(f"{prefix} source and reference labels disagree")

    tasks = _mapping(item["tasks"], f"{prefix}.tasks")
    require_exact_keys(tasks, set(TASKS), f"{prefix}.tasks")
    assigned_task(item)
    return item


def _expected_counts(
    items_by_timepoint: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Calculate every declared benchmark count from frozen items."""
    items = [*items_by_timepoint[3], *items_by_timepoint[4]]
    tier_counts = Counter(item["annotation_source"] for item in items)
    finding_counts = Counter(
        item["target_finding"]["canonical_name"] for item in items
    )
    eligible_counts = {
        task: sum(bool(item["tasks"][task]["eligible"]) for item in items)
        for task in TASKS
    }

    def patient_keys(selected: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
        """Return unique dataset-scoped patient identities."""
        return {(item["dataset"], item["patient_id"]) for item in selected}

    def maximum_per_patient(selected: Sequence[dict[str, Any]]) -> int:
        """Return the largest trajectory count for one scoped patient."""
        counts = Counter((item["dataset"], item["patient_id"]) for item in selected)
        return max(counts.values(), default=0)

    by_tier = {
        tier: [item for item in items if item["annotation_source"] == tier]
        for tier in ANNOTATION_SOURCES
    }
    by_timepoint_and_tier = {
        f"{count}tp": {
            tier: [
                item
                for item in items_by_timepoint[count]
                if item["annotation_source"] == tier
            ]
            for tier in ANNOTATION_SOURCES
        }
        for count in (3, 4)
    }
    patient_shape = {
        "overall": len(patient_keys(items)),
        "by_annotation_source": {
            tier: len(patient_keys(by_tier[tier])) for tier in ANNOTATION_SOURCES
        },
        "by_timepoint_and_annotation_source": {
            length: {
                tier: len(patient_keys(grouped[tier]))
                for tier in ANNOTATION_SOURCES
            }
            for length, grouped in by_timepoint_and_tier.items()
        },
    }
    maximum_shape = {
        "overall": maximum_per_patient(items),
        "by_annotation_source": {
            tier: maximum_per_patient(by_tier[tier]) for tier in ANNOTATION_SOURCES
        },
        "by_timepoint_and_annotation_source": {
            length: {
                tier: maximum_per_patient(grouped[tier])
                for tier in ANNOTATION_SOURCES
            }
            for length, grouped in by_timepoint_and_tier.items()
        },
    }

    overlap: dict[str, dict[str, int]] = {}
    for tier in ANNOTATION_SOURCES:
        three = by_timepoint_and_tier["3tp"][tier]
        four = by_timepoint_and_tier["4tp"][tier]

        def ids(item: Mapping[str, Any]) -> set[str]:
            """Return image IDs from one benchmark item."""
            return {image["image_id"] for image in item["images"]}

        def pairs(item: Mapping[str, Any]) -> set[tuple[str, str]]:
            """Return ordered adjacent image-ID pairs from one item."""
            image_ids = [image["image_id"] for image in item["images"]]
            return set(zip(image_ids, image_ids[1:], strict=False))

        three_ids = set().union(*(ids(item) for item in three)) if three else set()
        four_ids = set().union(*(ids(item) for item in four)) if four else set()
        shared_ids = three_ids & four_ids
        three_pairs = set().union(*(pairs(item) for item in three)) if three else set()
        four_pairs = set().union(*(pairs(item) for item in four)) if four else set()
        overlap[tier] = {
            "shared_image_ids": len(shared_ids),
            "shared_ordered_adjacent_image_pairs": len(three_pairs & four_pairs),
            "three_tp_trajectories_with_shared_images": sum(
                bool(ids(item) & shared_ids) for item in three
            ),
            "four_tp_trajectories_with_shared_images": sum(
                bool(ids(item) & shared_ids) for item in four
            ),
        }

    return {
        "total_items": len(items),
        "by_dataset": {DATASET: len(items)},
        "by_annotation_source": {
            tier: tier_counts[tier] for tier in ANNOTATION_SOURCES
        },
        "by_timepoint_count": {
            "3": len(items_by_timepoint[3]),
            "4": len(items_by_timepoint[4]),
        },
        "by_target_finding": dict(sorted(finding_counts.items())),
        "eligible_by_task": eligible_counts,
        "distinct_patients": patient_shape,
        "maximum_selected_trajectories_per_patient": maximum_shape,
        "cross_length_input_overlap_by_annotation_source": overlap,
    }


def _validate_provenance(manifest: Mapping[str, Any]) -> None:
    """Validate source provenance and completed-validation declarations."""
    source_files = _mapping(manifest["source_files"], "manifest.source_files")
    require_exact_keys(
        source_files,
        {"3tp", "4tp", "audit_sidecar"},
        "manifest.source_files",
    )
    for name, raw_record in source_files.items():
        record = _mapping(raw_record, f"manifest.source_files.{name}")
        _non_empty_string(record.get("path"), f"manifest.source_files.{name}.path")
        _sha256(record.get("sha256"), f"manifest.source_files.{name}.sha256")
        if name == "audit_sidecar":
            if record.get("role") != "read_only_validation_sidecar":
                raise ValueError("Audit sidecar has the wrong role")
        elif "item_count" in record:
            _non_negative_integer(
                record["item_count"], f"manifest.source_files.{name}.item_count"
            )

    validation = _mapping(manifest["validation"], "manifest.validation")
    require_exact_keys(validation, VALIDATION_FIELDS, "manifest.validation")
    if any(value != "passed" for value in validation.values()):
        raise ValueError("Benchmark manifest contains a failed validation")


def _validate_known_population(
    manifest: Mapping[str, Any],
    items_by_timepoint: Mapping[int, Sequence[dict[str, Any]]],
) -> None:
    """Apply the fixed smoke and pilot population rules."""
    version = manifest["benchmark_version"]
    if version == SMOKE_V2_BENCHMARK_VERSION:
        construction = _mapping(manifest["construction"], "manifest.construction")
        expected = {
            "kind": "derived_smoke",
            "derived_from_benchmark_version": GOLD_PILOT_BENCHMARK_VERSION,
            "selection_rule": "assigned-task-patient-preferred-shuffle-v1",
            "seed": 42,
            "length_order": [3, 4],
            "task_order": list(TASKS),
            "quota_per_task_per_length": 2,
            "selected_trajectory_ids": {
                f"{count}tp": [item["trajectory_id"] for item in items_by_timepoint[count]]
                for count in (3, 4)
            },
        }
        require_exact_keys(
            construction, set(expected) | {"parent_manifest_sha256", "parent_files"},
            "manifest.construction",
        )
        for key, value in expected.items():
            if type(construction[key]) is not type(value) or construction[key] != value:
                raise ValueError(f"Smoke v2 has invalid construction.{key}")
        _sha256(construction["parent_manifest_sha256"], "parent_manifest_sha256")
        parent_files = _mapping(construction["parent_files"], "parent_files")
        require_exact_keys(parent_files, {"3tp", "4tp"}, "parent_files")
        for key, record in parent_files.items():
            record = _mapping(record, f"parent_files.{key}")
            require_exact_keys(record, FILE_FIELDS, f"parent_files.{key}")
            _sha256(record["sha256"], f"parent_files.{key}.sha256")
        return
    if version not in {PILOT_BENCHMARK_VERSION, SMOKE_BENCHMARK_VERSION}:
        return
    if version == PILOT_BENCHMARK_VERSION:
        expected = {
            "3tp": {"silver": 100, "gold": 100},
            "4tp": {"silver": 100, "gold": 100},
        }
    else:
        expected = {
            "3tp": {"silver": 1, "gold": 0},
            "4tp": {"silver": 0, "gold": 1},
        }
    actual = {
        f"{count}tp": {
            tier: sum(
                item["annotation_source"] == tier
                for item in items_by_timepoint[count]
            )
            for tier in ANNOTATION_SOURCES
        }
        for count in (3, 4)
    }
    _require_count_tree(actual, expected, f"{version} population")
    construction = _mapping(manifest["construction"], "manifest.construction")
    _require_count_tree(
        construction.get("fixed_quotas"),
        expected,
        "manifest.construction.fixed_quotas",
    )
    if version == PILOT_BENCHMARK_VERSION:
        if construction.get("kind") != "pilot" or construction.get("seed") != 42:
            raise ValueError("Pilot manifest does not record its kind and seed")
        if construction.get("sampling_method") != "patient-first-round-robin-v1":
            raise ValueError(
                "Pilot manifest does not record the approved sampling method"
            )
    else:
        if (
            construction.get("kind") != "derived_smoke"
            or construction.get("derived_from_benchmark_version")
            != PILOT_BENCHMARK_VERSION
            or construction.get("selection_rule")
            != "first-all-task-eligible-by-trajectory-id-v1"
        ):
            raise ValueError("Smoke manifest does not record its frozen derivation")
        for item in [*items_by_timepoint[3], *items_by_timepoint[4]]:
            if not all(item["tasks"][task]["eligible"] for task in TASKS):
                raise ValueError("Smoke trajectories must be eligible for every task")


def load_and_validate_benchmark(manifest_path: Path) -> ValidatedBenchmark:
    """Load and completely validate one compatible benchmark manifest."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Benchmark manifest is not a regular file: {manifest_path}")
    manifest = _mapping(read_json(manifest_path), "benchmark manifest")
    fields = MANIFEST_FIELDS | ({"task_selection"} if "task_selection" in manifest else set())
    require_exact_keys(manifest, fields, "benchmark manifest")

    benchmark_version = _non_empty_string(
        manifest["benchmark_version"], "manifest.benchmark_version"
    )
    frozen = {
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "seq_prompt_version": SEQ_PROMPT_VERSION,
        "final_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "dataset": DATASET,
        "image_path_contract": IMAGE_PATH_CONTRACT,
    }
    for field, expected in frozen.items():
        if manifest[field] != expected:
            raise ValueError(f"Benchmark manifest has unsupported {field}")
    if manifest["tasks"] != list(TASKS):
        raise ValueError(
            "Benchmark manifest task contract differs from the frozen tasks"
        )
    _validate_provenance(manifest)

    files = _mapping(manifest["files"], "manifest.files")
    require_exact_keys(files, {"3tp", "4tp"}, "manifest.files")
    items_by_timepoint: dict[int, list[dict[str, Any]]] = {}
    seen_paths: set[Path] = set()
    seen_trajectories: set[str] = set()
    for count in (3, 4):
        key = f"{count}tp"
        record = _mapping(files[key], f"manifest.files.{key}")
        require_exact_keys(record, FILE_FIELDS, f"manifest.files.{key}")
        if record["path"] != f"{count}tp.json":
            raise ValueError(f"manifest.files.{key}.path must be {count}tp.json")
        path = safe_relative_path(manifest_path.parent, record["path"])
        if path in seen_paths or not path.is_file():
            raise ValueError(
                f"Benchmark file is missing, repeated, or not regular: {path}"
            )
        seen_paths.add(path)
        expected_digest = _sha256(record["sha256"], f"manifest.files.{key}.sha256")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"Benchmark file checksum mismatch: {path.name}")
        if record["timepoint_count"] != count:
            raise ValueError(f"manifest.files.{key} has the wrong timepoint count")
        raw_items = _list(read_json(path), path.name)
        declared_count = _non_negative_integer(
            record["item_count"], f"manifest.files.{key}.item_count"
        )
        if len(raw_items) != declared_count:
            raise ValueError(f"manifest.files.{key} item count does not match its file")
        validated = [
            _validate_item(
                item,
                benchmark_version=benchmark_version,
                timepoint_count=count,
                index=index,
            )
            for index, item in enumerate(raw_items)
        ]
        identifiers = [item["trajectory_id"] for item in validated]
        if identifiers != sorted(identifiers):
            raise ValueError(f"{path.name} is not sorted by trajectory_id")
        repeated = seen_trajectories.intersection(identifiers)
        if len(set(identifiers)) != len(identifiers) or repeated:
            raise ValueError("Benchmark contains a duplicate trajectory_id")
        seen_trajectories.update(identifiers)
        declared_tiers = _mapping(
            record["annotation_source_counts"],
            f"manifest.files.{key}.annotation_source_counts",
        )
        require_exact_keys(
            declared_tiers,
            set(ANNOTATION_SOURCES),
            f"manifest.files.{key}.annotation_source_counts",
        )
        actual_tiers = {
            tier: sum(item["annotation_source"] == tier for item in validated)
            for tier in ANNOTATION_SOURCES
        }
        _require_count_tree(
            declared_tiers,
            actual_tiers,
            f"manifest.files.{key}.annotation_source_counts",
        )
        items_by_timepoint[count] = validated

    items = [*items_by_timepoint[3], *items_by_timepoint[4]]
    counts = _mapping(manifest["counts"], "manifest.counts")
    require_exact_keys(counts, COUNT_FIELDS, "manifest.counts")
    expected_counts = _expected_counts(items_by_timepoint)
    try:
        _require_count_tree(counts, expected_counts, "manifest.counts")
    except ValueError as error:
        raise ValueError(
            "Manifest aggregate counts disagree with benchmark items"
        ) from error

    current_reference = run_reference_check(items)
    stored_reference = validate_reference_check_summary(manifest["reference_check"])
    if current_reference != stored_reference:
        raise ValueError("Current Reference Check counts disagree with manifest")
    _validate_known_population(manifest, items_by_timepoint)
    validate_task_selection(manifest, items)
    return ValidatedBenchmark(
        manifest_path=manifest_path,
        manifest=manifest,
        items_by_timepoint=items_by_timepoint,
        items=items,
    )


def _logical_user(messages: list[dict[str, Any]], markers: list[str]) -> str:
    """Convert safe image references to approved review-document markers."""
    content = messages[0]["content"]
    if isinstance(content, str):
        return re.sub(
            r"(T\d to T\d: )(deterioration|unchanged|improvement)",
            r"\1[CHANGE LABEL]",
            content,
        )
    parts: list[str] = []
    marker_iterator = iter(markers)
    for part in content:
        if part["type"] == "text":
            parts.append(part["text"])
        else:
            parts.append(next(marker_iterator) + "\n")
    return "".join(parts)


def validate_prompt_snapshots() -> dict[str, str]:
    """Verify all 19 renderers against the independent approved fingerprints."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "golden_prompts"
        / "sha256.json"
    )
    expected = _mapping(read_json(fixture), "golden prompt checksums")
    rendered = {
        "PAIR change-label prompt": _logical_user(
            render_pair_messages("[FINDING]", "earlier", "later"),
            ["[EARLIER IMAGE ATTACHMENT]", "[LATER IMAGE ATTACHMENT]"],
        )
    }
    for count in (3, 4):
        image_ids = [f"image-{index}" for index in range(1, count + 1)]
        markers = [f"[T{index} IMAGE ATTACHMENT]" for index in range(1, count + 1)]
        rendered[f"SEQ change-label prompt: {count}TP"] = _logical_user(
            render_seq_messages("[FINDING]", image_ids), markers
        )
        for task in TASKS:
            title = _TASK_TITLES[task]
            rendered[f"{title}: {count}TP SEQ-DIR"] = _logical_user(
                render_seq_dir_messages("[FINDING]", image_ids, task), markers
            )
            rendered[f"{title}: {count}TP shared LC"] = _logical_user(
                render_lc_messages(
                    "[FINDING]",
                    ["deterioration"] * (count - 1),
                    task,
                ),
                [],
            )
    actual = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in rendered.items()
    }
    if len(actual) != 19 or actual != expected:
        raise ValueError("Rendered prompts do not match all 19 approved snapshots")
    return actual


def _validate_registries(
    registries: Mapping[str, Any], items: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Validate prepared dependencies and their exact potential call plan."""
    required = {"pair_inputs", "seq_inputs", "final_cases", "model_inputs", "call_plan"}
    if not required.issubset(registries):
        raise ValueError("Prepared registries are incomplete")
    pair_inputs = _list(registries["pair_inputs"], "pair_inputs")
    seq_inputs = _list(registries["seq_inputs"], "seq_inputs")
    final_cases = _list(registries["final_cases"], "final_cases")
    model_inputs = _list(registries["model_inputs"], "model_inputs")

    def unique_ids(rows: Sequence[Any], field: str, name: str) -> set[str]:
        """Return unique required identifiers from validated row mappings."""
        identifiers: list[str] = []
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, f"{name}[{index}]")
            identifiers.append(
                _non_empty_string(row.get(field), f"{name}[{index}].{field}")
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} contains duplicate {field} values")
        return set(identifiers)

    pair_ids = unique_ids(pair_inputs, "prediction_id", "pair_inputs")
    seq_ids = unique_ids(seq_inputs, "prediction_id", "seq_inputs")
    case_ids = unique_ids(final_cases, "final_case_id", "final_cases")
    del case_ids
    assignments = [assigned_task(spec) for spec in seq_inputs]
    if any(task is not None for task in assignments):
        if any(task is None for task in assignments):
            raise ValueError("Prepared SEQ inputs mix assigned and all-task modes")
        expected = {
            (spec["prediction_id"], task)
            for spec, task in zip(seq_inputs, assignments, strict=True)
        }
        actual = {(case["seq_dependency_id"], case["task"]) for case in final_cases}
        if actual != expected or len(final_cases) != len(seq_inputs):
            raise ValueError("Final cases do not match frozen task assignments")
    all_result_ids: set[str] = set()
    for index, raw_case in enumerate(final_cases):
        case = _mapping(raw_case, f"final_cases[{index}]")
        pair_dependencies = _list(
            case.get("pair_dependency_ids"),
            f"final_cases[{index}].pair_dependency_ids",
        )
        if len(pair_dependencies) != case.get("timepoint_count", 0) - 1:
            raise ValueError("Final case has the wrong number of PAIR dependencies")
        if any(identifier not in pair_ids for identifier in pair_dependencies):
            raise ValueError("Final case refers to an absent PAIR dependency")
        if case.get("seq_dependency_id") not in seq_ids:
            raise ValueError("Final case refers to an absent SEQ dependency")
        result_ids = _mapping(
            case.get("result_ids"), f"final_cases[{index}].result_ids"
        )
        require_exact_keys(
            result_ids,
            set(SETTINGS),
            f"final_cases[{index}].result_ids",
        )
        for setting, identifier in result_ids.items():
            _non_empty_string(identifier, f"final_cases[{index}].result_ids.{setting}")
            if identifier in all_result_ids:
                raise ValueError("Final cases contain a duplicate result ID")
            all_result_ids.add(identifier)
        task = case.get("task")
        reference = case.get("reference_answer")
        if task not in TASKS or reference not in answer_choices(
            task, case["timepoint_count"]
        ):
            raise ValueError("Final case has an invalid task reference")

    expected_model_records = len(pair_inputs) + len(seq_inputs) + 2 * len(final_cases)
    if len(model_inputs) != expected_model_records:
        raise ValueError("Initial model-input registry has the wrong size")
    model_keys: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(model_inputs):
        record = _mapping(raw_record, f"model_inputs[{index}]")
        record_type = _non_empty_string(
            record.get("record_type"), f"model_inputs[{index}].record_type"
        )
        record_id = _non_empty_string(
            record.get("record_id"), f"model_inputs[{index}].record_id"
        )
        key = (record_type, record_id)
        if key in model_keys:
            raise ValueError("Initial model-input registry contains a duplicate")
        model_keys.add(key)
        _non_empty_string(
            record.get("model_input_id"), f"model_inputs[{index}].model_input_id"
        )
        assert_no_image_payload(record)

    expected_plan = {
        "pair_predictions": len(pair_inputs),
        "seq_predictions": len(seq_inputs),
        "seq_dir": len(final_cases),
        "ref_lc": len(final_cases),
        "pair_lc_potential": len(final_cases),
        "seq_lc_potential": len(final_cases),
        "potential_logical_calls_before_retries": (
            len(pair_inputs) + len(seq_inputs) + 4 * len(final_cases)
        ),
        "unreduced_maximum_calls_before_retries": unreduced_call_ceiling(items),
    }
    expected_plan["maximum_transport_attempts"] = (
        expected_plan["potential_logical_calls_before_retries"] * 3
    )
    call_plan = _mapping(registries["call_plan"], "call_plan")
    require_exact_keys(call_plan, CALL_PLAN_FIELDS, "call_plan")
    if call_plan != expected_plan:
        raise ValueError("Prepared call plan does not match its registries")
    return call_plan


def _experiment_mapping(experiment: Any) -> dict[str, Any]:
    """Normalize the current mapping or dataclass experiment interface."""
    if isinstance(experiment, Mapping):
        return dict(experiment)
    fields = ("run_name", "model", "base_url", "benchmark_path")
    if all(hasattr(experiment, field) for field in fields):
        return {field: getattr(experiment, field) for field in fields}
    raise ValueError("Experiment loader returned an unsupported value")


def _expected_run_manifest(
    experiment: Mapping[str, Any], benchmark: ValidatedBenchmark,
    *, lc_inference_policy: str = LC_INFERENCE_POLICY,
) -> dict[str, Any]:
    """Reproduce the immutable run identity written by prepare_run."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "golden_prompts"
        / "sha256.json"
    )
    manifest = benchmark.manifest
    return {
        "lc_inference_policy": lc_inference_policy,
        "experiment": experiment_resume_identity(dict(experiment)),
        "benchmark_manifest_sha256": sha256_file(benchmark.manifest_path),
        "benchmark_version": manifest["benchmark_version"],
        "pair_prompt_version": manifest["pair_prompt_version"],
        "seq_prompt_version": manifest["seq_prompt_version"],
        "final_prompt_version": manifest["final_prompt_version"],
        "json_schema_version": manifest["json_schema_version"],
        "prompt_fixture_checksums": {
            "golden_prompts/sha256.json": sha256_file(fixture)
        },
    }


def _check_run_target(
    run_dir: Path,
    experiment: Mapping[str, Any],
    benchmark: ValidatedBenchmark,
    *, lc_inference_policy: str = LC_INFERENCE_POLICY,
) -> None:
    """Require an unused run name or one exactly matching resumable run."""
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise ValueError("Selected run path exists but is not a directory")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Existing run directory lacks run_manifest.json")
    saved = read_json(manifest_path)
    if isinstance(saved, dict):
        # Preparation also ignores this legacy machine-specific duplicate path.
        saved.pop("benchmark_manifest_path", None)
        saved["experiment"] = experiment_resume_identity(saved["experiment"])
    if normalize_prompt_fixture(saved) != _expected_run_manifest(
        experiment, benchmark, lc_inference_policy=lc_inference_policy,
    ):
        raise ValueError("Existing run directory does not match this experiment")


def _load_private_paths(path: Path) -> dict[str, Path]:
    """Use the inference loader when available, retaining a strict fallback."""
    try:
        from ..inference.experiment_config import load_private_paths
    except ImportError:
        raw = _mapping(read_json(path), "private paths")
        require_exact_keys(raw, {"chexpert_plus_image_root"}, "private paths")
        root = Path(_non_empty_string(raw["chexpert_plus_image_root"], "image root"))
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("CheXpert Plus image root must be an absolute directory")
        return {"chexpert_plus_image_root": root.resolve()}
    loaded = load_private_paths(path)
    return {key: Path(value) for key, value in loaded.items()}


def _check_image_files(benchmark: ValidatedBenchmark, image_root: Path) -> int:
    """Check required PNG metadata without opening or rendering image content."""
    if not image_root.is_absolute() or not image_root.is_dir():
        raise ValueError("CheXpert Plus image root must be an absolute directory")
    checked: set[Path] = set()
    for item in benchmark.items:
        for image in item["images"]:
            path = safe_relative_path(image_root, image["relative_path"])
            if path.suffix != ".png":
                raise ValueError(f"Required image does not have .png suffix: {path}")
            if not path.is_file():
                raise ValueError(f"Required image is not a regular file: {path}")
            if path.stat().st_size <= 0:
                raise ValueError(f"Required image is empty: {path}")
            checked.add(path)
    return len(checked)


def _check_endpoint(experiment: Mapping[str, Any]) -> tuple[str, ...]:
    """Discover endpoint models and require the configured identifier exactly."""
    from ..inference.model_client import discover_models

    identifiers = tuple(discover_models(
        str(experiment["base_url"]),
        api_key=os.environ.get("OPENAI_API_KEY"),
    ))
    if experiment["model"] not in identifiers:
        raise ValueError("Endpoint does not advertise the configured model exactly")
    return identifiers


def preflight(
    experiment_path: Path,
    paths_path: Path,
    check_endpoint: bool = True,
) -> dict[str, Any]:
    """Run every read-only benchmark, prompt, path, endpoint, and call-plan gate."""
    from ..inference.experiment_config import (
        load_experiment,
        resolve_benchmark_path,
    )
    from ..inference.prepare_run import build_registries, run_dir_for

    loaded_experiment = load_experiment(Path(experiment_path))
    experiment = _experiment_mapping(loaded_experiment)
    benchmark_path = resolve_benchmark_path(experiment)
    benchmark = load_and_validate_benchmark(benchmark_path)
    validate_prompt_snapshots()
    registries = build_registries(
        experiment, benchmark.manifest, benchmark.items,
    )
    call_plan = _validate_registries(registries, benchmark.items)
    _check_run_target(run_dir_for(experiment), experiment, benchmark)

    private_paths = _load_private_paths(Path(paths_path))
    require_exact_keys(
        private_paths,
        {"chexpert_plus_image_root"},
        "private paths",
    )
    image_count = _check_image_files(
        benchmark,
        private_paths["chexpert_plus_image_root"],
    )
    if check_endpoint:
        _check_endpoint(experiment)

    return {
        "status": "passed",
        "benchmark_version": benchmark.manifest["benchmark_version"],
        "trajectory_count": len(benchmark.items),
        "eligible_final_cases": benchmark.manifest["reference_check"]["eligible_final_cases"],
        **({"selected_final_cases": len(registries["final_cases"])}
           if "task_selection" in benchmark.manifest else {}),
        "required_image_files": image_count,
        "prompt_variants": 19,
        "endpoint_checked": check_endpoint,
        "model": experiment["model"],
        "call_plan": call_plan,
    }


def _parse_args() -> argparse.Namespace:
    """Parse the standalone preflight command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--paths", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run preflight and print only its privacy-safe summary."""
    args = _parse_args()
    summary = preflight(
        args.experiment,
        args.paths,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

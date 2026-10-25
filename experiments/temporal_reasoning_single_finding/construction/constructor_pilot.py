"""Construct the frozen pilot benchmark and its derived smoke benchmark."""

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifact_io import write_json
from ..protocol.contracts import (
    ANNOTATION_SOURCES,
    PILOT_BENCHMARK_VERSION,
    SMOKE_BENCHMARK_VERSION,
    TASKS,
)
from .benchmark_artifacts import (
    DEFAULT_AUDIT_SIDECAR,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_FOUR_TP_SOURCE,
    DEFAULT_THREE_TP_SOURCE,
    build_manifest,
    source_files,
    validate_cross_source_references,
)
from .sampling import PILOT_QUOTA_PER_GROUP, PILOT_SEED, select_pilot_candidates
from .source_data import (
    SOURCE_CONTRACT_RECONCILIATION,
    build_benchmark_item,
    load_audit_index,
    load_source_candidates,
)


def _all_tasks_are_eligible(item: dict[str, Any]) -> bool:
    """Return whether one item supports every frozen final task."""
    return all(item["tasks"][task]["eligible"] for task in TASKS)


def _derive_smoke_items(
    pilot_items: Mapping[int, Sequence[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Select and version the two fixed all-task-eligible smoke items."""
    selections = ((3, "silver"), (4, "gold"))
    smoke: dict[int, list[dict[str, Any]]] = {3: [], 4: []}
    for timepoint_count, annotation_source in selections:
        eligible = [
            item
            for item in pilot_items[timepoint_count]
            if item["annotation_source"] == annotation_source
            and _all_tasks_are_eligible(item)
        ]
        if not eligible:
            raise ValueError(
                "Pilot has no all-task-eligible smoke item for "
                f"{timepoint_count}TP {annotation_source}"
            )
        selected = copy.deepcopy(min(eligible, key=lambda item: item["trajectory_id"]))
        selected["benchmark_version"] = SMOKE_BENCHMARK_VERSION
        smoke[timepoint_count].append(selected)
    return smoke


def construct_benchmarks(
    *,
    three_tp_source: Path = DEFAULT_THREE_TP_SOURCE,
    four_tp_source: Path = DEFAULT_FOUR_TP_SOURCE,
    audit_sidecar: Path = DEFAULT_AUDIT_SIDECAR,
    output_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, Any]:
    """Construct pilot_v1 first, then derive smoke_v1 from its frozen items."""
    audit_index = load_audit_index(audit_sidecar)
    source_results = {
        3: load_source_candidates(
            three_tp_source,
            timepoint_count=3,
            audit_index=audit_index,
        ),
        4: load_source_candidates(
            four_tp_source,
            timepoint_count=4,
            audit_index=audit_index,
        ),
    }
    validate_cross_source_references(source_results)
    selected = select_pilot_candidates(
        {
            timepoint_count: source_results[timepoint_count].candidates
            for timepoint_count in (3, 4)
        },
        quota_per_group=PILOT_QUOTA_PER_GROUP,
        seed=PILOT_SEED,
    )
    pilot_items = {
        timepoint_count: sorted(
            (
                build_benchmark_item(
                    candidate,
                    benchmark_version=PILOT_BENCHMARK_VERSION,
                )
                for candidate in selected[timepoint_count]
            ),
            key=lambda item: item["trajectory_id"],
        )
        for timepoint_count in (3, 4)
    }

    source_declarations = source_files(
        three_tp_source=three_tp_source,
        four_tp_source=four_tp_source,
        audit_sidecar=audit_sidecar,
        audit_record_count=audit_index.record_count,
        source_results=source_results,
    )
    quality_counts = {
        f"{timepoint_count}tp": (
            source_results[timepoint_count].quality_exclusion_counts
        )
        for timepoint_count in (3, 4)
    }

    pilot_dir = output_root / "pilot_v1"
    pilot_paths = {
        timepoint_count: pilot_dir / f"{timepoint_count}tp.json"
        for timepoint_count in (3, 4)
    }
    for timepoint_count in (3, 4):
        write_json(pilot_paths[timepoint_count], pilot_items[timepoint_count])
    pilot_manifest = build_manifest(
        benchmark_version=PILOT_BENCHMARK_VERSION,
        items_by_timepoint=pilot_items,
        file_paths=pilot_paths,
        source_files=source_declarations,
        construction={
            "kind": "pilot",
            "seed": PILOT_SEED,
            "sampling_method": "patient-first-round-robin-v1",
            "fixed_quotas": {
                f"{timepoint_count}tp": {
                    source: PILOT_QUOTA_PER_GROUP
                    for source in ANNOTATION_SOURCES
                }
                for timepoint_count in (3, 4)
            },
            "quality_exclusion_counts": quality_counts,
            "source_contract_reconciliation": SOURCE_CONTRACT_RECONCILIATION,
        },
    )
    write_json(pilot_dir / "manifest.json", pilot_manifest)

    smoke_items = _derive_smoke_items(pilot_items)
    smoke_dir = output_root / "smoke_v1"
    smoke_paths = {
        timepoint_count: smoke_dir / f"{timepoint_count}tp.json"
        for timepoint_count in (3, 4)
    }
    for timepoint_count in (3, 4):
        write_json(smoke_paths[timepoint_count], smoke_items[timepoint_count])
    smoke_manifest = build_manifest(
        benchmark_version=SMOKE_BENCHMARK_VERSION,
        items_by_timepoint=smoke_items,
        file_paths=smoke_paths,
        source_files=source_declarations,
        construction={
            "kind": "derived_smoke",
            "derived_from_benchmark_version": PILOT_BENCHMARK_VERSION,
            "selection_rule": "first-all-task-eligible-by-trajectory-id-v1",
            "fixed_quotas": {
                "3tp": {"silver": 1, "gold": 0},
                "4tp": {"silver": 0, "gold": 1},
            },
            "quality_exclusion_counts": quality_counts,
            "source_contract_reconciliation": SOURCE_CONTRACT_RECONCILIATION,
        },
    )
    write_json(smoke_dir / "manifest.json", smoke_manifest)

    return {
        "pilot_manifest": pilot_manifest,
        "pilot_manifest_path": str((pilot_dir / "manifest.json").resolve()),
        "smoke_manifest": smoke_manifest,
        "smoke_manifest_path": str((smoke_dir / "manifest.json").resolve()),
    }


def main() -> None:
    """Construct both benchmark versions at the frozen artifact root."""
    result = construct_benchmarks()
    print(
        json.dumps(
            {
                "pilot_manifest": result["pilot_manifest_path"],
                "smoke_manifest": result["smoke_manifest_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

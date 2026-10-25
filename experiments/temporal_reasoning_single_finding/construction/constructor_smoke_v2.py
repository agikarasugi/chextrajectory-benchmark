"""Freeze a 16-trajectory assigned-task smoke sample from Gold pilot v1."""

import copy
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from ..artifact_io import sha256_file, write_json
from ..inference.prepare_run import build_registries
from ..protocol.contracts import TASKS
from ..protocol.task_selection import (
    GOLD_PILOT_BENCHMARK_VERSION, SMOKE_V2_BENCHMARK_VERSION,
)
from ..validation.preflight import load_and_validate_benchmark
from .benchmark_artifacts import DEFAULT_BENCHMARK_ROOT, build_manifest
from .gold_common import _publish_or_reuse


SMOKE_SEED = 42
SELECTION_RULE = "assigned-task-patient-preferred-shuffle-v1"


def select_smoke_items(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Select two per task and length, preferring globally unused patients."""
    if len({item["trajectory_id"] for item in items}) != len(items):
        raise ValueError("Duplicate parent trajectory ID")
    if any(
        item["benchmark_version"] != GOLD_PILOT_BENCHMARK_VERSION
        or item["annotation_source"] != "gold"
        or item.get("assigned_task") not in TASKS
        or not item["tasks"][item["assigned_task"]]["eligible"]
        for item in items
    ):
        raise ValueError("Smoke v2 requires assigned, eligible Gold pilot items")
    rng = random.Random(SMOKE_SEED)
    used_patients: set[str] = set()
    selected: dict[int, list[dict[str, Any]]] = {3: [], 4: []}
    for length in (3, 4):
        for task in TASKS:
            candidates = sorted(
                (item for item in items if item["timepoint_count"] == length
                 and item["assigned_task"] == task),
                key=lambda item: item["trajectory_id"],
            )
            if len(candidates) < 2:
                raise ValueError(f"Insufficient Gold pilot items for {length}TP {task}")
            rng.shuffle(candidates)
            for _ in range(2):
                index = next(
                    (index for index, item in enumerate(candidates)
                     if item["patient_id"] not in used_patients),
                    0,
                )
                item = copy.deepcopy(candidates.pop(index))
                used_patients.add(item["patient_id"])
                item["benchmark_version"] = SMOKE_V2_BENCHMARK_VERSION
                selected[length].append(item)
        selected[length].sort(key=lambda item: item["trajectory_id"])
    return selected


def construct_smoke_v2(
    *, parent_manifest: Path = DEFAULT_BENCHMARK_ROOT / "gold_pilot_v1/manifest.json",
    output_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, Any]:
    """Validate frozen parent text, select the sample, and publish without inference."""
    parent = load_and_validate_benchmark(parent_manifest)
    if parent.manifest["benchmark_version"] != GOLD_PILOT_BENCHMARK_VERSION:
        raise ValueError("Smoke v2 must derive from Gold pilot v1")
    items = select_smoke_items(parent.items)
    construction = {
        "kind": "derived_smoke",
        "derived_from_benchmark_version": GOLD_PILOT_BENCHMARK_VERSION,
        "parent_manifest_sha256": sha256_file(parent.manifest_path),
        "parent_files": copy.deepcopy(parent.manifest["files"]),
        "selection_rule": SELECTION_RULE,
        "seed": SMOKE_SEED,
        "length_order": [3, 4],
        "task_order": list(TASKS),
        "quota_per_task_per_length": 2,
        "selected_trajectory_ids": {
            f"{length}tp": [item["trajectory_id"] for item in items[length]]
            for length in (3, 4)
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "smoke_v2"
    with tempfile.TemporaryDirectory(prefix=".smoke_v2-", dir=output_root) as temporary:
        staged = Path(temporary) / "smoke_v2"
        paths = {length: staged / f"{length}tp.json" for length in (3, 4)}
        for length, path in paths.items():
            write_json(path, items[length])
        manifest = build_manifest(
            benchmark_version=SMOKE_V2_BENCHMARK_VERSION,
            items_by_timepoint=items, file_paths=paths,
            source_files=copy.deepcopy(parent.manifest["source_files"]),
            construction=construction,
        )
        write_json(staged / "manifest.json", manifest)
        benchmark = load_and_validate_benchmark(staged / "manifest.json")
        call_plan = build_registries(
            {"model": "smoke-v2-call-plan"}, manifest, benchmark.items,
        )["call_plan"]
        _publish_or_reuse(staged, destination)
    return {
        "smoke_manifest_path": str((destination / "manifest.json").resolve()),
        "call_plan": call_plan,
    }


def main() -> None:
    """Construct the frozen sample and print its offline call budget."""
    print(json.dumps(construct_smoke_v2(), indent=2))


if __name__ == "__main__":
    main()

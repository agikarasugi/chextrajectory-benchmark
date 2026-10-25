"""Fixture-only tests for benchmark loading and preflight gates."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    sha256_file,
    write_json,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    DATASET,
    FINAL_PROMPT_VERSION,
    IMAGE_PATH_CONTRACT,
    JSON_SCHEMA_VERSION,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
    TASKS,
)
from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
    derive_answer,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    load_and_validate_benchmark,
    preflight,
    validate_prompt_snapshots,
)
from experiments.temporal_reasoning_single_finding.validation.reference_check import (
    run_reference_check,
)


def _item(count: int, annotation_source: str) -> dict[str, object]:
    """Create a complete valid metadata-only benchmark item."""
    source_id = f"{annotation_source}-{count}tp"
    labels = ["unchanged"] * (count - 1)
    tasks = {
        task: {
            "eligible": True,
            "reference_answer": derive_answer(task, labels),
            "exclusion_reason": None,
        }
        for task in TASKS
    }
    return {
        "benchmark_version": "fixture-compatible-v1",
        "trajectory_id": f"{DATASET}:{source_id}",
        "dataset": DATASET,
        "annotation_source": annotation_source,
        "source_sequence_id": source_id,
        "patient_id": f"patient-{annotation_source}-{count}",
        "timepoint_count": count,
        "target_finding": {
            "canonical_name": "edema",
            "display_name": "edema",
        },
        "images": [
            {
                "timepoint": f"T{index}",
                "image_id": f"{annotation_source}/{count}/visit-{index}.jpg",
                "relative_path": f"{annotation_source}/{count}/visit-{index}.png",
            }
            for index in range(1, count + 1)
        ],
        "source_interval_labels": ["Stable"] * (count - 1),
        "reference_change_labels": labels,
        "tasks": tasks,
    }


def _counts() -> dict[str, object]:
    """Return exact aggregate counts for the two fixture trajectories."""
    subgroup_distinct = {
        "3tp": {"silver": 1, "gold": 0},
        "4tp": {"silver": 0, "gold": 1},
    }
    return {
        "total_items": 2,
        "by_dataset": {DATASET: 2},
        "by_annotation_source": {"silver": 1, "gold": 1},
        "by_timepoint_count": {"3": 1, "4": 1},
        "by_target_finding": {"edema": 2},
        "eligible_by_task": {task: 2 for task in TASKS},
        "distinct_patients": {
            "overall": 2,
            "by_annotation_source": {"silver": 1, "gold": 1},
            "by_timepoint_and_annotation_source": subgroup_distinct,
        },
        "maximum_selected_trajectories_per_patient": {
            "overall": 1,
            "by_annotation_source": {"silver": 1, "gold": 1},
            "by_timepoint_and_annotation_source": subgroup_distinct,
        },
        "cross_length_input_overlap_by_annotation_source": {
            tier: {
                "shared_image_ids": 0,
                "shared_ordered_adjacent_image_pairs": 0,
                "three_tp_trajectories_with_shared_images": 0,
                "four_tp_trajectories_with_shared_images": 0,
            }
            for tier in ("silver", "gold")
        },
    }


class PreflightTests(unittest.TestCase):
    """Exercise checksums, counts, prompt snapshots, paths, and call plans."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark_dir = self.root / "benchmark"
        self.benchmark_dir.mkdir()
        self.image_root = self.root / "images"
        self.image_root.mkdir()
        self.items = {3: [_item(3, "silver")], 4: [_item(4, "gold")]}
        for items in self.items.values():
            for item in items:
                for image in item["images"]:
                    image_path = self.image_root / image["relative_path"]
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"fixture-not-an-image")

        files: dict[str, dict[str, object]] = {}
        for count in (3, 4):
            path = self.benchmark_dir / f"{count}tp.json"
            write_json(path, self.items[count])
            files[f"{count}tp"] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "item_count": 1,
                "timepoint_count": count,
                "annotation_source_counts": {
                    "silver": int(count == 3),
                    "gold": int(count == 4),
                },
            }
        all_items = [*self.items[3], *self.items[4]]
        self.manifest = {
            "benchmark_version": "fixture-compatible-v1",
            "pair_prompt_version": PAIR_PROMPT_VERSION,
            "seq_prompt_version": SEQ_PROMPT_VERSION,
            "final_prompt_version": FINAL_PROMPT_VERSION,
            "json_schema_version": JSON_SCHEMA_VERSION,
            "dataset": DATASET,
            "image_path_contract": IMAGE_PATH_CONTRACT,
            "tasks": list(TASKS),
            "files": files,
            "source_files": {
                "3tp": {"path": "source-3tp.json", "sha256": "0" * 64},
                "4tp": {"path": "source-4tp.json", "sha256": "1" * 64},
                "audit_sidecar": {
                    "path": "audit.json",
                    "sha256": "2" * 64,
                    "role": "read_only_validation_sidecar",
                },
            },
            "construction": {},
            "counts": _counts(),
            "reference_check": run_reference_check(all_items),
            "validation": {
                "status": "passed",
                "benchmark_item_schema": "passed",
                "output_checksums": "passed",
                "reference_check": "passed",
                "source_audit_reconciliation": "passed",
                "source_schema": "passed",
            },
        }
        self.manifest_path = self.benchmark_dir / "manifest.json"
        write_json(self.manifest_path, self.manifest)
        self.experiment_path = self.root / "experiment.json"
        write_json(
            self.experiment_path,
            {
                "run_name": "validation-fixture-never-created",
                "model": "fixture-model",
                "base_url": "http://127.0.0.1:8000/v1",
                "benchmark_path": str(self.manifest_path),
            },
        )
        self.paths_path = self.root / "paths.json"
        write_json(
            self.paths_path,
            {"chexpert_plus_image_root": str(self.image_root)},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loads_complete_compatible_benchmark(self) -> None:
        benchmark = load_and_validate_benchmark(self.manifest_path)
        self.assertEqual(len(benchmark.items), 2)
        self.assertEqual(set(benchmark.items_by_timepoint), {3, 4})

    def test_rejects_benchmark_checksum_change(self) -> None:
        path = self.benchmark_dir / "3tp.json"
        changed = read_json(path)
        changed[0]["patient_id"] = "changed-after-freeze"
        write_json(path, changed)

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_and_validate_benchmark(self.manifest_path)

    def test_rejects_manifest_count_change(self) -> None:
        self.manifest["counts"]["total_items"] = 3
        write_json(self.manifest_path, self.manifest)

        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            load_and_validate_benchmark(self.manifest_path)

    def test_all_prompt_snapshots_pass(self) -> None:
        self.assertEqual(len(validate_prompt_snapshots()), 19)

    def test_preflight_reports_exact_unreduced_call_plan(self) -> None:
        report = preflight(self.experiment_path, self.paths_path, check_endpoint=False)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["required_image_files"], 7)
        self.assertEqual(report["eligible_final_cases"], 8)
        self.assertEqual(
            report["call_plan"]["potential_logical_calls_before_retries"],
            39,
        )
        self.assertEqual(
            report["call_plan"]["unreduced_maximum_calls_before_retries"],
            39,
        )

    def test_preflight_checks_png_metadata_without_opening_content(self) -> None:
        missing = self.image_root / "silver/3/visit-1.png"
        missing.unlink()

        with self.assertRaisesRegex(ValueError, "not a regular file"):
            preflight(self.experiment_path, self.paths_path, check_endpoint=False)

    def test_preflight_requires_endpoint_model_when_enabled(self) -> None:
        target = (
            "experiments.temporal_reasoning_single_finding.validation.preflight."
            "_check_endpoint"
        )
        with patch(target, return_value=("fixture-model",)) as check:
            report = preflight(self.experiment_path, self.paths_path)

        check.assert_called_once()
        self.assertTrue(report["endpoint_checked"])


if __name__ == "__main__":
    unittest.main()

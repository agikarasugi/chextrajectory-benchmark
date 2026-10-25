"""Tests for strict source loading and benchmark-item conversion."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.temporal_reasoning_single_finding.artifact_io import write_json
from experiments.temporal_reasoning_single_finding.construction.source_data import (
    build_benchmark_item,
    load_audit_index,
    load_source_candidates,
    source_labels_are_state_consistent,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    MOST_SEVERE_EXCLUSION,
    PILOT_BENCHMARK_VERSION,
)


def _source_ids(record: dict[str, Any]) -> tuple[str, str]:
    """Return the two established MD5 source identities for a fixture row."""
    images = [point["image_path"] for point in record["timepoints"]]
    sequence = hashlib.md5(
        "|".join(
            [record["tier"], record["patient_id"], record["finding"], *images]
        ).encode("utf-8")
    ).hexdigest()
    image_sequence = hashlib.md5(
        "|".join([record["tier"], record["patient_id"], *images]).encode(
            "utf-8"
        )
    ).hexdigest()
    return sequence, image_sequence


def _record(
    *,
    patient_number: int,
    labels: list[str],
    tier: str = "silver",
    filenames: list[str] | None = None,
    order_mins: list[int] | None = None,
    order_maxs: list[int] | None = None,
) -> dict[str, Any]:
    """Build one source fixture with identities derived from its content."""
    timepoint_count = len(labels) + 1
    patient_id = f"patient{patient_number:05d}"
    filenames = filenames or [
        f"view{index}_frontal.jpg" for index in range(1, timepoint_count + 1)
    ]
    order_mins = order_mins or list(range(1, timepoint_count + 1))
    order_maxs = order_maxs or order_mins
    timepoints = [
        {
            "timepoint": f"T{index}",
            "study_id": f"study{index}",
            "image_path": (
                f"train/{patient_id}/study{index}/{filenames[index - 1]}"
            ),
            "patient_report_date_order_min": order_mins[index - 1],
            "patient_report_date_order_max": order_maxs[index - 1],
        }
        for index in range(1, timepoint_count + 1)
    ]
    record: dict[str, Any] = {
        "sequence_id": "",
        "image_sequence_id": "",
        "patient_id": patient_id,
        "tier": tier,
        "finding": "pleural_effusion",
        "timepoints": timepoints,
        "change_labels": labels,
    }
    record["sequence_id"], record["image_sequence_id"] = _source_ids(record)
    return record


def _audit_row(
    record: dict[str, Any],
    *,
    is_unambiguous: bool = True,
    all_images_match_csv: bool = True,
) -> dict[str, Any]:
    """Build the audit-sidecar row corresponding to one source fixture."""
    return {
        "sequence_id": record["sequence_id"],
        "image_sequence_id": record["image_sequence_id"],
        "tier": record["tier"],
        "time_points": len(record["timepoints"]),
        "patient_id": record["patient_id"],
        "finding_token": record["finding"],
        "study_ids": [point["study_id"] for point in record["timepoints"]],
        "image_paths": [
            point["image_path"] for point in record["timepoints"]
        ],
        "study_order_min": [
            point["patient_report_date_order_min"]
            for point in record["timepoints"]
        ],
        "study_order_max": [
            point["patient_report_date_order_max"]
            for point in record["timepoints"]
        ],
        "transition_labels": record["change_labels"],
        "is_unambiguous": is_unambiguous,
        "all_images_match_csv": all_images_match_csv,
    }


def _write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write metadata-only audit fixture rows to Parquet."""
    pd.DataFrame(rows).to_parquet(path, index=False)


class SourceDataTests(unittest.TestCase):
    """Exercise strict schemas, quality filters, and item mapping."""

    def test_loads_and_maps_one_valid_record(self) -> None:
        """A valid row maps exactly and omits internal image_sequence_id."""
        record = _record(patient_number=1, labels=["New", "Improved"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "3tp.json"
            audit_path = root / "audit.parquet"
            write_json(source_path, [record])
            _write_audit(audit_path, [_audit_row(record)])

            audit = load_audit_index(audit_path)
            result = load_source_candidates(
                source_path,
                timepoint_count=3,
                audit_index=audit,
            )
            self.assertEqual(len(result.candidates), 1)
            self.assertEqual(result.excluded_record_count, 0)
            item = build_benchmark_item(
                result.candidates[0],
                benchmark_version=PILOT_BENCHMARK_VERSION,
            )

        self.assertEqual(item["annotation_source"], "silver")
        self.assertEqual(item["dataset"], "chexpert_plus")
        self.assertEqual(
            item["trajectory_id"], f"chexpert_plus:{record['sequence_id']}"
        )
        self.assertEqual(
            item["reference_change_labels"],
            ["deterioration", "improvement"],
        )
        self.assertEqual(
            item["tasks"]["trajectory_shape"]["reference_answer"],
            "single_peak",
        )
        self.assertEqual(
            item["tasks"]["turning_regions"]["reference_answer"],
            "peak:T2",
        )
        self.assertEqual(
            item["images"][0]["relative_path"],
            "train/patient00001/study1/view1_frontal.png",
        )
        self.assertNotIn("image_sequence_id", item)

    def test_quality_exclusions_are_counted_without_stopping(self) -> None:
        """State, listing, frontal, and chronology failures are exclusions."""
        state_bad = _record(
            patient_number=2,
            labels=["Stable", "New"],
        )
        listing_bad = _record(
            patient_number=3,
            labels=["New", "Improved"],
        )
        structural_bad = _record(
            patient_number=4,
            labels=["New", "Improved"],
            filenames=[
                "view1_frontal.jpg",
                "view2_lateral.jpg",
                "view3_frontal.jpg",
            ],
            order_mins=[1, 1, 3],
            order_maxs=[1, 2, 3],
        )
        rows = [state_bad, listing_bad, structural_bad]
        audit_rows = [
            _audit_row(state_bad),
            _audit_row(listing_bad, all_images_match_csv=False),
            _audit_row(structural_bad),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "3tp.json"
            audit_path = root / "audit.parquet"
            write_json(source_path, rows)
            _write_audit(audit_path, audit_rows)
            result = load_source_candidates(
                source_path,
                timepoint_count=3,
                audit_index=load_audit_index(audit_path),
            )

        self.assertEqual(result.candidates, ())
        self.assertEqual(result.excluded_record_count, 3)
        self.assertEqual(result.quality_exclusion_counts["state_inconsistent"], 1)
        self.assertEqual(
            result.quality_exclusion_counts["source_image_not_listed"], 1
        )
        self.assertEqual(result.quality_exclusion_counts["non_frontal_images"], 1)
        self.assertEqual(result.quality_exclusion_counts["invalid_chronology"], 1)

    def test_ambiguous_or_conflicting_audit_row_is_fatal(self) -> None:
        """The JSON export must agree with one unambiguous audit row."""
        record = _record(patient_number=5, labels=["New", "Improved"])
        for mutation in ("ambiguous", "conflicting"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_path = root / "3tp.json"
                audit_path = root / "audit.parquet"
                write_json(source_path, [record])
                audit_row = _audit_row(
                    record,
                    is_unambiguous=mutation != "ambiguous",
                )
                if mutation == "conflicting":
                    audit_row["finding_token"] = "edema"
                _write_audit(audit_path, [audit_row])
                with self.assertRaises(ValueError):
                    load_source_candidates(
                        source_path,
                        timepoint_count=3,
                        audit_index=load_audit_index(audit_path),
                    )

    def test_wrong_schema_and_unsafe_path_are_fatal(self) -> None:
        """Malformed fields stop construction instead of becoming exclusions."""
        record = _record(patient_number=6, labels=["New", "Improved"])
        malformed = dict(record)
        malformed["unexpected"] = True
        unsafe = _record(patient_number=7, labels=["New", "Improved"])
        unsafe["timepoints"][0]["image_path"] = "/absolute/image.jpg"
        unsafe["sequence_id"], unsafe["image_sequence_id"] = _source_ids(unsafe)
        for source_record in (malformed, unsafe):
            with (
                self.subTest(record=source_record["patient_id"]),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_path = root / "3tp.json"
                audit_path = root / "audit.parquet"
                write_json(source_path, [source_record])
                _write_audit(audit_path, [_audit_row(source_record)])
                with self.assertRaises(ValueError):
                    load_source_candidates(
                        source_path,
                        timepoint_count=3,
                        audit_index=load_audit_index(audit_path),
                    )

    def test_unidentifiable_most_severe_task_is_explicit(self) -> None:
        """A valid trough retains the trajectory but excludes one task."""
        record = _record(patient_number=8, labels=["Improved", "Worse"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "3tp.json"
            audit_path = root / "audit.parquet"
            write_json(source_path, [record])
            _write_audit(audit_path, [_audit_row(record)])
            result = load_source_candidates(
                source_path,
                timepoint_count=3,
                audit_index=load_audit_index(audit_path),
            )
            item = build_benchmark_item(
                result.candidates[0],
                benchmark_version=PILOT_BENCHMARK_VERSION,
            )

        task = item["tasks"]["most_severe_visit_or_plateau"]
        self.assertFalse(task["eligible"])
        self.assertIsNone(task["reference_answer"])
        self.assertEqual(task["exclusion_reason"], MOST_SEVERE_EXCLUSION)
        self.assertTrue(source_labels_are_state_consistent(("Resolved", "New")))
        self.assertFalse(source_labels_are_state_consistent(("Stable", "New")))


if __name__ == "__main__":
    unittest.main()

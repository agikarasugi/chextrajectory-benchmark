"""Tests for patient-first selection and deterministic benchmark builds."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    sha256_file,
    write_json,
)
from experiments.temporal_reasoning_single_finding.construction import (
    benchmark_artifacts,
    constructor_pilot,
)
from experiments.temporal_reasoning_single_finding.construction.sampling import (
    patient_first_sample,
    select_pilot_candidates,
)
from experiments.temporal_reasoning_single_finding.construction.source_data import (
    SourceCandidate,
    SourceTimepoint,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    PILOT_BENCHMARK_VERSION,
    SMOKE_BENCHMARK_VERSION,
)


def _candidate(
    *,
    patient_number: int,
    timepoint_count: int,
    annotation_source: str,
    variant: int = 0,
) -> SourceCandidate:
    """Build one eligible in-memory sampling candidate."""
    patient_id = f"patient{patient_number:05d}"
    points = tuple(
        SourceTimepoint(
            timepoint=f"T{index}",
            study_id=f"study{variant * 10 + index}",
            image_path=(
                f"train/{patient_id}/study{variant * 10 + index}/"
                f"view{index}_frontal.jpg"
            ),
            order_min=variant * 10 + index,
            order_max=variant * 10 + index,
        )
        for index in range(1, timepoint_count + 1)
    )
    image_paths = [point.image_path for point in points]
    source_id = hashlib.md5(
        "|".join(
            [annotation_source, patient_id, "edema", *image_paths]
        ).encode("utf-8")
    ).hexdigest()
    image_id = hashlib.md5(
        "|".join([annotation_source, patient_id, *image_paths]).encode("utf-8")
    ).hexdigest()
    labels = (
        ("New", "Improved")
        if timepoint_count == 3
        else ("New", "Stable", "Improved")
    )
    return SourceCandidate(
        source_sequence_id=source_id,
        image_sequence_id=image_id,
        patient_id=patient_id,
        annotation_source=annotation_source,
        timepoint_count=timepoint_count,
        target_finding="edema",
        timepoints=points,
        source_interval_labels=labels,
    )


def _source_record(candidate: SourceCandidate) -> dict[str, Any]:
    """Convert an in-memory candidate to the strict source JSON schema."""
    return {
        "sequence_id": candidate.source_sequence_id,
        "image_sequence_id": candidate.image_sequence_id,
        "patient_id": candidate.patient_id,
        "tier": candidate.annotation_source,
        "finding": candidate.target_finding,
        "timepoints": [
            {
                "timepoint": point.timepoint,
                "study_id": point.study_id,
                "image_path": point.image_path,
                "patient_report_date_order_min": point.order_min,
                "patient_report_date_order_max": point.order_max,
            }
            for point in candidate.timepoints
        ],
        "change_labels": list(candidate.source_interval_labels),
    }


def _audit_row(candidate: SourceCandidate) -> dict[str, Any]:
    """Convert one candidate to the required audit validation fields."""
    return {
        "sequence_id": candidate.source_sequence_id,
        "image_sequence_id": candidate.image_sequence_id,
        "tier": candidate.annotation_source,
        "time_points": candidate.timepoint_count,
        "patient_id": candidate.patient_id,
        "finding_token": candidate.target_finding,
        "study_ids": [point.study_id for point in candidate.timepoints],
        "image_paths": [point.image_path for point in candidate.timepoints],
        "study_order_min": [point.order_min for point in candidate.timepoints],
        "study_order_max": [point.order_max for point in candidate.timepoints],
        "transition_labels": list(candidate.source_interval_labels),
        "is_unambiguous": True,
        "all_images_match_csv": True,
    }


def _contains_key(value: Any, key: str) -> bool:
    """Return whether a key occurs anywhere in a JSON-like value."""
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


class SamplingTests(unittest.TestCase):
    """Exercise round structure, quotas, and full deterministic construction."""

    def test_patient_first_round_precedes_second_trajectory(self) -> None:
        """Each patient contributes once before any patient contributes twice."""
        candidates = [
            _candidate(
                patient_number=patient,
                timepoint_count=3,
                annotation_source="silver",
                variant=variant,
            )
            for patient in (1, 2)
            for variant in (0, 1)
        ]
        first = patient_first_sample(candidates, quota=3, seed=42)
        second = patient_first_sample(list(reversed(candidates)), quota=3, seed=42)
        self.assertEqual(
            [candidate.source_sequence_id for candidate in first],
            [candidate.source_sequence_id for candidate in second],
        )
        self.assertEqual(len({candidate.patient_id for candidate in first[:2]}), 2)
        self.assertEqual(len({candidate.source_sequence_id for candidate in first}), 3)

    def test_selects_exact_independent_group_quotas(self) -> None:
        """The four subgroups each receive their complete independent quota."""
        candidates_by_timepoint = {
            timepoint_count: tuple(
                _candidate(
                    patient_number=(
                        timepoint_count * 1000 + source_index * 100 + index
                    ),
                    timepoint_count=timepoint_count,
                    annotation_source=annotation_source,
                )
                for source_index, annotation_source in enumerate(("silver", "gold"))
                for index in range(5)
            )
            for timepoint_count in (3, 4)
        }
        selected = select_pilot_candidates(
            candidates_by_timepoint,
            quota_per_group=3,
            seed=42,
        )
        for timepoint_count in (3, 4):
            counts = Counter(
                candidate.annotation_source
                for candidate in selected[timepoint_count]
            )
            self.assertEqual(counts, {"silver": 3, "gold": 3})

    def test_insufficient_group_fails(self) -> None:
        """A subgroup below quota cannot be silently backfilled or replaced."""
        candidates = [
            _candidate(
                patient_number=index,
                timepoint_count=3,
                annotation_source="silver",
            )
            for index in range(2)
        ]
        with self.assertRaises(ValueError):
            patient_first_sample(candidates, quota=3, seed=42)

    def test_full_constructor_is_byte_deterministic(self) -> None:
        """Two builds produce identical pilot and smoke files and manifests."""
        candidates_by_timepoint: dict[int, list[SourceCandidate]] = {3: [], 4: []}
        patient_base = {
            (3, "silver"): 10_000,
            (3, "gold"): 20_000,
            (4, "silver"): 30_000,
            (4, "gold"): 40_000,
        }
        for timepoint_count in (3, 4):
            for annotation_source in ("silver", "gold"):
                candidates_by_timepoint[timepoint_count].extend(
                    _candidate(
                        patient_number=patient_base[
                            (timepoint_count, annotation_source)
                        ]
                        + index,
                        timepoint_count=timepoint_count,
                        annotation_source=annotation_source,
                    )
                    for index in range(100)
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            three_path = root / "source_3tp.json"
            four_path = root / "source_4tp.json"
            audit_path = root / "audit.parquet"
            write_json(
                three_path,
                [_source_record(candidate) for candidate in candidates_by_timepoint[3]],
            )
            write_json(
                four_path,
                [_source_record(candidate) for candidate in candidates_by_timepoint[4]],
            )
            pd.DataFrame(
                [
                    _audit_row(candidate)
                    for timepoint_count in (3, 4)
                    for candidate in candidates_by_timepoint[timepoint_count]
                ]
            ).to_parquet(audit_path, index=False)
            output_a = root / "build-a"
            output_b = root / "build-b"
            constructor_pilot.construct_benchmarks(
                three_tp_source=three_path,
                four_tp_source=four_path,
                audit_sidecar=audit_path,
                output_root=output_a,
            )
            constructor_pilot.construct_benchmarks(
                three_tp_source=three_path,
                four_tp_source=four_path,
                audit_sidecar=audit_path,
                output_root=output_b,
            )

            for version in ("pilot_v1", "smoke_v1"):
                for filename in ("manifest.json", "3tp.json", "4tp.json"):
                    self.assertEqual(
                        (output_a / version / filename).read_bytes(),
                        (output_b / version / filename).read_bytes(),
                    )

            pilot_manifest = read_json(output_a / "pilot_v1/manifest.json")
            smoke_manifest = read_json(output_a / "smoke_v1/manifest.json")
            pilot_three = read_json(output_a / "pilot_v1/3tp.json")
            pilot_four = read_json(output_a / "pilot_v1/4tp.json")
            smoke_three = read_json(output_a / "smoke_v1/3tp.json")
            smoke_four = read_json(output_a / "smoke_v1/4tp.json")
            pilot_three_sha256 = sha256_file(output_a / "pilot_v1/3tp.json")

        self.assertEqual(set(pilot_manifest), benchmark_artifacts.MANIFEST_KEYS)
        self.assertEqual(pilot_manifest["benchmark_version"], PILOT_BENCHMARK_VERSION)
        self.assertEqual(smoke_manifest["benchmark_version"], SMOKE_BENCHMARK_VERSION)
        self.assertEqual(len(pilot_three), 200)
        self.assertEqual(len(pilot_four), 200)
        self.assertEqual(
            Counter(item["annotation_source"] for item in pilot_three),
            {"silver": 100, "gold": 100},
        )
        self.assertEqual(
            Counter(item["annotation_source"] for item in pilot_four),
            {"silver": 100, "gold": 100},
        )
        self.assertEqual(len(smoke_three), 1)
        self.assertEqual(len(smoke_four), 1)
        self.assertEqual(smoke_three[0]["annotation_source"], "silver")
        self.assertEqual(smoke_four[0]["annotation_source"], "gold")
        self.assertTrue(
            all(
                entry["eligible"]
                for entry in smoke_three[0]["tasks"].values()
            )
        )
        self.assertTrue(
            all(
                entry["eligible"]
                for entry in smoke_four[0]["tasks"].values()
            )
        )
        self.assertEqual(
            pilot_manifest["reference_check"]["checked_task_entries"],
            1600,
        )
        self.assertEqual(
            pilot_manifest["reference_check"]["eligible_final_cases"],
            1600,
        )
        self.assertEqual(smoke_manifest["reference_check"]["checked_task_entries"], 8)
        self.assertEqual(
            pilot_manifest["files"]["3tp"]["sha256"],
            pilot_three_sha256,
        )
        self.assertFalse(_contains_key(pilot_manifest, "timestamp"))
        self.assertFalse(_contains_key(smoke_manifest, "timestamp"))
        self.assertNotIn("image_sequence_id", json.dumps([pilot_three, pilot_four]))


if __name__ == "__main__":
    unittest.main()

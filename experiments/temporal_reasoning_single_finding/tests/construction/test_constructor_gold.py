"""Gold-only construction, frozen publication, and generic inference tests."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    sha256_file,
    write_json,
)
from experiments.temporal_reasoning_single_finding.construction.constructor_gold import (
    GOLD_BENCHMARK_VERSION,
    construct_gold,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import (
    build_registries,
)
from experiments.temporal_reasoning_single_finding.tests.construction.test_source_data import (
    _audit_row,
    _record,
    _source_ids,
    _write_audit,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    load_and_validate_benchmark,
)


class GoldConstructorTests(unittest.TestCase):
    """Exercise all-valid-Gold selection without image access or model calls."""

    def setUp(self) -> None:
        """Prepare mixed sources with shared images and one task-only exclusion."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        first = _record(patient_number=1, labels=["New", "Stable"], tier="gold")
        second_finding = copy.deepcopy(first)
        second_finding["finding"] = "edema"
        second_finding["sequence_id"], second_finding["image_sequence_id"] = (
            _source_ids(second_finding)
        )
        self.three = [
            first,
            second_finding,
            _record(patient_number=2, labels=["Improved", "Worse"], tier="gold"),
            _record(patient_number=3, labels=["Stable", "New"], tier="gold"),
            _record(patient_number=20, labels=["New", "Stable"]),
        ]
        self.four = [
            _record(
                patient_number=1, labels=["New", "Stable", "Improved"], tier="gold"
            ),
            _record(
                patient_number=4, labels=["Stable", "New", "Stable"], tier="gold"
            ),
            _record(patient_number=20, labels=["New", "Stable", "Improved"]),
        ]
        self.kwargs = {
            "three_tp_source": self.root / "source-3tp.json",
            "four_tp_source": self.root / "source-4tp.json",
            "audit_sidecar": self.root / "audit.parquet",
            "output_root": self.root / "output",
        }
        self._write_sources()

    def _write_sources(self) -> None:
        """Write fixture source records and their matching audit metadata."""
        write_json(self.kwargs["three_tp_source"], self.three)
        write_json(self.kwargs["four_tp_source"], self.four)
        _write_audit(
            self.kwargs["audit_sidecar"],
            [_audit_row(record) for record in self.three + self.four],
        )

    def _snapshot(self) -> dict[str, tuple[str, int]]:
        """Capture hashes and modification times without reading images."""
        return {
            path.name: (sha256_file(path), path.stat().st_mtime_ns)
            for path in (self.kwargs["output_root"] / "gold_v1").iterdir()
        }

    def test_retains_every_valid_gold_trajectory_and_compatible_manifest(self) -> None:
        """Gold retention is complete, including shared patients/images and troughs."""
        result = construct_gold(**self.kwargs)
        benchmark = load_and_validate_benchmark(Path(result["gold_manifest_path"]))
        manifest = benchmark.manifest
        self.assertEqual(manifest["benchmark_version"], GOLD_BENCHMARK_VERSION)
        self.assertEqual(
            manifest["counts"]["by_annotation_source"], {"gold": 4, "silver": 0}
        )
        self.assertEqual(manifest["counts"]["by_timepoint_count"], {"3": 3, "4": 1})
        expected = {r["sequence_id"] for r in self.three[:3] + self.four[:1]}
        self.assertEqual({i["source_sequence_id"] for i in benchmark.items}, expected)
        self.assertEqual(manifest["counts"]["distinct_patients"]["overall"], 2)
        self.assertEqual(manifest["reference_check"]["eligible_final_cases"], 15)
        self.assertEqual(
            manifest["reference_check"]["ineligible_most_severe_task_entries"], 1
        )
        construction = manifest["construction"]
        self.assertEqual(construction["kind"], "gold_only")
        self.assertEqual(construction["selection_rule"], "all-valid-gold-v1")
        self.assertNotIn("seed", construction)
        self.assertNotIn("fixed_quotas", construction)
        self.assertEqual(manifest["source_files"]["3tp"]["item_count"], 5)
        gold_counts = construction["gold_source_counts"]["3tp"]
        self.assertEqual(gold_counts["source_record_count"], 4)
        self.assertEqual(gold_counts["eligible_record_count"], 3)
        self.assertEqual(gold_counts["excluded_record_count"], 1)
        self.assertEqual(
            gold_counts["quality_exclusion_counts"]["state_inconsistent"], 1
        )
        self.assertEqual(
            gold_counts["source_record_count"],
            gold_counts["eligible_record_count"] + gold_counts["excluded_record_count"],
        )
        for count in (3, 4):
            path = Path(result["gold_manifest_path"]).parent / f"{count}tp.json"
            items = read_json(path)
            ids = [item["trajectory_id"] for item in items]
            self.assertEqual(ids, sorted(ids))
        self.assertEqual(
            {p.name for p in self.kwargs["output_root"].iterdir()}, {"gold_v1"}
        )

    def test_reuses_pairs_across_lengths_and_separates_benchmark_versions(self) -> None:
        """Generic preparation deduplicates shared PAIR inputs, not final calls."""
        result = construct_gold(**self.kwargs)
        benchmark = load_and_validate_benchmark(Path(result["gold_manifest_path"]))
        registry = build_registries(
            {"model": "fixture"}, benchmark.manifest, benchmark.items
        )
        self.assertEqual(registry["call_plan"]["pair_predictions"], 7)
        self.assertEqual(registry["call_plan"]["seq_predictions"], 4)
        self.assertEqual(registry["call_plan"]["seq_dir"], 15)
        self.assertEqual(
            registry["call_plan"]["potential_logical_calls_before_retries"], 71
        )
        other_manifest = dict(
            benchmark.manifest, benchmark_version="another-benchmark-v1"
        )
        other_items = [
            dict(item, benchmark_version="another-benchmark-v1")
            for item in benchmark.items
        ]
        other = build_registries({"model": "fixture"}, other_manifest, other_items)
        for key in ("pair_inputs", "seq_inputs"):
            self.assertTrue(
                {row["prediction_id"] for row in registry[key]}.isdisjoint(
                    row["prediction_id"] for row in other[key]
                )
            )
        # Case/setting IDs stay stable. Versioned inputs prevent response reuse.
        self.assertEqual(
            [row["result_ids"] for row in registry["final_cases"]],
            [row["result_ids"] for row in other["final_cases"]],
        )
        self.assertTrue(
            {row["model_input_id"] for row in registry["model_inputs"]}.isdisjoint(
                row["model_input_id"] for row in other["model_inputs"]
            )
        )

    def test_repeat_build_reuses_files_and_fresh_build_is_identical(self) -> None:
        """Identical sources yield identical bytes without rewriting frozen files."""
        construct_gold(**self.kwargs)
        before = self._snapshot()
        construct_gold(**self.kwargs)
        self.assertEqual(self._snapshot(), before)
        fresh_root = self.root / "fresh"
        construct_gold(**dict(self.kwargs, output_root=fresh_root))
        self.assertEqual(
            {p.name: sha256_file(p) for p in (fresh_root / "gold_v1").iterdir()},
            {name: value[0] for name, value in before.items()},
        )

    def test_source_order_does_not_change_item_order_or_bytes(self) -> None:
        """Source order affects provenance hashes, but never the retained items."""
        result = construct_gold(**self.kwargs)
        original = Path(result["gold_manifest_path"]).parent
        self.three.reverse()
        self.four.reverse()
        self._write_sources()
        result = construct_gold(
            **dict(self.kwargs, output_root=self.root / "reordered")
        )
        reordered = Path(result["gold_manifest_path"]).parent
        for name in ("3tp.json", "4tp.json"):
            self.assertEqual(
                sha256_file(original / name), sha256_file(reordered / name)
            )

    def test_gold_population_is_not_capped_at_pilot_quota(self) -> None:
        """The constructor retains more than 100 Gold trajectories in one cohort."""
        self.three.extend(
            _record(patient_number=number, labels=["New", "Stable"], tier="gold")
            for number in range(100, 201)
        )
        self._write_sources()
        result = construct_gold(**self.kwargs)
        self.assertEqual(
            result["gold_manifest"]["counts"]["by_timepoint_count"],
            {"3": 104, "4": 1},
        )

    def test_symlink_output_cannot_redirect_publication(self) -> None:
        """An existing Gold directory symlink is refused without changing its target."""
        target = self.root / "elsewhere"
        target.mkdir()
        output = self.kwargs["output_root"]
        output.mkdir()
        (output / "gold_v1").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(FileExistsError, "Conflicting frozen"):
            construct_gold(**self.kwargs)
        self.assertEqual(list(target.iterdir()), [])
        self.assertTrue((output / "gold_v1").is_symlink())

    def test_conflicting_frozen_files_are_not_overwritten(self) -> None:
        """A changed or incomplete frozen artifact set fails without any replacement."""
        construct_gold(**self.kwargs)
        path = self.kwargs["output_root"] / "gold_v1/3tp.json"
        write_json(path, [])
        before = self._snapshot()
        with self.assertRaisesRegex(FileExistsError, "Conflicting frozen"):
            construct_gold(**self.kwargs)
        self.assertEqual(self._snapshot(), before)
        path.unlink()
        before = self._snapshot()
        with self.assertRaisesRegex(FileExistsError, "Conflicting frozen"):
            construct_gold(**self.kwargs)
        self.assertEqual(self._snapshot(), before)

    def test_source_change_cannot_replace_frozen_version(self) -> None:
        """Even unchanged Gold membership cannot silently alter source provenance."""
        construct_gold(**self.kwargs)
        before = self._snapshot()
        self.three.append(_record(patient_number=21, labels=["New", "Stable"]))
        self._write_sources()
        with self.assertRaises(FileExistsError):
            construct_gold(**self.kwargs)
        self.assertEqual(self._snapshot(), before)

    def test_failure_before_publication_leaves_no_partial_benchmark(self) -> None:
        """Manifest-validation failure cleans staged files without publishing them."""
        with patch(
            "experiments.temporal_reasoning_single_finding.construction."
            "gold_common.build_manifest",
            side_effect=ValueError("fixture validation failure"),
        ):
            with self.assertRaisesRegex(ValueError, "fixture validation"):
                construct_gold(**self.kwargs)
        self.assertEqual(list(self.kwargs["output_root"].iterdir()), [])

    def test_cross_length_reference_conflict_is_fatal(self) -> None:
        """The same Gold finding and adjacent images cannot have competing labels."""
        self.four[0]["change_labels"][1] = "Worse"
        self._write_sources()
        with self.assertRaisesRegex(
            ValueError, "Conflicting interval reference across"
        ):
            construct_gold(**self.kwargs)
        self.assertFalse(self.kwargs["output_root"].exists())

    def test_duplicate_or_inconsistent_source_identity_is_fatal(self) -> None:
        """Duplicate rows and forged identity fields never get deduplicated away."""
        self.three.append(copy.deepcopy(self.three[0]))
        write_json(self.kwargs["three_tp_source"], self.three)
        with self.assertRaisesRegex(ValueError, "Duplicate source sequence_id"):
            construct_gold(**self.kwargs)
        self.three.pop()
        self.three[0]["finding"] = "pneumothorax"
        write_json(self.kwargs["three_tp_source"], self.three)
        with self.assertRaisesRegex(ValueError, "Source sequence_id conflicts"):
            construct_gold(**self.kwargs)

    def test_gold_quality_counts_are_separate_from_silver(self) -> None:
        """Each tier's quality exclusions are counted only within that tier."""
        missing = _record(patient_number=30, labels=["New", "Stable"], tier="gold")
        lateral = _record(
            patient_number=31, labels=["New", "Stable"], tier="gold",
            filenames=["view1_frontal.jpg", "view2_lateral.jpg", "view3_frontal.jpg"],
        )
        self.three.extend([
            missing,
            lateral,
            _record(patient_number=32, labels=["Stable", "New"]),
        ])
        self._write_sources()
        _write_audit(
            self.kwargs["audit_sidecar"],
            [
                _audit_row(record, all_images_match_csv=record is not missing)
                for record in self.three + self.four
            ],
        )
        result = construct_gold(**self.kwargs)
        manifest = result["gold_manifest"]
        gold_counts = manifest["construction"]["gold_source_counts"]["3tp"]
        self.assertEqual(gold_counts["excluded_record_count"], 3)
        self.assertEqual(
            gold_counts["quality_exclusion_counts"]["state_inconsistent"], 1
        )
        self.assertEqual(
            gold_counts["quality_exclusion_counts"]["source_image_not_listed"], 1
        )
        self.assertEqual(
            gold_counts["quality_exclusion_counts"]["non_frontal_images"], 1
        )
        self.assertEqual(manifest["source_files"]["3tp"]["excluded_item_count"], 4)

    def test_absent_gold_cohort_fails_without_publication(self) -> None:
        """A two-length Gold benchmark cannot silently contain only one cohort."""
        self.four = [row for row in self.four if row["tier"] == "silver"]
        self._write_sources()
        with self.assertRaisesRegex(ValueError, "both 3TP and 4TP"):
            construct_gold(**self.kwargs)
        self.assertFalse(self.kwargs["output_root"].exists())


if __name__ == "__main__":
    unittest.main()

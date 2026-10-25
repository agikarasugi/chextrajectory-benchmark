"""Gold pilot assignment, schema compatibility, and offline integration checks."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json, read_jsonl, sha256_file, write_json, write_jsonl,
)
from experiments.temporal_reasoning_single_finding.construction import constructor_gold_pilot as pilot
from experiments.temporal_reasoning_single_finding.construction.source_data import (
    build_benchmark_item, load_audit_index, load_source_candidates,
)
from experiments.temporal_reasoning_single_finding.evaluation.metrics import summarize_run
from experiments.temporal_reasoning_single_finding.inference.prepare_run import build_registries
from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS, SETTINGS
from experiments.temporal_reasoning_single_finding.protocol.task_selection import (
    GOLD_PILOT_BENCHMARK_VERSION, selection_summary, validate_task_selection,
)
from experiments.temporal_reasoning_single_finding.tests.construction.test_source_data import (
    _record, _audit_row, _write_audit,
)
from experiments.temporal_reasoning_single_finding.tests.evaluation.fixtures import write_run
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    _validate_registries, load_and_validate_benchmark, preflight,
)
from experiments.temporal_reasoning_single_finding.validation.validate_run import load_validated_run


class GoldPilotTests(unittest.TestCase):
    """Use small synthetic metadata cohorts while retaining production contracts."""

    def setUp(self) -> None:
        """Create valid, overlapping Gold fixtures and test-sized task quotas."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.three = [
            _record(patient_number=1, labels=["New", "Stable"], tier="gold"),
            _record(patient_number=2, labels=["Improved", "Worse"], tier="gold"),
            _record(patient_number=3, labels=["New", "Stable"], tier="gold"),
            _record(patient_number=4, labels=["New", "Stable"], tier="silver"),
        ]
        self.four = [
            _record(patient_number=1, labels=["New", "Stable", "Improved"], tier="gold")
        ]
        self.kwargs = {
            "three_tp_source": self.root / "source-3tp.json",
            "four_tp_source": self.root / "source-4tp.json",
            "audit_sidecar": self.root / "audit.parquet",
            "output_root": self.root / "output",
        }
        write_json(self.kwargs["three_tp_source"], self.three)
        write_json(self.kwargs["four_tp_source"], self.four)
        _write_audit(self.kwargs["audit_sidecar"], [
            _audit_row(record) for record in self.three + self.four
        ])
        self.quotas = {
            3: dict(zip(TASKS, (1, 1, 1, 0), strict=True)),
            4: dict(zip(TASKS, (0, 0, 0, 1), strict=True)),
        }

    def _build(self):
        """Build a real compatible manifest with test-sized constructor quotas."""
        with patch.object(pilot, "TASK_QUOTAS", self.quotas):
            result = pilot.construct_gold_pilot(**self.kwargs)
        return load_and_validate_benchmark(Path(result["gold_manifest_path"]))

    def _refresh(self, benchmark, items) -> None:
        """Refresh item checksums to exercise semantic rather than checksum errors."""
        for count in (3, 4):
            path = benchmark.manifest_path.parent / f"{count}tp.json"
            write_json(path, [item for item in items if item["timepoint_count"] == count])
            benchmark.manifest["files"][f"{count}tp"]["sha256"] = sha256_file(path)
        write_json(benchmark.manifest_path, benchmark.manifest)

    def test_exact_quota_assignment_and_reference_preservation(self) -> None:
        """All Gold is retained, with one task selected and all reference entries kept."""
        benchmark = self._build()
        self.assertEqual(len(benchmark.items), 4)
        self.assertEqual(benchmark.manifest["counts"]["by_annotation_source"], {"gold": 4, "silver": 0})
        self.assertEqual(benchmark.manifest["reference_check"]["eligible_final_cases"], 15)
        self.assertEqual(benchmark.manifest["task_selection"]["selected_final_cases"], 4)
        audit = load_audit_index(self.kwargs["audit_sidecar"])
        for count in (3, 4):
            sources = load_source_candidates(
                self.kwargs[f"{'three' if count == 3 else 'four'}_tp_source"],
                timepoint_count=count, audit_index=audit,
            )
            expected = {
                candidate.trajectory_id: build_benchmark_item(
                    candidate, benchmark_version=GOLD_PILOT_BENCHMARK_VERSION
                )
                for candidate in sources.candidates if candidate.annotation_source == "gold"
            }
            selected = [item for item in benchmark.items if item["timepoint_count"] == count]
            self.assertEqual(Counter(item["assigned_task"] for item in selected), Counter(self.quotas[count]))
            for item in selected:
                original = dict(item)
                assigned = original.pop("assigned_task")
                self.assertEqual(original, expected[item["trajectory_id"]])
                self.assertTrue(item["tasks"][assigned]["eligible"])

    def test_determinism_and_frozen_output_reuse(self) -> None:
        """Repeated builds are identical and never rewrite existing frozen files."""
        benchmark = self._build()
        paths = list(benchmark.manifest_path.parent.iterdir())
        before = {path.name: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
        self._build()
        self.assertEqual(before, {path.name: (sha256_file(path), path.stat().st_mtime_ns) for path in paths})
        for count in (3, 4):
            plain = [dict(item) for item in benchmark.items if item["timepoint_count"] == count]
            for item in plain:
                item.pop("assigned_task")
            a = pilot.assign_tasks(plain, self.quotas[count])
            b = pilot.assign_tasks(list(reversed(plain)), self.quotas[count])
            self.assertEqual(a, b)
            self.assertTrue(all("assigned_task" not in item for item in plain))
        write_json(benchmark.manifest_path.parent / "3tp.json", [])
        with self.assertRaisesRegex(FileExistsError, "Conflicting frozen"):
            self._build()

    def test_quota_mismatch_and_insufficient_severe_fail_before_writes(self) -> None:
        """Production quotas never truncate a changed population or invent eligibility."""
        with self.assertRaisesRegex(ValueError, "every valid Gold"):
            pilot.construct_gold_pilot(**self.kwargs)
        self.assertFalse(self.kwargs["output_root"].exists())
        benchmark = self._build()
        plain = [dict(item) for item in benchmark.items if item["timepoint_count"] == 3]
        for item in plain:
            item.pop("assigned_task")
        quotas = dict(zip(TASKS, (0, 0, 3, 0), strict=True))
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            pilot.assign_tasks(plain, quotas)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pilot.assign_tasks([plain[0]] * 3, self.quotas[3])
        quotas[TASKS[2]] = True
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            pilot.assign_tasks(plain, quotas)

    def test_missing_null_unknown_and_ineligible_assignments_are_rejected(self) -> None:
        """Assignment validation distinguishes omission, invalid names, and eligibility."""
        benchmark = self._build()
        original = copy.deepcopy(benchmark.items)
        trough_index = next(i for i, item in enumerate(original) if not item["tasks"][TASKS[2]]["eligible"])
        for mutation in ("missing", "null", "unknown", "ineligible"):
            with self.subTest(mutation=mutation):
                items = copy.deepcopy(original)
                if mutation == "missing":
                    items[0].pop("assigned_task")
                elif mutation == "null":
                    items[0]["assigned_task"] = None
                elif mutation == "unknown":
                    items[0]["assigned_task"] = "unknown"
                else:
                    items[trough_index]["assigned_task"] = TASKS[2]
                self._refresh(benchmark, items)
                with self.assertRaises(ValueError):
                    load_and_validate_benchmark(benchmark.manifest_path)

    def test_selection_summary_and_contract_are_strict(self) -> None:
        """Summary counts, JSON types, and mode declarations are all checked."""
        benchmark = self._build()
        for mutation in ("missing", "wrong_count", "boolean", "extra", "legacy"):
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(benchmark.manifest)
                if mutation == "missing":
                    manifest.pop("task_selection")
                elif mutation == "wrong_count":
                    manifest["task_selection"]["selected_final_cases"] += 1
                elif mutation == "boolean":
                    manifest["task_selection"]["selected_by_cohort"]["3tp_gold"][TASKS[0]] = True
                elif mutation == "extra":
                    manifest["task_selection"]["extra"] = 1
                else:
                    manifest["benchmark_version"] = "temporal-reasoning-gold-v1"
                with self.assertRaises(ValueError):
                    validate_task_selection(manifest, benchmark.items)

    def test_registries_select_one_case_per_trajectory(self) -> None:
        """PAIR/SEQ stay complete while final calls and results follow assignments."""
        benchmark = self._build()
        registry = build_registries({"model": "fixture"}, benchmark.manifest, benchmark.items)
        plan = _validate_registries(registry, benchmark.items)
        self.assertEqual(plan["pair_predictions"], 7)
        self.assertEqual(plan["seq_predictions"], 4)
        self.assertEqual(plan["potential_logical_calls_before_retries"], 27)
        self.assertEqual(plan["unreduced_maximum_calls_before_retries"], 29)
        self.assertEqual(len(registry["final_cases"]), 4)
        self.assertTrue(all(len(case["result_ids"]) == 6 for case in registry["final_cases"]))
        registry["final_cases"][0]["task"] = next(
            task for task in TASKS if task != registry["final_cases"][0]["task"]
        )
        with self.assertRaisesRegex(ValueError, "frozen task assignments"):
            _validate_registries(registry, benchmark.items)

    def test_complete_offline_run_uses_selected_denominators(self) -> None:
        """Synthetic completion covers six results per selection and correct metrics."""
        benchmark = self._build()
        run_dir = write_run(self.root, benchmark.items, benchmark_manifest=benchmark.manifest_path)
        run = load_validated_run(run_dir)
        self.assertEqual(run.coverage["expected"]["final_cases"], 4)
        self.assertEqual(run.coverage["expected"]["final_results"], 24)
        summary = summarize_run(run)
        self.assertEqual(summary["metadata"]["schema_version"], 4)
        three = summary["cohorts"]["3tp_gold"]
        for task in TASKS:
            report = three["tasks"][task]
            selected = self.quotas[3][task]
            eligible = 2 if task == TASKS[2] else 3
            self.assertEqual(report["population"]["selected_questions"], selected)
            self.assertEqual(report["population"]["eligible_questions"], eligible)
            self.assertEqual(report["population"]["unselected_eligible_questions"], eligible - selected)
            for setting in SETTINGS:
                accuracy = report["settings"][setting]["final_answer_accuracy"]
                self.assertEqual(accuracy["total"], selected)
                self.assertEqual(accuracy["accuracy"], 1.0 if selected else None)
        for method in ("PAIR", "SEQ"):
            self.assertEqual(three["change_label_metrics"][method]["whole_trajectory_accuracy"]["total"], 3)
        path = run_dir / "responses/seq_dir.jsonl"
        rows = read_jsonl(path)
        extra = dict(rows[0], result_id="unexpected-unselected-result")
        write_jsonl(path, rows + [extra])
        with self.assertRaisesRegex(ValueError, "extra=1"):
            load_validated_run(run_dir)

    def test_offline_validation_accepts_historical_answer_only_prompts(self) -> None:
        """Historical answer-only registries remain valid for offline scoring."""
        benchmark = self._build()
        run_dir = write_run(
            self.root, benchmark.items, benchmark_manifest=benchmark.manifest_path,
            legacy_answer_only=True,
        )
        run = load_validated_run(run_dir)
        self.assertEqual(run.coverage["expected"]["final_results"], 24)
        self.assertEqual(summarize_run(run)["metadata"]["schema_version"], 4)

    def test_old_short_explanation_fixture_remains_valid_for_scoring(self) -> None:
        """Saved explanation-enabled runs retain their original fixture metadata."""
        from experiments.temporal_reasoning_single_finding.protocol.prompt_compatibility import (
            FIXTURE_KEY, LEGACY_FIXTURE_SHA256,
        )

        benchmark = self._build()
        run_dir = write_run(
            self.root, benchmark.items, benchmark_manifest=benchmark.manifest_path,
        )
        path = run_dir / "run_manifest.json"
        manifest = read_json(path)
        manifest["prompt_fixture_checksums"] = {FIXTURE_KEY: LEGACY_FIXTURE_SHA256}
        write_json(path, manifest)
        original = path.read_bytes()
        run = load_validated_run(run_dir)
        self.assertEqual(run.coverage["expected"]["final_results"], 24)
        self.assertEqual(summarize_run(run)["metadata"]["schema_version"], 4)
        self.assertEqual(path.read_bytes(), original)

    def test_preflight_reports_eligibility_separately_from_selection(self) -> None:
        """Preflight shows both denominators without visiting images or an endpoint."""
        benchmark = self._build()
        experiment = self.root / "experiment.json"
        paths = self.root / "paths.json"
        write_json(experiment, {
            "run_name": "gold-pilot-fixture", "model": "fixture-model",
            "base_url": "http://127.0.0.1:1/v1",
            "benchmark_path": str(benchmark.manifest_path),
        })
        write_json(paths, {"chexpert_plus_image_root": str(self.root)})
        target = "experiments.temporal_reasoning_single_finding.validation.preflight"
        with (
            patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "prepare_run.run_dir_for", return_value=self.root / "new-run",
            ),
            patch(target + "._check_image_files", return_value=7),
        ):
            result = preflight(experiment, paths, check_endpoint=False)
        self.assertEqual(result["eligible_final_cases"], 15)
        self.assertEqual(result["selected_final_cases"], 4)
        self.assertEqual(result["call_plan"]["potential_logical_calls_before_retries"], 27)
        self.assertFalse(result["endpoint_checked"])

    def test_offline_validation_rejects_a_different_assignment_with_same_quotas(self) -> None:
        """A quota-correct prepared run must still match the frozen assignments."""
        benchmark = self._build()
        changed = copy.deepcopy(benchmark.items)
        swappable = [
            item for item in changed
            if item["timepoint_count"] == 3 and item["tasks"][TASKS[2]]["eligible"]
        ]
        a, b = swappable
        a["assigned_task"], b["assigned_task"] = b["assigned_task"], a["assigned_task"]
        self.assertEqual(selection_summary(changed), benchmark.manifest["task_selection"])
        for enabled in (False, True):
            with self.subTest(legacy_answer_only=enabled):
                run_dir = write_run(
                    self.root, changed, benchmark_manifest=benchmark.manifest_path,
                    run_name=f"changed-assignment-{int(enabled)}",
                    legacy_answer_only=enabled,
                )
                with self.assertRaisesRegex(ValueError, "conflicts with frozen task assignments"):
                    load_validated_run(run_dir)


if __name__ == "__main__":
    unittest.main()

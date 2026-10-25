"""Independent expected counts for the approved offline evaluator contract."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    read_jsonl,
    repository_root,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.temporal_reasoning_single_finding.evaluation import evaluate
from experiments.temporal_reasoning_single_finding.evaluation.metrics import ratio
from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS
from experiments.temporal_reasoning_single_finding.tests.evaluation.fixtures import (
    item,
    write_run,
)


class EvaluationTests(unittest.TestCase):
    """Exercise complete metadata-only runs without endpoint or image access."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(
            evaluate, "repository_root", return_value=self.root
        )
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.constant = ["constant", "none", "T1-T3", "none"]

    def test_preserves_existing_metrics_from_historical_json_example(self) -> None:
        """Keep historical scores and diagnostics while extending the output."""
        items = [
            item("a", ["deterioration", "improvement"],
                 ["single_peak", "peak:T2", "T2", "T2-T3"]),
            item("b", ["unchanged", "unchanged"], self.constant,
                 finding="pleural_effusion"),
        ]
        write_run(
            self.root, items,
            pair={("b", 1): "invalid_response"},
            seq={"a": ["improvement", "deterioration"]},
            final={
                ("a", TASKS[1], "SEQ-DIR"): "invalid_response",
                ("a", TASKS[2], "SEQ-DIR"): "response_error",
                ("a", TASKS[3], "SEQ-DIR"): "none",
                ("a", TASKS[1], "PAIR-LC"): "invalid_response",
                ("a", TASKS[2], "PAIR-LC"): "T1",
                ("b", TASKS[1], "SEQ-LC"): "response_error",
            },
        )
        expected = read_json(
            repository_root() / "experiments/temporal_reasoning_single_finding"
            / "tests/fixtures/evaluation_output_example.json"
        )
        expected["metadata"]["run_name"] = "fixture-run"
        expected["metadata"]["benchmark_version"] = "fixture-compatible-v1"
        expected["metadata"]["evaluator_version"] = "2.0.0"
        expected["metadata"]["schema_version"] = 4
        for task in expected["cohorts"]["3tp_silver"]["tasks"].values():
            for agreement in task["rule_based_answer_agreement"].values():
                agreement.update(
                    total=2, excluded=0, agreement_rate=agreement["agreements"] / 2,
                )
        actual = evaluate.evaluate_run("fixture-run")
        for task in actual["cohorts"]["3tp_silver"]["tasks"].values():
            for setting, metrics in task["settings"].items():
                metrics.pop("balanced_accuracy")
                if setting in {"PAIR-LC", "SEQ-LC"}:
                    metrics.pop("accuracy_when_fc_correct")
                    metrics.pop("accuracy_when_fc_wrong")
        self.assertEqual(actual, expected)

    def test_reference_supported_policy_scores_and_validates_skipped_cases(self) -> None:
        """The gate preserves denominators and excludes skips from call metrics."""
        run_dir = write_run(
            self.root,
            [item("a", ["deterioration", "deterioration"],
                  ["monotonic_worsening", "none", "T3", "none"])],
            pair={("a", 1): "unchanged"},
            seq={"a": ["deterioration", "unchanged"]},
            lc_inference_policy="reference-supported-answer-v1",
        )
        result = evaluate.evaluate_run("fixture-run")
        self.assertEqual(result["metadata"]["schema_version"], 4)
        self.assertEqual(result["metadata"]["evaluator_version"], "2.0.0")
        self.assertEqual(
            result["metadata"]["lc_inference_policy"],
            "reference-supported-answer-v1",
        )
        cohort = result["cohorts"]["3tp_silver"]
        for source in ("PAIR", "SEQ"):
            setting = f"{source}-LC"
            self.assertEqual(result["direct_failure_rates"][setting]["submitted_logical_calls"], 3)
            severe = cohort["tasks"][TASKS[2]]
            self.assertEqual(severe["settings"][setting]["final_answer_accuracy"], {
                "correct": 0, "total": 1, "accuracy": 0.0,
            })
            failures = severe["settings"][setting]["downstream_failure_rates"]
            self.assertEqual(failures["upstream_answer_mismatch"], {
                "count": 1, "rate": 1.0,
            })
            self.assertEqual(severe["rule_based_answer_agreement"][f"{setting} vs {source}-FC"], {
                "agreements": 0, "total": 1, "excluded": 0, "agreement_rate": 0.0,
            })
            self.assertEqual(severe["settings"][setting]["accuracy_when_fc_wrong"], {
                "correct": 0, "total": 1, "accuracy": 0.0,
            })
            self.assertEqual(severe["settings"][setting]["accuracy_when_fc_correct"], {
                "correct": 0, "total": 0, "accuracy": None,
            })
            shape = cohort["tasks"][TASKS[0]]["settings"][setting]
            self.assertEqual(shape["final_answer_accuracy"]["correct"], 1)
            labels = cohort["change_label_metrics"][source]
            self.assertEqual(labels["whole_trajectory_accuracy"]["correct"], 0)

        path = run_dir / "responses/pair_lc.jsonl"
        original = read_jsonl(path)
        for field, value in (
            ("correct", True), ("predicted_answer", "T3"),
            ("model_input_id", "fabricated"), ("status", "valid"),
            ("consumed_change_labels", ["deterioration", "deterioration"]),
        ):
            with self.subTest(field=field):
                rows = [dict(row) for row in original]
                row = next(row for row in rows if row["task"] == TASKS[2])
                row[field] = value
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    evaluate.evaluate_run("fixture-run")
        write_jsonl(path, original[:-1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate.evaluate_run("fixture-run")

    def test_reused_pair_counts_per_trajectory_but_call_counts_once(self) -> None:
        images = ["shared/1.jpg", "shared/2.jpg", "shared/3.jpg", "shared/4.jpg"]
        write_run(self.root, [
            item("a", ["unchanged"] * 2, self.constant,
                 patient="same-patient", image_ids=images[:3]),
            item("b", ["unchanged"] * 2, self.constant,
                 patient="same-patient", image_ids=images[1:]),
            item("c", ["unchanged"] * 2, self.constant),
            item("d", ["unchanged"] * 3, ["constant", "none", "T1-T4", "none"],
                 patient="same-patient", image_ids=images),
        ], pair={("a", 1): "invalid_response"})
        result = evaluate.evaluate_run("fixture-run")
        cohort = result["cohorts"]["3tp_silver"]
        self.assertEqual(cohort["population"]["patients"], 2)
        self.assertEqual(cohort["population"]["trajectories"], 3)
        pair = cohort["change_label_metrics"]["PAIR"]
        self.assertEqual(pair["interval_accuracy"], {
            "correct": 4, "total": 6, "value": 2 / 3
        })
        self.assertEqual(pair["whole_trajectory_accuracy"]["value"], 1 / 3)
        for task in cohort["tasks"].values():
            score = task["settings"]["PAIR-FC"]["final_answer_accuracy"]
            self.assertEqual(score, {"correct": 1, "total": 3, "accuracy": 1 / 3})
        self.assertEqual(result["direct_failure_rates"]["PAIR"], {
            "submitted_logical_calls": 5,
            "invalid_response": {"count": 1, "rate": 0.2},
            "technical_failure": {"count": 0, "rate": 0.0},
        })
        four = result["cohorts"]["4tp_silver"]["change_label_metrics"]["PAIR"]
        self.assertEqual(four["interval_accuracy"]["value"], 2 / 3)
        self.assertEqual(four["interval_error_counts"]["1"]["trajectories"], 1)

    def test_four_cohorts_and_failed_seq_vectors(self) -> None:
        items = [
            item(f"{count}-{source}", ["unchanged"] * (count - 1),
                 ["constant", "none", f"T1-T{count}", "none"], source=source)
            for count in (3, 4) for source in ("silver", "gold")
        ]
        write_run(self.root, items, seq={
            "3-gold": "invalid_response", "4-gold": "response_error"
        }, pair={("4-silver", 2): "response_error"})
        result = evaluate.evaluate_run("fixture-run")
        self.assertEqual(result["metadata"]["cohorts"], [
            "3tp_silver", "3tp_gold", "4tp_silver", "4tp_gold"
        ])
        for name, errors in (("3tp_gold", 2), ("4tp_gold", 3)):
            seq = result["cohorts"][name]["change_label_metrics"]["SEQ"]
            self.assertEqual(seq["interval_accuracy"], {
                "correct": 0, "total": errors, "value": 0.0
            })
            self.assertEqual(seq["interval_error_counts"][str(errors)], {
                "trajectories": 1, "proportion": 1.0
            })
            self.assertEqual(seq["per_class"]["unchanged"]["counts"], {
                "true_positives": 0, "false_positives": 0, "false_negatives": errors
            })
        four = result["cohorts"]["4tp_silver"]["change_label_metrics"]["PAIR"]
        self.assertEqual(four["interval_accuracy"]["value"], 2 / 3)
        self.assertEqual(set(four["interval_error_counts"]), {"0", "1", "2", "3"})
        self.assertEqual(result["cohorts"]["3tp_silver"]["change_label_metrics"]["SEQ"]
                         ["whole_trajectory_accuracy"]["value"], 1.0)

    def test_failed_dependencies_count_once_with_technical_precedence(self) -> None:
        write_run(self.root, [item("a", ["unchanged"] * 2, self.constant)], pair={
            ("a", 0): "invalid_response", ("a", 1): "response_error"
        })
        result = evaluate.evaluate_run("fixture-run")
        cohort = result["cohorts"]["3tp_silver"]
        for task in cohort["tasks"].values():
            for setting in ("PAIR-FC", "PAIR-LC"):
                score = task["settings"][setting]
                self.assertEqual(score["final_answer_accuracy"]["accuracy"], 0.0)
                self.assertEqual(score["downstream_failure_rates"], {
                    "eligible_questions": 1,
                    "invalid_response": {"count": 0, "rate": 0.0},
                    "technical_failure": {"count": 1, "rate": 1.0},
                })
            agreement = task["rule_based_answer_agreement"]["PAIR-LC vs PAIR-FC"]
            self.assertEqual(agreement, {
                "agreements": 0, "total": 1, "excluded": 0, "agreement_rate": 0.0
            })
        self.assertEqual(result["direct_failure_rates"]["PAIR-LC"], {
            "submitted_logical_calls": 0,
            "invalid_response": {"count": 0, "rate": None},
            "technical_failure": {"count": 0, "rate": None},
        })

    def test_reference_ineligible_question_and_null_denominators(self) -> None:
        write_run(self.root, [item("a", ["improvement", "deterioration"],
                                  ["single_trough", "trough:T2", None, "T1-T2"])],
                  pair={("a", 0): "deterioration"},
                  seq={"a": ["unchanged", "deterioration"]})
        result = evaluate.evaluate_run("fixture-run")
        cohort = result["cohorts"]["3tp_silver"]
        severe = cohort["tasks"][TASKS[2]]
        self.assertEqual(severe["population"]["eligible_questions"], 0)
        self.assertEqual(severe["population"]["excluded_questions"], 1)
        for entry in severe["population"]["reference_answers"].values():
            self.assertEqual(entry, {"questions": 0, "proportion": None})
        for setting in severe["settings"].values():
            self.assertEqual(setting["final_answer_accuracy"], {
                "correct": 0, "total": 0, "accuracy": None
            })
        pair = cohort["change_label_metrics"]["PAIR"]
        self.assertEqual(pair["interval_accuracy"]["total"], 2)
        self.assertEqual(pair["per_class"]["improvement"]["metrics"], {
            "precision": None, "recall": 0.0, "f1": 0.0
        })
        self.assertEqual(pair["per_class"]["unchanged"]["metrics"], {
            "precision": None, "recall": None, "f1": None
        })
        self.assertEqual(pair["per_class"]["deterioration"]["metrics"], {
            "precision": 0.5, "recall": 1.0, "f1": 2 / 3
        })
        seq_class = cohort["change_label_metrics"]["SEQ"]["per_class"]["unchanged"]
        self.assertEqual(seq_class["metrics"], {
            "precision": 0.0, "recall": None, "f1": 0.0
        })
        self.assertIsNone(ratio(0, 0))
        self.assertEqual(ratio(0, 5), 0.0)

    def test_offline_cli_is_deterministic_and_preserves_other_files(self) -> None:
        selected = write_run(self.root, [item("a", ["unchanged"] * 2, self.constant)])
        other = write_run(self.root, [item("b", ["unchanged"] * 2, self.constant)],
                          run_name="other-run")
        before = {p: sha256_file(p) for p in self.root.rglob("*") if p.is_file()}
        with (
            patch.dict(os.environ, {}, clear=True),
            redirect_stdout(io.StringIO()) as output,
        ):
            evaluate.main(["--run-name", "fixture-run"])
        self.assertIn(str(selected / "evaluation.json"), output.getvalue())
        self.assertFalse((other / "evaluation.json").exists())
        self.assertFalse((selected / "validation/coverage.json").exists())
        first = (selected / "evaluation.json").read_text()
        expected = read_json(selected / "evaluation.json")
        self.assertEqual(evaluate.evaluate_run("fixture-run"), expected)
        self.assertEqual((selected / "evaluation.json").read_text(), first)
        after = {p: sha256_file(p) for p in before}
        self.assertEqual(before, after)
        self.assertEqual({p for p in self.root.rglob("*") if p.is_file()} - set(before),
                         {selected / "evaluation.json"})
        self.assertIn(
            '"interval_accuracy": {"correct": 2, "total": 2, "value": 1.0}', first
        )
        self.assertEqual(list(expected), [
            "metadata", "direct_failure_rates", "cohorts"
        ])

    def test_missing_results_stop_without_overwriting_existing_evaluation(self) -> None:
        run_dir = write_run(self.root, [item("a", ["unchanged"] * 2, self.constant)])
        evaluate.evaluate_run("fixture-run")
        before = sha256_file(run_dir / "evaluation.json")
        path = run_dir / "responses/seq_lc.jsonl"
        write_jsonl(path, read_jsonl(path)[:-1])
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            evaluate.evaluate_run("fixture-run")
        self.assertEqual(sha256_file(run_dir / "evaluation.json"), before)

    def test_relocated_legacy_run_uses_saved_relative_path_and_checksum(self) -> None:
        original = self.root / "original"
        run_dir = write_run(
            original, [item("a", ["unchanged"] * 2, self.constant)]
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest = read_json(manifest_path)
        self.assertNotIn("benchmark_manifest_path", manifest)
        manifest["benchmark_manifest_path"] = str(
            original / "fixture-run-benchmark.json"
        )
        manifest["experiment"]["benchmark_path"] = "fixture-run-benchmark.json"
        write_json(manifest_path, manifest)
        saved_digest = sha256_file(manifest_path)
        relative_run = run_dir.relative_to(original)
        moved = self.root / "moved"
        original.rename(moved)
        with (
            patch.object(evaluate, "repository_root", return_value=moved),
            patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "experiment_config.repository_root", return_value=moved,
            ),
        ):
            result = evaluate.evaluate_run("fixture-run")
            self.assertEqual(result["metadata"]["run_name"], "fixture-run")
            self.assertEqual(
                sha256_file(moved / relative_run / "run_manifest.json"), saved_digest
            )
            summary_path = moved / relative_run / "evaluation.json"
            summary_digest = sha256_file(summary_path)
            write_json(moved / "fixture-run-benchmark.json", {"different": "benchmark"})
            with self.assertRaisesRegex(ValueError, "checksum no longer matches"):
                evaluate.evaluate_run("fixture-run")
            self.assertEqual(sha256_file(summary_path), summary_digest)

    def test_missing_prediction_stops_without_creating_evaluation(self) -> None:
        run_dir = write_run(self.root, [item("a", ["unchanged"] * 2, self.constant)])
        path = run_dir / "responses/pair.jsonl"
        write_jsonl(path, read_jsonl(path)[:-1])
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            evaluate.evaluate_run("fixture-run")
        self.assertFalse((run_dir / "evaluation.json").exists())

    def test_cli_requires_explicit_existing_safe_run_name(self) -> None:
        for name in ("../other", "/tmp/run", "RUN", " padded"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "run_name"),
            ):
                evaluate.evaluate_run(name)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing_argument:
                evaluate.main([])
            self.assertEqual(missing_argument.exception.code, 2)
            with self.assertRaises(SystemExit) as missing_run:
                evaluate.main(["--run-name", "absent"])
            self.assertEqual(missing_run.exception.code, 1)


if __name__ == "__main__":
    unittest.main()

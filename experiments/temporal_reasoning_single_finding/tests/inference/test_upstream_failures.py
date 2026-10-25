"""Tests for fixed-code composition and upstream failure propagation."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    LC_INFERENCE_POLICY,
)

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    build_fixed_code_records,
)
from experiments.temporal_reasoning_single_finding.inference.run_lc import (
    build_predicted_lc_specs,
    run_predicted_lc,
)


def _case(task: str = "trajectory_shape") -> dict:
    """Create one prepared final case."""
    return {
        "final_case_id": "case",
        "result_ids": {
            "PAIR-FC": "pair-result",
            "SEQ-FC": "seq-result",
            "PAIR-LC": "pair-lc-result",
            "SEQ-LC": "seq-lc-result",
        },
        "benchmark_version": "benchmark",
        "trajectory_id": "chexpert_plus:sequence",
        "dataset": "chexpert_plus",
        "annotation_source": "silver",
        "patient_id": "patient",
        "timepoint_count": 4,
        "target_finding": {"canonical_name": "edema", "display_name": "edema"},
        "task": task,
        "reference_answer": "single_peak" if task == "trajectory_shape" else "T2",
        "reference_change_labels": ["deterioration", "unchanged", "improvement"],
        "pair_dependency_ids": ["p1", "p2", "p3"],
        "seq_dependency_id": "seq",
    }


class UpstreamFailureTests(unittest.TestCase):
    """Check exact fixed-code status and dependency semantics."""

    def test_valid_fixed_code_consumes_exact_labels(self) -> None:
        """PAIR-FC and SEQ-FC derive the same valid answer from saved labels."""
        labels = ["deterioration", "unchanged", "improvement"]
        pair = {
            f"p{index}": {"status": "valid", "predicted_change_label": label}
            for index, label in enumerate(labels, start=1)
        }
        seq = {"seq": {"status": "valid", "predicted_change_labels": labels}}
        pair_rows, seq_rows = build_fixed_code_records([_case()], pair, seq)
        for row in pair_rows + seq_rows:
            self.assertEqual(row["status"], "valid")
            self.assertEqual(row["predicted_answer"], "single_peak")
            self.assertEqual(row["consumed_change_labels"], labels)
            self.assertTrue(row["correct"])

    def test_response_error_takes_propagation_precedence(self) -> None:
        """Any technical dependency error yields upstream_response_error."""
        pair = {
            "p1": {"status": "invalid_response"},
            "p2": {"status": "response_error"},
            "p3": {"status": "valid", "predicted_change_label": "unchanged"},
        }
        seq = {"seq": {"status": "response_error"}}
        pair_rows, seq_rows = build_fixed_code_records([_case()], pair, seq)
        self.assertEqual(pair_rows[0]["status"], "upstream_response_error")
        self.assertEqual(len(pair_rows[0]["dependency_statuses"]), 2)
        self.assertEqual(seq_rows[0]["status"], "upstream_response_error")

    def test_no_identifiable_predicted_maximum(self) -> None:
        """A valid trough vector receives the dedicated no-answer status."""
        labels = ["improvement", "deterioration", "unchanged"]
        pair = {
            f"p{index}": {"status": "valid", "predicted_change_label": label}
            for index, label in enumerate(labels, start=1)
        }
        seq = {"seq": {"status": "valid", "predicted_change_labels": labels}}
        pair_rows, seq_rows = build_fixed_code_records(
            [_case("most_severe_visit_or_plateau")], pair, seq
        )
        for row in pair_rows + seq_rows:
            self.assertEqual(
                row["status"], "no_identifiable_most_severe_visit_or_plateau"
            )
            self.assertIsNone(row["predicted_answer"])
            self.assertFalse(row["correct"])
            self.assertIsNone(row["failure_reason"])

    def test_fc_to_lc_requires_exact_saved_dependencies_and_labels(self) -> None:
        """PAIR-LC fails before submission when FC labels differ from predictions."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "derived").mkdir()
            (run_dir / "responses").mkdir()
            fc = {
                "result_id": "pair-result",
                "final_case_id": "case",
                "status": "valid",
                "dependency_ids": ["p1", "p2", "p3"],
                "dependency_statuses": [],
                "consumed_change_labels": [
                    "deterioration",
                    "deterioration",
                    "improvement",
                ],
            }
            write_json(run_dir / "derived" / "pair_fc.json", [fc])
            write_jsonl(
                run_dir / "responses" / "pair.jsonl",
                [
                    {
                        "prediction_id": "p1",
                        "status": "valid",
                        "predicted_change_label": "deterioration",
                    },
                    {
                        "prediction_id": "p2",
                        "status": "valid",
                        "predicted_change_label": "unchanged",
                    },
                    {
                        "prediction_id": "p3",
                        "status": "valid",
                        "predicted_change_label": "improvement",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "consumed labels conflict"):
                build_predicted_lc_specs(run_dir, [_case()], "model", "PAIR")

    def test_identical_pair_and_seq_lc_inputs_remain_separate(self) -> None:
        """Equal LC text shares a fingerprint but retains setting result IDs."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "derived").mkdir()
            (run_dir / "responses").mkdir()
            labels = ["deterioration", "unchanged", "improvement"]
            for name, result, dependencies in (
                ("pair_fc.json", "pair-result", ["p1", "p2", "p3"]),
                ("seq_fc.json", "seq-result", ["seq"]),
            ):
                write_json(
                    run_dir / "derived" / name,
                    [{
                        "result_id": result,
                        "final_case_id": "case",
                        "status": "valid",
                        "dependency_ids": dependencies,
                        "dependency_statuses": [],
                        "consumed_change_labels": labels,
                        "predicted_answer": "single_peak",
                    }],
                )
            write_jsonl(
                run_dir / "responses" / "pair.jsonl",
                [
                    {
                        "prediction_id": f"p{index}",
                        "status": "valid",
                        "predicted_change_label": label,
                    }
                    for index, label in enumerate(labels, start=1)
                ],
            )
            write_jsonl(
                run_dir / "responses" / "seq.jsonl",
                [
                    {
                        "prediction_id": "seq",
                        "status": "valid",
                        "predicted_change_labels": labels,
                    }
                ],
            )
            pair_specs, pair_inputs = build_predicted_lc_specs(
                run_dir, [_case()], "model", "PAIR"
            )
            seq_specs, seq_inputs = build_predicted_lc_specs(
                run_dir, [_case()], "model", "SEQ"
            )
            self.assertEqual(
                pair_inputs[0]["model_input_id"],
                seq_inputs[0]["model_input_id"],
            )
            self.assertNotEqual(pair_specs[0]["result_id"], seq_specs[0]["result_id"])

    def test_propagated_lc_makes_no_model_or_discovery_call(self) -> None:
        """A wholly failed upstream stage writes LC results without an endpoint."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in ("prepared", "derived", "responses"):
                (run_dir / name).mkdir()
            case = _case()
            write_json(run_dir / "prepared" / "final_cases.json", [case])
            write_jsonl(run_dir / "prepared" / "model_inputs.jsonl", [])
            write_json(
                run_dir / "prepared" / "manifest.json",
                {"counts": {}, "files": {"model_inputs.jsonl": "placeholder"}},
            )
            write_json(
                run_dir / "derived" / "pair_fc.json",
                [{
                    "result_id": "pair-result",
                    "final_case_id": "case",
                    "status": "upstream_invalid",
                    "dependency_ids": ["p1", "p2", "p3"],
                    "dependency_statuses": [
                        {
                            "dependency_id": "p1",
                            "status": "invalid_response",
                        }
                    ],
                    "consumed_change_labels": None,
                }],
            )
            write_jsonl(
                run_dir / "responses" / "pair.jsonl",
                [{"prediction_id": "p1", "status": "invalid_response"}],
            )
            experiment = {
                "model": "model", "base_url": "http://unreachable/v1",
                "run_name": "test", "benchmark_path": "unused.json",
            }
            write_json(run_dir / "run_manifest.json", {
                "experiment": experiment, "lc_inference_policy": LC_INFERENCE_POLICY,
            })
            summary = run_predicted_lc(
                run_dir,
                experiment,
                "PAIR",
                client=None,
            )
            self.assertEqual(summary["submitted"], 0)
            self.assertEqual(summary["propagated"], 1)
            row = read_jsonl(run_dir / "responses" / "pair_lc.jsonl")[0]
            self.assertEqual(row["status"], "upstream_invalid")
            self.assertIsNone(row["model_input_id"])

    def test_wrong_fc_answer_is_submitted_once_and_reused_on_resume(self) -> None:
        """Usable labels trigger LC independently of task-specific FC correctness."""
        labels = ["deterioration", "deterioration", "improvement"]
        shape = _case()
        turning = _case("turning_regions")
        turning["reference_answer"] = "peak:T2-T3"
        turning["final_case_id"] = "turning-case"
        turning["result_ids"] = {
            key: value + "-turning" for key, value in turning["result_ids"].items()
        }
        pair = {
            f"p{i}": {"prediction_id": f"p{i}", "status": "valid",
                       "predicted_change_label": label}
            for i, label in enumerate(labels, 1)
        }
        seq = {"seq": {"prediction_id": "seq", "status": "valid",
                       "predicted_change_labels": labels}}
        for source in ("PAIR", "SEQ"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pair_fc, seq_fc = build_fixed_code_records([shape, turning], pair, seq)
                write_json(root / "derived/pair_fc.json", pair_fc)
                write_json(root / "derived/seq_fc.json", seq_fc)
                write_jsonl(root / "responses/pair.jsonl", list(pair.values()))
                write_jsonl(root / "responses/seq.jsonl", list(seq.values()))
                specs, inputs = build_predicted_lc_specs(root, [shape, turning], "model", source)
                self.assertEqual(len(inputs), 2)
                self.assertTrue(all("precomputed_record" not in spec for spec in specs))
                self.assertEqual(specs[1]["consumed_change_labels"], labels)
                write_json(root / "prepared/final_cases.json", [turning])
                write_jsonl(root / "prepared/model_inputs.jsonl", [])
                write_json(root / "prepared/manifest.json", {"counts": {}, "files": {}})
                experiment = {
                    "model": "model", "base_url": "http://unreachable/v1",
                    "run_name": "test", "benchmark_path": "unused.json",
                }
                write_json(root / "run_manifest.json", {
                    "experiment": experiment, "lc_inference_policy": LC_INFERENCE_POLICY,
                })
                with (
                    patch(
                        "experiments.temporal_reasoning_single_finding.inference.run_lc._client",
                        return_value=object(),
                    ),
                    patch(
                        "experiments.temporal_reasoning_single_finding.inference."
                        "run_change_labels.submit_messages",
                        return_value={"kind": "text", "error": None,
                                      "raw_text": '{"answer":"peak:T2-T3"}'},
                    ) as submit,
                ):
                    first = run_predicted_lc(root, experiment, source)
                    second = run_predicted_lc(root, experiment, source)
                self.assertEqual(first["submitted"], 1)
                self.assertEqual(first["propagated"], 0)
                self.assertEqual(second["submitted"], 0)
                self.assertEqual(second["reused"], 1)
                submit.assert_called_once()
                self.assertIsNone(submit.call_args.args[3])
                response_path = root / f"responses/{source.lower()}_lc.jsonl"
                rows = read_jsonl(response_path)
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["correct"])

    def test_legacy_run_cannot_silently_change_lc_policy(self) -> None:
        """A saved legacy run cannot silently switch to the default LC policy."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for policy in (None, "reference-supported-answer-v1"):
                manifest = {"experiment": {}}
                if policy is not None:
                    manifest["lc_inference_policy"] = policy
                write_json(root / "run_manifest.json", manifest)
                with self.subTest(policy=policy), self.assertRaisesRegex(
                    ValueError, "Prepare a new run name",
                ):
                    run_predicted_lc(root, {}, "PAIR")


if __name__ == "__main__":
    unittest.main()

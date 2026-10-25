"""Validate default LC eligibility and ordinary pipeline reuse across policies."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ...artifact_io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from ...evaluation.metrics import summarize_run
from ...inference.run_lc import _propagated_lc_record, build_predicted_lc_specs
from ...inference.run_pipeline import run_pipeline
from ...modal.runtime import require_run_provenance
from ...protocol.contracts import (
    LC_INFERENCE_POLICY, NO_IDENTIFIABLE, REFERENCE_SUPPORTED_LC_POLICY,
)
from ...validation.preflight import ValidatedBenchmark
from ...validation.validate_run import load_validated_run
from ..evaluation.fixtures import item, write_run


PACKAGE = "experiments.temporal_reasoning_single_finding"
SEVERE = "most_severe_visit_or_plateau"


class LCPolicyTests(unittest.TestCase):
    """Keep historical runs readable and resume new-policy calls without images."""

    def setUp(self) -> None:
        """Create synthetic references for wrong, undefined, and failed FC cases."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.items = [
            item(name, ["deterioration", "deterioration"],
                 ["monotonic_worsening", "none", "T3", "none"])
            for name in ("wrong", "undefined", "failed")
        ]

    def make_run(self, policy: str | None) -> Path:
        """Write completed responses under the selected historical or new policy."""
        return write_run(
            self.root, self.items,
            run_name=policy or "legacy",
            pair={("wrong", 1): "unchanged", ("undefined", 0): "improvement",
                  ("failed", 0): "invalid_response"},
            seq={"wrong": ["deterioration", "unchanged"],
                 "undefined": ["improvement", "deterioration"],
                 "failed": "response_error"},
            final={(name, SEVERE, setting): "T3"
                   for name in ("wrong", "undefined")
                   for setting in ("PAIR-LC", "SEQ-LC")},
            lc_inference_policy=policy,
        )

    def test_policy_controls_validation_and_evaluation_preserves_identity(self) -> None:
        """Historical skips validate unchanged while new LC may recover wrong FC."""
        for policy, correct in ((None, 1), (REFERENCE_SUPPORTED_LC_POLICY, 0),
                                (LC_INFERENCE_POLICY, 2)):
            with self.subTest(policy=policy):
                run_dir = self.make_run(policy)
                before = {p: sha256_file(p) for p in run_dir.rglob("*") if p.is_file()}
                run = load_validated_run(run_dir)
                summary = summarize_run(run)
                self.assertEqual(summary["metadata"].get("lc_inference_policy"), policy)
                severe = summary["cohorts"]["3tp_silver"]["tasks"][SEVERE]
                for source in ("PAIR", "SEQ"):
                    self.assertEqual(
                        severe["settings"][f"{source}-LC"]["accuracy_when_fc_wrong"],
                        {"correct": correct, "total": 3, "accuracy": correct / 3},
                    )
                self.assertEqual(before, {p: sha256_file(p) for p in before})

    def test_new_policy_rejects_old_skips_and_unknown_policy(self) -> None:
        """Changing a manifest alone cannot turn old skips into completed LC calls."""
        run_dir = self.make_run(LC_INFERENCE_POLICY)
        run = load_validated_run(run_dir)
        cases = [case for case in run.prepared["final_cases"] if case["task"] == SEVERE]
        path = run_dir / "responses/pair_lc.jsonl"
        original = read_jsonl(path)
        for case in cases:
            fc = run.results["PAIR-FC"][case["result_ids"]["PAIR-FC"]]
            if fc["status"] == "upstream_invalid":
                continue
            with self.subTest(fc_status=fc["status"]):
                skipped = _propagated_lc_record(
                    case, "PAIR-LC", fc, "fixture-model",
                    status=NO_IDENTIFIABLE if fc["status"] == NO_IDENTIFIABLE
                    else "upstream_answer_mismatch",
                )
                write_jsonl(path, [skipped if r["result_id"] == skipped["result_id"]
                                  else r for r in original])
                with self.assertRaises(ValueError):
                    load_validated_run(run_dir)
        write_jsonl(path, original)
        manifest = read_json(run_dir / "run_manifest.json")
        manifest["lc_inference_policy"] = "unknown-policy"
        write_json(run_dir / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Unsupported LC inference policy"):
            load_validated_run(run_dir)

    def test_undefined_fc_still_requires_exact_usable_labels(self) -> None:
        """Undefined FC answers do not bypass canonical upstream input checks."""
        run_dir = self.make_run(LC_INFERENCE_POLICY)
        cases = read_json(run_dir / "prepared/final_cases.json")
        for source in ("PAIR", "SEQ"):
            path = run_dir / f"derived/{source.lower()}_fc.json"
            rows = read_json(path)
            row = next(r for r in rows if r["status"] == NO_IDENTIFIABLE)
            row["consumed_change_labels"] = ["unchanged", "unchanged"]
            write_json(path, rows)
            with self.subTest(source=source), self.assertRaisesRegex(
                ValueError, "consumed labels conflict",
            ):
                build_predicted_lc_specs(run_dir, cases, "fixture-model", source)

    def test_full_pipeline_resumes_only_missing_lc_after_interruption(self) -> None:
        """The ordinary pipeline reuses image stages and submits only pending LC."""
        run_dir = self.make_run(LC_INFERENCE_POLICY)
        for source in ("pair", "seq"):
            path = run_dir / f"responses/{source}_lc.jsonl"
            write_jsonl(path, [row for row in read_jsonl(path)
                              if row["task"] != SEVERE or row["model_input_id"] is None])
        manifest = read_json(run_dir / "run_manifest.json")
        experiment = manifest["experiment"]
        benchmark_path = Path(experiment["benchmark_path"])
        benchmark = ValidatedBenchmark(
            benchmark_path, read_json(benchmark_path), {3: self.items}, self.items,
        )
        experiment_path, paths_path = self.root / "experiment.json", self.root / "paths.json"
        image_root = self.root / "empty-image-root"
        image_root.mkdir()
        write_json(experiment_path, experiment)
        write_json(paths_path, {"chexpert_plus_image_root": str(image_root)})
        preserved = [run_dir / f"responses/{name}.jsonl"
                     for name in ("pair", "seq", "seq_dir", "ref_lc")]
        preserved.append(run_dir / "run_manifest.json")
        before = {p: sha256_file(p) for p in preserved}
        client = SimpleNamespace(models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="fixture-model")]),
        ))
        response = {"kind": "text", "error": None, "raw_text": '{"answer":"T3"}'}
        with (
            patch(f"{PACKAGE}.inference.run_pipeline.preflight", return_value={}),
            patch(f"{PACKAGE}.inference.run_pipeline.load_and_validate_benchmark",
                  return_value=benchmark),
            patch(f"{PACKAGE}.validation.preflight.load_and_validate_benchmark",
                  return_value=benchmark),
            patch(f"{PACKAGE}.inference.prepare_run.repository_root", return_value=self.root),
            patch(f"{PACKAGE}.inference.run_pipeline.create_client", return_value=client),
            patch(f"{PACKAGE}.inference.run_pipeline.require_api_key", return_value="fixture"),
            patch(f"{PACKAGE}.inference.run_change_labels.submit_messages",
                  side_effect=[response, response, RuntimeError("interrupted")]) as submit,
        ):
            metadata = run_dir.parent / ".modal" / run_dir.name
            require_run_provenance(run_dir, metadata, {"experiment": experiment})
            self.assertEqual(read_json(metadata / "import.json")["origin"], "external")
            submit.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_pipeline(experiment_path, paths_path, workers=1)
            self.assertEqual(submit.call_count, 3)
            for call in submit.call_args_list:
                self.assertIsNone(call.args[3])
                self.assertTrue(all(m["role"] in {"system", "user"} for m in call.args[2]))
            submit.reset_mock(side_effect=True)
            submit.return_value = response
            result = run_pipeline(experiment_path, paths_path, workers=1)
            self.assertEqual(submit.call_count, 2)
            self.assertTrue(all(call.args[3] is None for call in submit.call_args_list))
            self.assertEqual(result["stages"]["pair_lc"]["submitted"], 0)
            self.assertEqual(result["stages"]["seq_lc"]["submitted"], 2)
            submit.reset_mock()
            run_pipeline(experiment_path, paths_path, workers=1)
            submit.assert_not_called()
        self.assertEqual(before, {p: sha256_file(p) for p in preserved})
        self.assertEqual(load_validated_run(run_dir).coverage["status"], "passed")

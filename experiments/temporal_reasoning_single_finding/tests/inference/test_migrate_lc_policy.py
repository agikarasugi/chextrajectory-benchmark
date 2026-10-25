"""Offline regression tests for copying legacy runs into gated LC inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json, read_jsonl, sha256_file, write_json, write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    derive_fixed_code,
)
from experiments.temporal_reasoning_single_finding.inference.migrate_lc_policy import (
    _snapshot, migrate_lc_policy,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import (
    prepare_run,
)
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
    _pair_spec, _record_from_spec, _response_fields, _seq_spec, execute_specs,
)
from experiments.temporal_reasoning_single_finding.inference.run_lc import (
    _append_model_inputs, build_predicted_lc_specs, build_ref_lc_specs,
)
from experiments.temporal_reasoning_single_finding.inference.run_seq_dir import (
    build_seq_dir_specs,
)
from experiments.temporal_reasoning_single_finding.tests.inference.test_resume import (
    _item, _manifest,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    ValidatedBenchmark,
)
from experiments.temporal_reasoning_single_finding.validation.validate_run import (
    load_validated_run,
)


_MODULE = "experiments.temporal_reasoning_single_finding.inference.migrate_lc_policy"
_INFERENCE = "experiments.temporal_reasoning_single_finding.inference"


def _response(spec: dict, output: dict) -> dict:
    """Create a synthetic saved answer using the real parser and finalizer."""
    if "precomputed_record" in spec:
        return spec["precomputed_record"]
    row = _record_from_spec(spec)
    row.update(_response_fields(
        {"kind": "text", "raw_text": json.dumps(output), "error": None},
        spec["response_type"],
        timepoint_count=spec.get("timepoint_count_for_parser"),
        task=spec.get("task_for_parser"),
    ))
    row["usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    spec["finalize"](row)
    return row


class MigrateLCPolicyTests(unittest.TestCase):
    """Preserve legacy evidence and the historical gated migration contract."""

    def setUp(self) -> None:
        """Build a small synthetic legacy run with mixed supported LC cases."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest_path = self.root / "benchmark.json"
        manifest = _manifest()
        write_json(manifest_path, manifest)
        items = [_item("one", 4, ["improvement"] * 3)]
        self.benchmark = ValidatedBenchmark(manifest_path, manifest, {4: items}, items)
        self.experiment = {
            "run_name": "legacy", "model": "model", "base_url": "http://unused/v1",
            "benchmark_path": str(manifest_path),
        }
        with patch(
            f"{_INFERENCE}.prepare_run.repository_root",
            return_value=self.root,
        ):
            result = prepare_run(self.experiment, manifest_path, manifest, items)
        self.source = result["run_dir"]
        saved_manifest = read_json(self.source / "run_manifest.json")
        saved_manifest.pop("lc_inference_policy")
        write_json(self.source / "run_manifest.json", saved_manifest)
        predictions = ["improvement", "deterioration", "improvement"]
        pair_inputs = read_json(self.source / "prepared/pair_inputs.json")
        pair_rows = []
        for value in pair_inputs:
            label = (
                "deterioration" if "study2/" in value["earlier_image_id"]
                else "improvement"
            )
            pair_rows.append(_response(_pair_spec(value), {"change_label": label}))
        seq_rows = [
            _response(_seq_spec(value), {"change_labels": {
                "T1_to_T2": predictions[0], "T2_to_T3": predictions[1],
                "T3_to_T4": predictions[2],
            }})
            for value in read_json(self.source / "prepared/seq_inputs.json")
        ]
        write_jsonl(self.source / "responses/pair.jsonl", pair_rows)
        write_jsonl(self.source / "responses/seq.jsonl", seq_rows)
        derive_fixed_code(self.source)
        self.cases = read_json(self.source / "prepared/final_cases.json")
        for name, builder in (
            ("seq_dir", build_seq_dir_specs), ("ref_lc", build_ref_lc_specs),
        ):
            specs = builder(self.source, self.cases, "model")
            write_jsonl(self.source / f"responses/{name}.jsonl", [
                _response(spec, {"answer": spec["reference_answer"]}) for spec in specs
            ])
        for label_source in ("PAIR", "SEQ"):
            specs, inputs = build_predicted_lc_specs(
                self.source, self.cases, "model", label_source,
                lc_inference_policy=None,
            )
            _append_model_inputs(self.source, inputs)
            write_jsonl(self.source / f"responses/{label_source.lower()}_lc.jsonl", [
                _response(spec, {"answer": spec["reference_answer"]}) for spec in specs
                if label_source == "PAIR" or spec["task"] not in {
                    "trajectory_shape", "first_improvement_between_visits",
                }
            ])
        write_json(self.source / "validation/coverage.json", {"stale": True})
        self.loader = patch(
            f"{_MODULE}.load_and_validate_benchmark", return_value=self.benchmark,
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def _hashes(self) -> dict[str, str]:
        """Fingerprint the synthetic source tree to detect any migration writes."""
        return {str(p.relative_to(self.source)): sha256_file(p)
                for p in self.source.rglob("*") if p.is_file()}

    def test_partial_migration_preserves_source_and_resumes(self) -> None:
        """Old answers stay in the source and only one supported call remains."""
        before = self._hashes()
        with patch(
            f"{_INFERENCE}.run_change_labels.submit_messages",
            side_effect=AssertionError("Migration must never submit a request"),
        ):
            report = migrate_lc_policy(self.source, "gated")
        self.assertEqual(before, self._hashes())
        destination = Path(report["run_dir"])
        self.assertEqual(report["pending_model_calls"], 1)
        self.assertEqual(report["stages"]["pair_lc"]["replaced_model_responses"], 2)
        self.assertEqual(report["stages"]["seq_lc"]["replaced_model_responses"], 1)
        self.assertEqual(report["stages"]["seq_lc"]["new_skips"], 1)
        old_pair = read_jsonl(self.source / "responses/pair_lc.jsonl")
        new_pair = read_jsonl(destination / "responses/pair_lc.jsonl")
        retained = next(
            row for row in new_pair
            if row["task"] == "first_improvement_between_visits"
        )
        self.assertEqual(retained, next(
            row for row in old_pair if row["result_id"] == retained["result_id"]
        ))
        self.assertEqual(retained["usage"]["total_tokens"], 15)
        for name in ("pair", "seq", "seq_dir", "ref_lc"):
            self.assertEqual(
                sha256_file(self.source / f"responses/{name}.jsonl"),
                sha256_file(destination / f"responses/{name}.jsonl"),
            )
        self.assertFalse((destination / "validation/coverage.json").exists())
        experiment = read_json(destination / "experiment.json")
        self.assertEqual(experiment, dict(self.experiment, run_name="gated"))
        with patch(
            f"{_INFERENCE}.prepare_run.repository_root",
            return_value=self.root,
        ), self.assertRaisesRegex(ValueError, "does not match"):
            prepare_run(
                experiment, self.benchmark.manifest_path,
                self.benchmark.manifest, self.benchmark.items,
            )
        specs, _ = build_predicted_lc_specs(
            destination, self.cases, "model", "SEQ",
            lc_inference_policy="reference-supported-answer-v1",
        )
        remaining = next(
            spec for spec in specs
            if spec["task"] == "first_improvement_between_visits"
        )
        with patch(
            f"{_INFERENCE}.run_change_labels.submit_messages",
            return_value={
                "kind": "text", "error": None,
                "raw_text": json.dumps({"answer": remaining["reference_answer"]}),
            },
        ) as submit:
            resumed = execute_specs(
                specs, destination / "responses/seq_lc.jsonl", "result_id",
                client=object(), model="model", image_root=None,
                workers=1, progress=False,
            )
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(resumed["reused"], 3)
        self.assertEqual(load_validated_run(destination).coverage["status"], "passed")
        self.assertEqual(before, self._hashes())

    def test_invalid_saved_response_prevents_destination_creation(self) -> None:
        """Even an unsupported legacy answer must validate before replacement."""
        path = self.source / "responses/pair_lc.jsonl"
        rows = read_jsonl(path)
        row = next(row for row in rows if row["task"] == "trajectory_shape")
        row["final_case_id"] = "wrong"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "final_case_id"):
            migrate_lc_policy(self.source, "gated")
        self.assertFalse((self.source.parent / "gated").exists())

    def test_destination_and_current_policy_cannot_be_overwritten(self) -> None:
        """Existing runs and repeat migrations cannot be silently overwritten."""
        destination = self.source.parent / "gated"
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, "new run name"):
            migrate_lc_policy(self.source, "gated")
        destination.rmdir()
        destination.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "new run name"):
            migrate_lc_policy(self.source, "gated")
        self.assertTrue(destination.is_symlink())
        destination.unlink()
        report = migrate_lc_policy(self.source, "gated")
        with self.assertRaisesRegex(ValueError, "legacy run"):
            migrate_lc_policy(Path(report["run_dir"]), "another")

    def test_source_change_aborts_publication(self) -> None:
        """A writer or sync change during migration cannot publish a mixed copy."""
        before = _snapshot(self.source)
        after = dict(before, **{"responses/seq_lc.jsonl": "changed"})
        with patch(f"{_MODULE}._snapshot", side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, "changed during migration"):
                migrate_lc_policy(self.source, "gated")
        self.assertFalse((self.source.parent / "gated").exists())

    def test_incomplete_predictions_and_missing_inputs_are_rejected(self) -> None:
        """Migration cannot derive a policy from incomplete upstream evidence."""
        path = self.source / "responses/pair.jsonl"
        original = read_jsonl(path)
        write_jsonl(path, original[:-1])
        with self.assertRaisesRegex(ValueError, "complete PAIR"):
            migrate_lc_policy(self.source, "gated")
        write_jsonl(path, original)
        path = self.source / "prepared/model_inputs.jsonl"
        inputs = read_jsonl(path)
        kept = [row for row in inputs if row["record_type"] != "PAIR-LC"]
        write_jsonl(path, kept)
        manifest_path = self.source / "prepared/manifest.json"
        manifest = read_json(manifest_path)
        manifest["files"]["model_inputs.jsonl"] = sha256_file(path)
        manifest["counts"]["model_inputs"] = len(kept)
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "exact registered input"):
            migrate_lc_policy(self.source, "gated")
        self.assertFalse((self.source.parent / "gated").exists())

    def test_complete_migration_passes_full_validation(self) -> None:
        """A completed legacy run becomes a complete validated gated run."""
        specs, _ = build_predicted_lc_specs(
            self.source, self.cases, "model", "SEQ", lc_inference_policy=None,
        )
        write_jsonl(self.source / "responses/seq_lc.jsonl", [
            _response(spec, {"answer": spec["reference_answer"]}) for spec in specs
        ])
        report = migrate_lc_policy(self.source, "gated")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["pending_model_calls"], 0)
        self.assertEqual(
            load_validated_run(Path(report["run_dir"])).coverage["status"], "passed",
        )


if __name__ == "__main__":
    unittest.main()

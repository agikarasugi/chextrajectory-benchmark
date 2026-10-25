"""Offline recovery of completed technical failures through ordinary resume."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json, read_jsonl, sha256_file, write_json, write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference import retry_failed
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    derive_fixed_code,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import prepare_run
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
    _pair_spec, _record_from_spec, _response_fields, _seq_spec,
)
from experiments.temporal_reasoning_single_finding.inference.run_lc import (
    _append_model_inputs, build_predicted_lc_specs, build_ref_lc_specs,
)
from experiments.temporal_reasoning_single_finding.inference.run_pipeline import run_pipeline
from experiments.temporal_reasoning_single_finding.inference.run_seq_dir import (
    build_seq_dir_specs,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import interval_keys
from experiments.temporal_reasoning_single_finding.tests.inference.test_resume import (
    _item, _manifest,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import ValidatedBenchmark
from experiments.temporal_reasoning_single_finding.validation.validate_run import (
    load_validated_run, validate_run,
)


_INFERENCE = "experiments.temporal_reasoning_single_finding.inference"
_ERROR = {
    "kind": "response_error", "raw_text": None,
    "error": {"type": "TimeoutError", "message": "Retries exhausted"},
    "attempt_count": 3, "attempt_seconds": [1.0, 1.0, 1.0],
    "request_seconds": 3.0, "usage": None,
}


class RetryFailedTests(unittest.TestCase):
    """Exercise exact reuse, dependency repair, backups, and interrupted cleanup."""

    def setUp(self) -> None:
        """Prepare two synthetic trajectories sharing their first two PAIR calls."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        manifest_path = self.root / "benchmark.json"
        manifest = _manifest()
        write_json(manifest_path, manifest)
        items = [
            _item("three", 3, ["deterioration"] * 2),
            _item("four", 4, ["deterioration"] * 3),
        ]
        self.benchmark = ValidatedBenchmark(manifest_path, manifest, {3: items[:1], 4: items[1:]}, items)
        self.experiment = {
            "run_name": "retry-test", "model": "model", "base_url": "http://unused/v1",
            "benchmark_path": str(manifest_path),
        }
        self.experiment_path = self.root / "experiment.json"
        self.paths_path = self.root / "paths.json"
        (self.root / "no-images").mkdir()
        write_json(self.experiment_path, self.experiment)
        write_json(self.paths_path, {"chexpert_plus_image_root": str(self.root / "no-images")})
        with patch(f"{_INFERENCE}.prepare_run.repository_root", return_value=self.root):
            self.run_dir = prepare_run(self.experiment, manifest_path, manifest, items)["run_dir"]
        self.responses = {}
        self.calls = []

    def _write_specs(self, stage: str, specs: list[dict], failures: dict[int, str]) -> None:
        """Write real parser/finalizer records and register successful fake replies."""
        rows = []
        for index, spec in enumerate(specs):
            if "precomputed_record" in spec:
                rows.append(spec["precomputed_record"])
                continue
            if stage == "pair":
                output = {"change_label": spec["reference_change_label"]}
            elif stage == "seq":
                output = {"change_labels": dict(zip(
                    interval_keys(spec["timepoint_count"]), spec["reference_change_labels"], strict=True,
                ))}
            else:
                output = {"answer": spec["reference_answer"]}
            transport = {"kind": "text", "raw_text": json.dumps(output), "error": None}
            self.responses[json.dumps(spec["safe_model_input"], sort_keys=True)] = transport
            failure = failures.get(index)
            if failure == "response_error":
                transport = _ERROR
            elif failure == "invalid_response":
                transport = {"kind": "text", "raw_text": "invalid", "error": None}
            row = _record_from_spec(spec)
            row.update(_response_fields(
                transport, spec["response_type"],
                timepoint_count=spec.get("timepoint_count_for_parser"),
                task=spec.get("task_for_parser"),
            ))
            for field in ("usage", "attempt_count", "attempt_seconds", "request_seconds"):
                if field in transport:
                    row[field] = transport[field]
            spec["finalize"](row)
            rows.append(row)
        write_jsonl(self.run_dir / "responses" / f"{stage}.jsonl", rows)

    def _complete(self, failures: dict[str, dict[int, str]]) -> None:
        """Create a validated completed run with the requested terminal outcomes."""
        prepared = self.run_dir / "prepared"
        for stage, builder in (("pair", _pair_spec), ("seq", _seq_spec)):
            specs = [builder(value) for value in read_json(prepared / f"{stage}_inputs.json")]
            self._write_specs(stage, specs, failures.get(stage, {}))
        derive_fixed_code(self.run_dir)
        cases = read_json(prepared / "final_cases.json")
        for stage, builder in (("seq_dir", build_seq_dir_specs), ("ref_lc", build_ref_lc_specs)):
            self._write_specs(stage, builder(self.run_dir, cases, "model"), failures.get(stage, {}))
        for source in ("pair", "seq"):
            specs, inputs = build_predicted_lc_specs(self.run_dir, cases, "model", source.upper())
            _append_model_inputs(self.run_dir, inputs)
            self._write_specs(f"{source}_lc", specs, failures.get(f"{source}_lc", {}))
        validate_run(self.run_dir)
        write_json(self.run_dir / "evaluation.json", {"previous": "evaluation"})

    def _pipeline(
        self, *, retry: bool = False, fail_again: bool = False,
        interrupt_after: int | None = None,
    ) -> dict:
        """Execute the real pipeline with only preflight and transport stubbed."""
        client = SimpleNamespace(models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="model")]),
        ))

        def submit(_client, _model, messages, *_args, **_kwargs) -> dict:
            """Resolve only safe text inputs and never open any image file."""
            self.calls.append(messages)
            if interrupt_after is not None and len(self.calls) > interrupt_after:
                raise RuntimeError("fixture process interruption")
            return _ERROR if fail_again else self.responses[json.dumps(messages, sort_keys=True)]

        with (
            patch(f"{_INFERENCE}.run_pipeline.preflight", return_value={}),
            patch(f"{_INFERENCE}.run_pipeline.load_and_validate_benchmark", return_value=self.benchmark),
            patch(f"{_INFERENCE}.prepare_run.repository_root", return_value=self.root),
            patch(f"{_INFERENCE}.run_pipeline.create_client", return_value=client),
            patch(f"{_INFERENCE}.run_pipeline.require_api_key", return_value="fixture"),
            patch(f"{_INFERENCE}.run_change_labels.submit_messages", side_effect=submit),
        ):
            return run_pipeline(self.experiment_path, self.paths_path, retry_failed=retry, workers=2)

    def test_final_failures_retry_once_and_preserve_other_outcomes(self) -> None:
        """Every final model stage retries only technical errors, with no duplicates."""
        self._complete({stage: {0: "response_error", 1: "invalid_response"}
                        for stage in ("seq_dir", "ref_lc", "pair_lc", "seq_lc")})
        before = {path: read_jsonl(path) for path in (self.run_dir / "responses").glob("*.jsonl")}
        self._pipeline()
        self.assertEqual(self.calls, [])
        result = self._pipeline(retry=True)
        self.assertEqual(result["retry_failed"]["response_errors"], 4)
        self.assertEqual(len(self.calls), 4)
        for path, rows in before.items():
            after = read_jsonl(path)
            self.assertEqual(len(rows), len(after))
            for row in rows:
                if row["status"] != "response_error":
                    self.assertIn(row, after)
        archive = self.run_dir / result["retry_failed"]["backup_directory"]
        plan = read_json(archive / "manifest.json")
        for relative, checksum in plan["source_sha256"].items():
            self.assertEqual(sha256_file(archive / relative), checksum)
        self.assertTrue((archive / "evaluation.json").exists())
        self.assertFalse((self.run_dir / "evaluation.json").exists())
        self.calls.clear()
        self.assertEqual(self._pipeline(retry=True)["retry_failed"]["response_errors"], 0)
        self.assertEqual(self.calls, [])

    def test_upstream_recovery_rebuilds_only_affected_lc_cases(self) -> None:
        """Shared PAIR recovery reaches both trajectories and preserves invalid SEQ."""
        self._complete({"pair": {0: "response_error"},
                        "seq": {0: "response_error", 1: "invalid_response"}})
        saved = load_validated_run(self.run_dir)
        affected = sum(row["status"] == "upstream_response_error"
                       for stage in ("PAIR-LC", "SEQ-LC") for row in saved.results[stage].values())
        result = self._pipeline(retry=True)
        self.assertEqual(result["retry_failed"]["response_errors"], 2)
        self.assertEqual(result["retry_failed"]["downstream_results"], affected)
        self.assertEqual(len(self.calls), 2 + affected)
        updated = load_validated_run(self.run_dir)
        self.assertTrue(all(row["status"] == "valid" for row in updated.pair_predictions.values()))
        invalid = [row for row in saved.seq_predictions.values() if row["status"] == "invalid_response"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(updated.seq_predictions[invalid[0]["prediction_id"]], invalid[0])
        self.calls.clear()
        self._pipeline()
        self.assertEqual(self.calls, [])

    def test_repeated_failure_remains_terminal_until_next_explicit_retry(self) -> None:
        """One invocation never repeatedly reopens a newly failed request."""
        self._complete({"pair": {0: "response_error"}})
        self._pipeline(retry=True, fail_again=True)
        self.assertEqual(len(self.calls), 1)
        self.calls.clear()
        self._pipeline()
        self.assertEqual(self.calls, [])
        self._pipeline(retry=True)
        self.assertGreater(len(self.calls), 1)
        self.assertEqual(len(list((self.run_dir / "retries").iterdir())), 2)

    def test_interrupted_cleanup_recovers_without_flag(self) -> None:
        """A saved plan safely replays even after one response file was replaced."""
        self._complete({"pair": {0: "response_error"}, "seq_dir": {0: "response_error"}})
        original = retry_failed.write_jsonl
        calls = 0

        def interrupt(path: Path, rows: list[dict]) -> None:
            """Stop immediately after the first atomic response-file replacement."""
            nonlocal calls
            original(path, rows)
            calls += 1
            if calls == 1:
                raise InterruptedError("fixture interruption")

        with patch.object(retry_failed, "write_jsonl", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                retry_failed.prepare_failed_retry(self.run_dir, retry_failed=True)
        self.assertTrue((self.run_dir / "retry_pending.json").exists())
        result = self._pipeline()
        self.assertEqual(result["retry_failed"]["response_errors"], 2)
        self.assertFalse((self.run_dir / "retry_pending.json").exists())
        load_validated_run(self.run_dir)

    def test_interrupted_inference_resumes_only_remaining_work(self) -> None:
        """A retry saved before interruption is reused on the next ordinary resume."""
        self._complete({"seq_dir": {0: "response_error", 2: "response_error"}})
        with self.assertRaisesRegex(RuntimeError, "fixture process interruption"):
            self._pipeline(retry=True, interrupt_after=1)
        rows = read_jsonl(self.run_dir / "responses/seq_dir.jsonl")
        self.assertEqual(len(rows), 7)
        self.assertFalse((self.run_dir / "retry_pending.json").exists())
        self.calls.clear()
        self._pipeline()
        self.assertEqual(len(self.calls), 1)
        load_validated_run(self.run_dir)

    def test_incomplete_or_corrupt_run_is_unchanged(self) -> None:
        """Full validation precedes backups and mutation of the saved evidence."""
        self._complete({"pair": {0: "response_error"}})
        path = self.run_dir / "responses/seq_dir.jsonl"
        write_jsonl(path, read_jsonl(path)[:-1])
        before = {p: p.read_bytes() for p in self.run_dir.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            retry_failed.prepare_failed_retry(self.run_dir, retry_failed=True)
        self.assertEqual(before, {p: p.read_bytes() for p in self.run_dir.rglob("*") if p.is_file()})

    def test_pipeline_cli_accepts_flag(self) -> None:
        """Expose retry as an opt-in runtime flag rather than experiment identity."""
        from experiments.temporal_reasoning_single_finding.inference.run_pipeline import parse_args

        with patch("sys.argv", ["run_pipeline", "--experiment", "e", "--paths", "p", "--retry-failed"]):
            self.assertTrue(parse_args().retry_failed)

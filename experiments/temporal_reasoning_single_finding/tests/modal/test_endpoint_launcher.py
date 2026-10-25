"""Verify endpoint readiness and direct execution with synthetic text artifacts."""

import os
import signal
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, APIStatusError

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json, read_jsonl, sha256_file, write_json,
)
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import execute_specs
from experiments.temporal_reasoning_single_finding.modal import endpoint_launcher as launcher
from .test_endpoint_config import experiment, profile


class ReadinessTests(unittest.TestCase):
    """Exercise real model-list decoding over a mocked SDK transport."""

    def setUp(self) -> None:
        """Inject a synthetic credential and context-managed client."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, OPENAI_API_KEY="synthetic-token"))
        self.factory = self.stack.enter_context(patch.object(launcher, "create_client"))
        self.client = self.factory.return_value.__enter__.return_value
        self.request_client = self.client.with_options.return_value

    def test_exact_model_and_authentication(self) -> None:
        """Authenticate discovery and bound its request timeout without SDK retries."""
        self.request_client.models.list.return_value.data = [Mock(id="synthetic-model")]
        self.assertEqual(
            launcher.wait_ready("https://endpoint.invalid/v1", "synthetic-model", 30),
            "synthetic-model",
        )
        self.factory.assert_called_once_with("https://endpoint.invalid/v1", "synthetic-token")
        self.client.with_options.assert_called_once_with(timeout=10.0, max_retries=0)
        self.factory.return_value.__exit__.assert_called_once()

    def test_mismatch_and_malformed_metadata_fail_promptly(self) -> None:
        """Never log the server's model metadata when discovery is incompatible."""
        for data in ([Mock(id="another-model")], [], "not-an-array"):
            self.request_client.models.list.return_value.data = data
            with self.assertRaises(ValueError) as raised:
                launcher.wait_ready("https://endpoint.invalid/v1", "synthetic-model", 30)
            self.assertNotIn("another-model", str(raised.exception))

    def test_permanent_http_errors_omit_response_bodies(self) -> None:
        """Reject authentication and route errors without exposing token-bearing text."""
        for status in (400, 401, 403, 404):
            self.request_client.models.list.side_effect = APIStatusError(
                "synthetic-token", response=httpx.Response(
                    status, request=httpx.Request("GET", "https://endpoint.invalid/v1/models"),
                ), body={"unsafe": "synthetic-token"},
            )
            with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as raised:
                launcher.wait_ready("https://endpoint.invalid/v1", "synthetic-model", 30)
            self.assertNotIn("synthetic-token", str(raised.exception))
            self.assertTrue(raised.exception.__suppress_context__)

    def test_retryable_errors_can_recover(self) -> None:
        """Retry rate limits and server failures only within readiness polling."""
        for status in (408, 429, 503):
            error = APIStatusError("unavailable", response=httpx.Response(
                status, request=httpx.Request("GET", "https://endpoint.invalid/v1/models"),
            ), body=None)
            self.request_client.models.list.side_effect = [
                error, Mock(data=[Mock(id="synthetic-model")]),
            ]
            with patch.object(launcher.time, "sleep"):
                self.assertEqual(
                    launcher.wait_ready("https://endpoint.invalid/v1", "synthetic-model", 30),
                    "synthetic-model",
                )

    def test_unavailable_endpoint_times_out(self) -> None:
        """Stop polling after the configured deadline on connection failures."""
        self.request_client.models.list.side_effect = APIConnectionError(
            request=httpx.Request("GET", "https://endpoint.invalid/v1/models"),
        )
        with (
            patch.object(launcher.time, "monotonic", side_effect=[0, 0, 1, 2]),
            patch.object(launcher.time, "sleep"), self.assertRaises(TimeoutError),
        ):
            launcher.wait_ready("https://endpoint.invalid/v1", "synthetic-model", 1)


class ExecutionTests(unittest.TestCase):
    """Check direct calls, checkpoint failures, credential isolation, and resume."""

    def setUp(self) -> None:
        """Create only synthetic benchmark text and mocked mounted-image metadata."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.experiment = experiment()
        self.profile = profile()
        benchmark = self.root / self.experiment["benchmark_path"]
        write_json(benchmark, {"synthetic": True})
        write_json(benchmark.parent / "modal_images.json", {
            "benchmark_sha256": sha256_file(benchmark), "images": {"synthetic.png": {"bytes": 1}},
        })
        self.metadata = self.root / launcher.RUN_ROOT / ".modal" / self.experiment["run_name"]
        self.stack.enter_context(patch.dict(os.environ, OPENAI_API_KEY="synthetic-token"))
        for name, value in (("repository_root", self.root), ("source_digest", "code"),
                            ("package_versions", {"python": "3.12", "openai": "2.48.0"})):
            self.stack.enter_context(patch.object(launcher, name, return_value=value))
        self.check_metadata = self.stack.enter_context(
            patch.object(launcher, "check_file_metadata")
        )
        self.ready = self.stack.enter_context(
            patch.object(launcher, "wait_ready", return_value="synthetic-model")
        )
        self.pipeline = self.stack.enter_context(
            patch.object(launcher, "run_pipeline", return_value={"status": "passed"})
        )
        self.evaluate = self.stack.enter_context(patch.object(launcher, "evaluate_run"))
        self.commit = Mock()
        self.previous_handler = signal.getsignal(signal.SIGTERM)

    def execute(self, workers: int = 2, *, retry_failed: bool = False) -> dict:
        """Invoke the wrapper with a caller-owned commit callback."""
        return launcher.execute_job(
            self.experiment, self.profile, "code", self.commit, workers=workers,
            retry_failed=retry_failed,
        )

    def test_direct_call_order_and_safe_artifacts(self) -> None:
        """Pass temporary configuration and injected auth directly through the pipeline."""
        self.experiment["max_completion_tokens"] = 2048
        events = []

        def pipeline(experiment_path, paths_path, *, workers, progress, retry_failed):
            """Validate the direct-call contract while its temporary files exist."""
            self.assertEqual(read_json(experiment_path), self.experiment)
            self.assertEqual(read_json(paths_path), {"chexpert_plus_image_root": "/data/images"})
            self.assertEqual((workers, progress), (2, True))
            self.assertTrue(retry_failed)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "synthetic-token")
            events.append("inference")
            return {"status": "passed"}

        self.pipeline.side_effect = pipeline
        self.evaluate.side_effect = lambda name: events.append("evaluate")
        self.commit.side_effect = lambda: events.append("commit")
        with patch("subprocess.Popen", side_effect=AssertionError("subprocess launched")):
            result = self.execute(retry_failed=True)
        self.assertEqual(events, ["commit", "commit", "inference", "commit", "evaluate", "commit"])
        self.assertEqual(result["status"], "complete")
        self.assertFalse(self.pipeline.call_args.args[0].exists())
        for path in self.metadata.rglob("*.json"):
            self.assertNotIn("synthetic-token", path.read_text())
        self.assertEqual(read_json(self.metadata / "status.json"), {"status": "complete"})
        self.assertIs(signal.getsignal(signal.SIGTERM), self.previous_handler)

    def test_missing_secret_and_changed_source_fail_before_readiness(self) -> None:
        """Do not make requests when credentials or submitted source identity fail."""
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(EnvironmentError):
            self.execute()
        with (
            patch.object(launcher, "source_digest", return_value="wrong"),
            self.assertRaises(ValueError),
        ):
            self.execute()
        self.ready.assert_not_called()
        self.check_metadata.assert_not_called()
        self.commit.assert_not_called()

    def test_failure_preserves_original_exception_if_cleanup_fails(self) -> None:
        """Retain the inference error while recording a failed persistence attempt."""
        error = RuntimeError("synthetic failure")
        self.pipeline.side_effect = error
        self.commit.side_effect = [None, None, OSError("commit failure")]
        with self.assertRaises(RuntimeError) as raised:
            self.execute()
        self.assertIs(raised.exception, error)
        self.assertIn("Failed to persist", error.__notes__[0])
        self.assertEqual(read_json(self.metadata / "status.json"), {"status": "failed"})
        self.evaluate.assert_not_called()
        self.assertIs(signal.getsignal(signal.SIGTERM), self.previous_handler)

    def test_every_required_commit_failure_aborts_success(self) -> None:
        """Preparation, discovery, inference, and completion checkpoints propagate errors."""
        for boundary in range(4):
            with self.subTest(boundary=boundary):
                self.commit.side_effect = [None] * boundary + [OSError("storage failure"), None]
                self.pipeline.reset_mock()
                self.evaluate.reset_mock()
                with self.assertRaises(OSError):
                    self.execute()
                self.assertEqual(read_json(self.metadata / "status.json"), {"status": "failed"})
                if boundary < 2:
                    self.pipeline.assert_not_called()
                if boundary < 3:
                    self.evaluate.assert_not_called()

    def test_readiness_or_scoring_failure_marks_failed(self) -> None:
        """Report failed attempts whether failure precedes or follows inference."""
        for mock in (self.ready, self.evaluate):
            with self.subTest(stage=mock):
                mock.side_effect = ValueError("synthetic")
                with self.assertRaises(ValueError):
                    self.execute()
                self.assertEqual(read_json(self.metadata / "status.json"), {"status": "failed"})
                mock.side_effect = None

    def test_completion_limit_resume(self) -> None:
        """Accept old omitted limits and reject changes before endpoint readiness."""
        self.execute()
        self.experiment["max_completion_tokens"] = None
        self.execute()
        self.pipeline.reset_mock()
        with patch.object(launcher, "wait_ready") as ready:
            self.experiment["max_completion_tokens"] = 2048
            with self.assertRaisesRegex(ValueError, "new run_name"):
                self.execute()
            ready.assert_not_called()
        self.pipeline.assert_not_called()
        self.experiment["run_name"] += "-capped"
        self.execute()
        self.execute()
        for limit in (None, 1024):
            self.experiment["max_completion_tokens"] = limit
            with patch.object(launcher, "wait_ready") as ready:
                with self.assertRaisesRegex(ValueError, "new run_name"):
                    self.execute()
                ready.assert_not_called()

    def test_changed_endpoint_benchmark_or_deployment_rejects_resume(self) -> None:
        """Keep endpoint identity strict while allowing new execution resources."""
        self.execute()
        original = (self.metadata / "provenance.json").read_bytes()
        self.profile["cpu"] = 4
        self.execute()
        self.assertEqual((self.metadata / "provenance.json").read_bytes(), original)
        for changed in (
            {"base_url": "https://other.invalid/v1"}, {"model": "other-model"},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "provenance changed"),
            ):
                launcher.execute_job(
                    dict(self.experiment, **changed), self.profile, "code", self.commit,
                    workers=2,
                )
        self.profile["deployment_identity"] = {"deployment_id": "new"}
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            self.execute()
        self.profile.pop("deployment_identity")
        benchmark = self.root / self.experiment["benchmark_path"]
        write_json(benchmark, {"synthetic": "changed"})
        staged = read_json(benchmark.parent / "modal_images.json")
        staged["benchmark_sha256"] = sha256_file(benchmark)
        write_json(benchmark.parent / "modal_images.json", staged)
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            self.execute()

    def test_resume_repairs_tail_and_submits_only_missing_records(self) -> None:
        """Reuse saved records with new concurrency after an interrupted append."""
        output = (
            self.root / launcher.RUN_ROOT / self.experiment["run_name"]
            / "responses/synthetic.jsonl"
        )

        def finalize(record):
            """Retain the parsed answer in a synthetic saved response."""
            record["answer"] = record["parsed_output"]["answer"]

        specs = [{
            "result_id": name, "model_input_id": f"input-{name}",
            "safe_model_input": [{"role": "user", "content": name}],
            "response_type": "final", "timepoint_count_for_parser": 3,
            "task_for_parser": "trajectory_shape",
            "identity_fields": ("result_id", "model_input_id"),
            "finalize": finalize,
        } for name in ("first", "second")]

        def pipeline(*args, **kwargs):
            """Use the production response scheduler with synthetic text requests."""
            return execute_specs(
                specs, output, "result_id", client=Mock(), model="synthetic-model",
                image_root=None, workers=kwargs["workers"], progress=False,
            )

        self.pipeline.side_effect = pipeline
        response = {"kind": "text", "raw_text": '{"answer":"constant"}'}
        with patch(
            "experiments.temporal_reasoning_single_finding.inference."
            "run_change_labels.submit_messages",
            side_effect=[response, RuntimeError("interrupted")],
        ) as transport:
            with self.assertRaises(RuntimeError):
                self.execute(workers=1)
            provenance_path = self.metadata / "provenance.json"
            original_provenance = provenance_path.read_bytes()
            saved = output.read_bytes()
            with output.open("ab") as handle:
                handle.write(b'{"partial":')
            transport.reset_mock(side_effect=True)
            transport.return_value = response
            self.execute()
            transport.assert_called_once()
            self.assertEqual(transport.call_args.args[2], specs[1]["safe_model_input"])
        self.assertEqual(len(read_jsonl(output)), 2)
        self.assertTrue(output.read_bytes().startswith(saved))
        self.assertEqual(provenance_path.read_bytes(), original_provenance)
        attempts = list((self.metadata / "attempts").iterdir())
        self.assertEqual(sorted(
            read_json(attempt / "provenance.json")["provenance"]["workers"]
            for attempt in attempts
        ), [1, 2])
        self.assertEqual(len(list((self.metadata / "attempts").iterdir())), 2)
        self.assertEqual(len(list((self.metadata / "recovery").iterdir())), 1)

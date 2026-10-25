"""Check interruption and writer failure against real queued worker execution."""

import importlib
import signal
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import read_jsonl

stage = importlib.import_module(
    "experiments.temporal_reasoning_single_finding.inference.run_change_labels"
)


def specs() -> list[dict]:
    """Build eight resumable final-response specs containing synthetic text only."""
    def finalize(record: dict) -> None:
        """Keep the parsed synthetic answer for response validation on resume."""
        record["answer"] = record["parsed_output"]["answer"]

    return [{
        "result_id": str(index), "model_input_id": f"input-{index}",
        "safe_model_input": [{"identifier": str(index)}],
        "response_type": "final", "timepoint_count_for_parser": 3,
        "task_for_parser": "trajectory_shape",
        "identity_fields": ("result_id", "model_input_id"), "finalize": finalize,
    } for index in range(8)]


class CancellationTests(unittest.TestCase):
    """Keep interruption bounded to running work and retain recoverable responses."""

    def exercise(self, mode: str) -> None:
        """Interrupt after one saved record with two workers and six queued specs."""
        second_started = threading.Event()
        release = threading.Event()
        started = []
        interrupted = False
        failure = OSError("synthetic append failure") if mode == "write" else KeyboardInterrupt()

        class ControlledExecutor(ThreadPoolExecutor):
            """Release active requests only after shutdown has applied queue cancellation."""

            def shutdown(self, wait=True, *, cancel_futures=False):
                """Use real executor cancellation before releasing deterministic requests."""
                super().shutdown(wait=False, cancel_futures=cancel_futures)
                release.set()
                if wait:
                    super().shutdown(wait=True)

        def submit(_client, _model, messages, _image_root):
            """Hold active requests while the coordinator handles its interruption."""
            identifier = messages[0]["identifier"]
            started.append(identifier)
            if identifier == "0":
                self.assertTrue(second_started.wait(timeout=3))
            else:
                second_started.set()
                self.assertTrue(release.wait(timeout=3))
            return {"kind": "text", "raw_text": '{"answer":"constant"}'}

        def completions(futures):
            """Interrupt the real completion iterator after its first persisted result."""
            nonlocal interrupted
            for future in as_completed(futures):
                yield future
                if mode != "write" and not interrupted:
                    interrupted = True
                    if mode == "signal":
                        signal.raise_signal(signal.SIGTERM)
                    else:
                        raise failure

        original_append = stage.append_jsonl

        def append(path, record):
            """Simulate a writer that raises after some bytes have reached its file."""
            original_append(path, record)
            if mode == "write":
                raise failure

        def terminate(signum, frame):
            """Model the endpoint wrapper's graceful termination handler."""
            raise InterruptedError("synthetic termination")

        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "responses.jsonl"
                with (
                    patch.object(stage, "ThreadPoolExecutor", ControlledExecutor),
                    patch.object(stage, "as_completed", side_effect=completions),
                    patch.object(stage, "submit_messages", side_effect=submit),
                    patch.object(stage, "append_jsonl", side_effect=append),
                ):
                    expected = InterruptedError if mode == "signal" else type(failure)
                    with self.assertRaises(expected) as raised:
                        stage.execute_specs(
                            specs(), path, "result_id", client=object(), model="synthetic",
                            image_root=None, workers=2, progress=False,
                        )
                if mode != "signal":
                    self.assertIs(raised.exception, failure)
                # A worker may take one additional spec before the interruption.
                self.assertLessEqual(len(started), 3)
                saved = read_jsonl(path)
                saved_ids = {row["result_id"] for row in saved}
                self.assertEqual(len(saved), len(saved_ids))
                if mode == "write":
                    self.assertEqual(saved_ids, {"0"})
                else:
                    self.assertEqual(saved_ids, set(started))
                resumed_ids = []

                def resumed_submit(_client, _model, messages, _image_root):
                    """Record only the missing specs submitted after interruption."""
                    resumed_ids.append(messages[0]["identifier"])
                    return {"kind": "text", "raw_text": '{"answer":"constant"}'}

                with patch.object(stage, "submit_messages", side_effect=resumed_submit):
                    summary = stage.execute_specs(
                        specs(), path, "result_id", client=object(), model="synthetic",
                        image_root=None, workers=2, progress=False,
                    )
                self.assertEqual(set(resumed_ids), set(map(str, range(8))) - saved_ids)
                self.assertEqual(summary["reused"], len(saved_ids))
                self.assertEqual(len(read_jsonl(path)), 8)
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_sigterm_cancels_queue_and_retains_active_responses(self) -> None:
        """Stop queued calls on actual SIGTERM while keeping completed active responses."""
        self.exercise("signal")

    def test_base_exception_cancels_queue_and_retains_active_responses(self) -> None:
        """Cover cancellation exceptions that bypass ordinary worker-error handling."""
        self.exercise("base")

    def test_writer_failure_cancels_queue_without_retrying_append(self) -> None:
        """Stop new calls and avoid duplicate or corrupt appends after a writer error."""
        self.exercise("write")

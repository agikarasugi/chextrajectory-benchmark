"""Tests for endpoint token usage persistence and backward-compatible resume."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_jsonl,
    write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference.model_client import (
    call_chat_model,
)
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
    execute_specs,
)
from .test_model_client import FakeClient, _response, _text_messages


class TokenUsageTests(unittest.TestCase):
    """Verify counts without endpoint calls or image payloads."""

    def test_transport_usage_for_terminal_responses(self) -> None:
        """Text, refusals, and filtered answers retain reported counts only."""
        for response in (
            _response('{"answer":"constant"}'),
            _response(None, refusal="Cannot answer"),
            _response(None, finish_reason="content_filter"),
        ):
            with self.subTest(response=response):
                response.usage = SimpleNamespace(
                    prompt_tokens=120, completion_tokens=0, total_tokens=120,
                    extra="ignored",
                )
                result = call_chat_model(FakeClient([response]), "model", _text_messages())
                self.assertEqual(result["usage"], {
                    "input_tokens": 120, "output_tokens": 0, "total_tokens": 120,
                })

    def test_request_seconds_include_retries_and_failures(self) -> None:
        """Measure elapsed transport time for success and exhausted retries."""
        for responses in (
            [_response("answer")],
            [_response(None), _response("answer")],
            [_response(None)] * 3,
            [_response(None, refusal="Cannot answer")],
            [_response(None, finish_reason="content_filter")],
        ):
            count = len(responses)
            ticks = [0.0]
            for index in range(count):
                ticks.extend([index * 3.0, index * 3.0 + 2.5])
            ticks.append(count * 3.0)
            with self.subTest(attempts=count), patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "model_client.time.perf_counter", side_effect=ticks,
            ):
                result = call_chat_model(FakeClient(responses), "model", _text_messages())
                self.assertEqual(result["request_seconds"], count * 3.0)
                self.assertEqual(result["attempt_count"], count)
                self.assertEqual(result["attempt_seconds"], [2.5] * count)

    def test_missing_and_partial_usage(self) -> None:
        """Unavailable or malformed counts are unknown rather than zero."""
        response = _response("answer")
        result = call_chat_model(FakeClient([response]), "model", _text_messages())
        self.assertIsNone(result["usage"])
        for invalid in (None, -1, True, "5", 1.5):
            with self.subTest(invalid=invalid):
                response.usage = SimpleNamespace(prompt_tokens=5, completion_tokens=invalid)
                result = call_chat_model(FakeClient([response]), "model", _text_messages())
                self.assertEqual(result["usage"], {
                    "input_tokens": 5, "output_tokens": None, "total_tokens": None,
                })

    def test_retries_report_returned_attempt_only(self) -> None:
        """Empty attempts do not inflate the returned completion's usage."""
        empty = _response(None)
        empty.usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        success = _response("answer")
        success.usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
        result = call_chat_model(FakeClient([empty, success]), "model", _text_messages())
        self.assertEqual(result["usage"]["total_tokens"], 70)
        failed = call_chat_model(FakeClient([empty] * 3), "model", _text_messages())
        self.assertEqual(failed["kind"], "response_error")
        self.assertIsNone(failed["usage"])

    def test_jsonl_usage_and_legacy_resume(self) -> None:
        """Persist usage, reuse new and legacy records, and reject bad counts."""
        def finalize(record: dict) -> None:
            """Keep this fixture focused on shared response fields."""

        spec = {
            "result_id": "result", "model_input_id": "input",
            "safe_model_input": [], "response_type": "final",
            "timepoint_count_for_parser": 3,
            "task_for_parser": "trajectory_shape",
            "identity_fields": ("result_id", "model_input_id"),
            "finalize": finalize,
        }
        usage = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
        outcomes = (
            {"kind": "text", "raw_text": '{"answer":"constant"}', "error": None},
            {"kind": "text", "raw_text": "invalid JSON", "error": None},
            {"kind": "refusal", "raw_text": "Cannot answer", "error": None},
            {"kind": "content_filter", "raw_text": None, "error": None},
            {"kind": "response_error", "raw_text": None,
             "error": {"type": "APIConnectionError", "message": "Connection failed"}},
        )
        for outcome in outcomes:
            with self.subTest(kind=outcome["kind"]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "responses.jsonl"
                options = dict(client=object(), model="model", image_root=None,
                               workers=1, progress=False)
                expected_usage = None if outcome["kind"] == "response_error" else usage
                with patch(
                    "experiments.temporal_reasoning_single_finding.inference."
                    "run_change_labels.submit_messages",
                    return_value={
                        **outcome, "usage": expected_usage, "request_seconds": 2.5,
                        "attempt_count": 1, "attempt_seconds": [2.5],
                    },
                ) as submit:
                    execute_specs([spec], path, "result_id", **options)
                    row = read_jsonl(path)[0]
                    self.assertEqual(row["usage"], expected_usage)
                    self.assertEqual(row["request_seconds"], 2.5)
                    self.assertEqual(row["attempt_count"], 1)
                    self.assertEqual(row["attempt_seconds"], [2.5])
                    resumed = execute_specs([spec], path, "result_id", **options)
                    self.assertEqual(resumed["reused"], 1)
                    del row["attempt_count"]
                    del row["attempt_seconds"]
                    write_jsonl(path, [row])
                    self.assertEqual(
                        execute_specs([spec], path, "result_id", **options)["reused"], 1
                    )
                    del row["usage"]
                    del row["request_seconds"]
                    write_jsonl(path, [row])
                    resumed = execute_specs([spec], path, "result_id", **options)
                    self.assertEqual(resumed["reused"], 1)
                    submit.assert_called_once()
                    for timing in (
                        {"attempt_count": 0, "attempt_seconds": []},
                        {"attempt_count": True, "attempt_seconds": [1.0]},
                        {"attempt_count": 2, "attempt_seconds": [1.0]},
                        {"attempt_count": 1},
                        {"attempt_seconds": [1.0]},
                        *({"attempt_count": 1, "attempt_seconds": [value]}
                          for value in (-1, True, "2", float("nan"), float("inf"))),
                    ):
                        path.write_text(json.dumps({**row, **timing}) + "\n")
                        with self.assertRaises(ValueError):
                            execute_specs([spec], path, "result_id", **options)
                    for seconds in (-1, True, "2", float("nan"), float("inf")):
                        path.write_text(json.dumps({**row, "request_seconds": seconds}) + "\n")
                        with self.assertRaises(ValueError):
                            execute_specs([spec], path, "result_id", **options)
                    for malformed in (True, {}, {**usage, "input_tokens": -1},
                                      {**usage, "output_tokens": True}):
                        write_jsonl(path, [{**row, "usage": malformed}])
                        with self.assertRaises(ValueError):
                            execute_specs([spec], path, "result_id", **options)

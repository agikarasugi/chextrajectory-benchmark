"""Tests for the exact private Chat Completions transport."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from experiments.temporal_reasoning_single_finding.inference import model_client
from experiments.temporal_reasoning_single_finding.inference.api_errors import EndpointRequestError
from experiments.temporal_reasoning_single_finding.protocol.prompts import (
    ROLE_INSTRUCTION,
)


def _text_messages(text: str = "Return JSON") -> list[dict]:
    """Build one safe text-only message fixture."""
    return [
        {"role": "user", "content": ROLE_INSTRUCTION + "\n\n" + text},
    ]


def _response(
    content: str | None,
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build one minimal Chat Completions response fixture."""
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _status_error(status: int) -> APIStatusError:
    """Build one SDK status exception without external traffic."""
    request = httpx.Request("POST", "http://server/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(
        "unsafe body data:image/png;base64,secret",
        response=response,
        body={"unsafe": "data:image/png;base64,secret"},
    )


class FakeCompletions:
    """Record deep-copied requests and return configured outcomes."""

    def __init__(self, outcomes: list[object]):
        """Store an ordered outcome stream."""
        self.outcomes = iter(outcomes)
        self.requests: list[dict] = []

    def create(self, **request: object) -> object:
        """Raise or return the next configured outcome."""
        self.requests.append(copy.deepcopy(request))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    """Expose fake models and Chat Completions resources."""

    def __init__(
        self,
        outcomes: list[object],
        model_ids: tuple[str, ...] = ("exact-model",),
    ) -> None:
        """Create independent fake endpoint resources."""
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=value) for value in model_ids]
            )
        )


class ModelClientTests(unittest.TestCase):
    """Exercise request privacy, discovery, extraction, and retry policy."""

    def setUp(self) -> None:
        """Create a synthetic non-medical PNG path fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.image_root = Path(self.temporary.name).resolve()
        image = self.image_root / "train/patient/study/view1_frontal.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic-png-fixture")
        self.image_messages = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": ROLE_INSTRUCTION + "\n\nEarlier image:\n"},
                    {
                        "type": "image_ref",
                        "image_id": "train/patient/study/view1_frontal.jpg",
                    },
                ],
            },
        ]

    def tearDown(self) -> None:
        """Remove the synthetic image fixture."""
        self.temporary.cleanup()

    def test_transport_rejects_system_roles_and_missing_preamble(self) -> None:
        """Only one user message with the leading instruction is submitted."""
        invalid = [
            [{"role": "system", "content": ROLE_INSTRUCTION}, *_text_messages()],
            [{"role": "system", "content": ROLE_INSTRUCTION}],
            [{"role": "user", "content": "Missing leading instruction"}],
            [{"role": "user", "content": [{"type": "text", "text": "Missing"}]}],
        ]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                model_client.build_chat_request("exact-model", messages)
        request = model_client.build_chat_request("exact-model", _text_messages())
        self.assertEqual(request["messages"], _text_messages())

    def test_create_client_disables_sdk_retries(self) -> None:
        """Set the exact base URL, timeout, key, and zero SDK retries."""
        sentinel = object()
        with patch.object(model_client, "OpenAI", return_value=sentinel) as constructor:
            result = model_client.create_client(
                "http://server:8000/v1/",
                api_key="key",
            )
        self.assertIs(result, sentinel)
        constructor.assert_called_once_with(
            api_key="key",
            base_url="http://server:8000/v1",
            timeout=300.0,
            max_retries=0,
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                model_client.create_client("http://server/v1")

    def test_model_discovery_requires_exact_identifier(self) -> None:
        """Sort discovered IDs and reject aliases or substrings."""
        client = FakeClient([], ("z-model", "exact-model", "exact-model"))
        self.assertEqual(
            model_client.discover_model_ids(client),
            ("exact-model", "z-model"),
        )
        self.assertEqual(
            model_client.require_exact_model(client, "exact-model"),
            "exact-model",
        )
        with self.assertRaisesRegex(ValueError, "not advertised exactly"):
            model_client.require_exact_model(client, "model")
        with self.assertRaisesRegex(ValueError, "no model IDs"):
            model_client.discover_model_ids(FakeClient([], ()))

    def test_request_contains_only_model_and_messages(self) -> None:
        """Materialize PNG data without any forbidden request control."""
        request = model_client.build_chat_request(
            "exact-model",
            self.image_messages,
            self.image_root,
        )
        self.assertEqual(set(request), {"model", "messages"})
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(
            request["messages"][0]["content"][0],
            self.image_messages[0]["content"][0],
        )
        image_part = request["messages"][0]["content"][1]
        self.assertEqual(set(image_part), {"type", "image_url"})
        self.assertEqual(set(image_part["image_url"]), {"url"})
        self.assertTrue(
            image_part["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        serialized_keys = {
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "detail",
        }
        self.assertTrue(serialized_keys.isdisjoint(request))
        self.assertNotIn("detail", image_part["image_url"])

    def test_reasoning_effort_transport_and_retries(self) -> None:
        """Forward explicit effort unchanged and omit null on every attempt."""
        for effort in (None, "none", "high", "custom"):
            with self.subTest(effort=effort):
                client = FakeClient([_status_error(429), _response("ok")])
                model_client.submit_messages(
                    client, "exact-model", _text_messages(), reasoning_effort=effort
                )
                requests = client.chat.completions.requests
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                if effort is None:
                    self.assertNotIn("reasoning_effort", requests[0])
                else:
                    self.assertEqual(requests[0]["reasoning_effort"], effort)
        client = FakeClient([_status_error(400)])
        with self.assertRaises(EndpointRequestError):
            model_client.submit_messages(
                client, "exact-model", _text_messages(), reasoning_effort="unsupported"
            )
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_completion_limit_transport_and_retries(self) -> None:
        """Keep the exact cap and effort across retries without fallback on rejection."""
        for limit in (None, 2048):
            client = FakeClient([_status_error(429), _response("ok")])
            model_client.submit_messages(
                client, "exact-model", _text_messages(),
                reasoning_effort="high", max_completion_tokens=limit,
            )
            requests = client.chat.completions.requests
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(requests[0]["reasoning_effort"], "high")
            if limit is None:
                self.assertNotIn("max_completion_tokens", requests[0])
            else:
                self.assertEqual(requests[0]["max_completion_tokens"], limit)
            self.assertNotIn("max_tokens", requests[0])
        client = FakeClient([_status_error(400)])
        with self.assertRaises(EndpointRequestError):
            model_client.submit_messages(
                client, "exact-model", _text_messages(), max_completion_tokens=2048,
            )
        self.assertEqual(len(client.chat.completions.requests), 1)
        self.assertEqual(client.chat.completions.requests[0]["max_completion_tokens"], 2048)

    def test_length_limited_answers_keep_existing_outcomes(self) -> None:
        """Parse nonempty text once and retry empty answers with the identical cap."""
        from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
            _response_fields,
        )

        for content, status, attempts in (
            ('{"change_label":"deterioration"}', "valid", 1),
            ('{"change_label":', "invalid_response", 1),
            (None, "response_error", 3),
        ):
            with self.subTest(content=content):
                client = FakeClient([_response(content, finish_reason="length")] * attempts)
                result = model_client.submit_messages(
                    client, "exact-model", _text_messages(), max_completion_tokens=16,
                )
                parsed = _response_fields(result, "pair")
                self.assertEqual(parsed["status"], status)
                requests = client.chat.completions.requests
                self.assertEqual(len(requests), attempts)
                self.assertTrue(all(request == requests[0] for request in requests))
                self.assertEqual(requests[0]["max_completion_tokens"], 16)

    def test_three_identical_technical_attempts_then_safe_error(self) -> None:
        """Exhaust exactly three attempts and retain only a sanitized error."""
        request = httpx.Request("POST", "http://server/v1/chat/completions")
        unsafe = APIConnectionError(
            message="data:image/png;base64,secret",
            request=request,
        )
        client = FakeClient([unsafe, unsafe, unsafe])
        result = model_client.call_chat_model(
            client,
            "exact-model",
            self.image_messages,
            self.image_root,
        )
        requests = client.chat.completions.requests
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[1], requests[2])
        self.assertEqual(result["kind"], "response_error")
        self.assertEqual(result["raw_text"], None)
        self.assertEqual(
            set(result["error"]),
            {"type", "message"},
        )
        self.assertNotIn("data:image", str(result).lower())
        self.assertNotIn("attempt", str(result["error"]).lower())
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(len(result["attempt_seconds"]), 3)
        self.assertTrue(all(seconds >= 0 for seconds in result["attempt_seconds"]))

    def test_every_approved_status_and_timeout_is_retried(self) -> None:
        """Retry timeout, 408, 429, and every server-error range."""
        request = httpx.Request("POST", "http://server/v1/chat/completions")
        errors: list[Exception] = [
            APITimeoutError(request),
            *(_status_error(status) for status in (408, 429, 500, 503, 599)),
        ]
        for index, error in enumerate(errors):
            with self.subTest(error=type(error).__name__, index=index):
                client = FakeClient([error, _response('{"answer":"none"}')])
                result = model_client.call_chat_model(
                    client,
                    "exact-model",
                    _text_messages(),
                )
                self.assertEqual(result["kind"], "text")
                self.assertEqual(result["raw_text"], '{"answer":"none"}')
                self.assertEqual(len(client.chat.completions.requests), 2)

    def test_nonretryable_statuses_stop_immediately(self) -> None:
        """Do not convert setup or validation HTTP errors into model outcomes."""
        for status in (400, 401, 403, 404, 409, 413, 415, 422):
            with self.subTest(status=status):
                error = _status_error(status)
                client = FakeClient([error])
                with self.assertRaises(EndpointRequestError):
                    model_client.call_chat_model(
                        client,
                        "exact-model",
                        _text_messages(),
                    )
                self.assertEqual(len(client.chat.completions.requests), 1)

    def test_returned_text_refusal_and_filter_are_never_retried(self) -> None:
        """Treat every returned semantic outcome as final on its first call."""
        cases = (
            (
                _response("not JSON"),
                {"kind": "text", "raw_text": "not JSON"},
            ),
            (
                _response(None, refusal="I cannot answer"),
                {"kind": "refusal", "raw_text": "I cannot answer"},
            ),
            (
                _response(None, finish_reason="content_filter"),
                {"kind": "content_filter", "raw_text": None},
            ),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                client = FakeClient([response])
                result = model_client.call_chat_model(
                    client,
                    "exact-model",
                    _text_messages(),
                )
                self.assertEqual(len(client.chat.completions.requests), 1)
                self.assertEqual(result["kind"], expected["kind"])
                self.assertEqual(result["raw_text"], expected["raw_text"])
                self.assertIsNone(result["error"])

    def test_empty_success_is_a_retryable_technical_failure(self) -> None:
        """Retry an unusable successful response without re-asking valid text."""
        client = FakeClient([_response("  "), _response('{"answer":"T2"}')])
        result = model_client.call_chat_model(
            client,
            "exact-model",
            _text_messages(),
        )
        self.assertEqual(result["kind"], "text")
        self.assertEqual(result["raw_text"], '{"answer":"T2"}')
        self.assertEqual(len(client.chat.completions.requests), 2)

        exhausted = FakeClient([SimpleNamespace(choices=[])] * 3)
        failed = model_client.call_chat_model(
            exhausted,
            "exact-model",
            _text_messages(),
        )
        self.assertEqual(failed["kind"], "response_error")
        self.assertEqual(failed["error"]["type"], "EmptyModelResponseError")
        self.assertEqual(len(exhausted.chat.completions.requests), 3)

    def test_private_paths_and_returned_payloads_fail_closed(self) -> None:
        """Reject traversal, missing roots, and returned image data URLs."""
        unsafe_messages = copy.deepcopy(self.image_messages)
        unsafe_messages[0]["content"][1]["image_id"] = "../secret.jpg"
        with self.assertRaises(ValueError):
            model_client.build_chat_request(
                "exact-model",
                unsafe_messages,
                self.image_root,
            )
        with self.assertRaises(ValueError):
            model_client.build_chat_request(
                "exact-model",
                self.image_messages,
                None,
            )

        client = FakeClient([_response("data:image/png;base64,secret")])
        with self.assertRaises(model_client.UnsafeModelResponseError):
            model_client.call_chat_model(
                client,
                "exact-model",
                _text_messages(),
            )
        self.assertEqual(len(client.chat.completions.requests), 1)


if __name__ == "__main__":
    unittest.main()

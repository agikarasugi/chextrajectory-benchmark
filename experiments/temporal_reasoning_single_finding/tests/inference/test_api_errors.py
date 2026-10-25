"""Check diagnostic usefulness and serialized privacy using synthetic HTTP failures."""

import importlib.util
import os
import pickle
import traceback
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import APIResponseValidationError, APIStatusError, AuthenticationError

from experiments.temporal_reasoning_single_finding.inference import model_client
from experiments.temporal_reasoning_single_finding.inference.api_errors import EndpointRequestError
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import _submit_spec
from experiments.temporal_reasoning_single_finding.modal import endpoint_launcher
from .test_model_client import FakeClient, _text_messages


TOKEN_ID = "synthetic-proxy-token-id"
TOKEN_SECRET = "synthetic-proxy-token-secret"
TOKEN = TOKEN_ID + "." + TOKEN_SECRET
PAYLOAD = "data:image/png;base64,c3ludGhldGljLXByaXZhdGUtcGF5bG9hZA=="
COOKIE = "synthetic-cookie-secret"


def api_error(
    status: int = 400, *, poisoned: bool = False,
    parameter: str = "reasoning_effort",
) -> APIStatusError:
    """Build a token-bearing exception with synthetic bodies and debugging fields."""
    request = httpx.Request(
        "POST", "https://endpoint.invalid/v1/chat/completions?token=" + TOKEN,
        headers={"authorization": "Bearer " + TOKEN, "cookie": "session=" + COOKIE},
        content=PAYLOAD,
    )
    response = httpx.Response(
        status, request=request, headers={"x-request-id": TOKEN_SECRET if poisoned else "req-smoke-123"},
        content=PAYLOAD,
    )
    body = {
        "code": TOKEN if poisoned else "unsupported_parameter",
        "param": PAYLOAD if poisoned else parameter,
        "message": f"Unsupported parameter. {TOKEN} {PAYLOAD} {COOKIE}",
        "nested": {"private": [TOKEN_ID, TOKEN_SECRET, PAYLOAD]},
    }
    cls = AuthenticationError if status == 401 else APIStatusError
    return cls(body["message"], response=response, body=body)


class APIErrorTests(unittest.TestCase):
    """Ensure setup failures stay fatal without transporting private HTTP state."""

    def setUp(self) -> None:
        """Supply a fake credential without accessing any real environment value."""
        self.environment = patch.dict(os.environ, OPENAI_API_KEY=TOKEN)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def capture(self, call) -> EndpointRequestError:
        """Keep a propagated error and its traceback for serialization checks."""
        try:
            call()
        except EndpointRequestError as error:
            return error
        self.fail("Expected a fatal sanitized API error")

    def assert_private(self, error: EndpointRequestError) -> None:
        """Check text, attached state, pickle roundtrip, and exception chaining."""
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        self.assertTrue(error.__suppress_context__)
        for name in ("request", "response", "body", "original_error"):
            self.assertFalse(hasattr(error, name))
        serialized = pickle.dumps(error)
        restored = pickle.loads(serialized)
        self.assertEqual(restored.args, error.args)
        self.assertEqual(restored.__dict__, error.__dict__)
        exposed = "".join(traceback.format_exception(error)) + repr(error.__dict__)
        for marker in (TOKEN, TOKEN_ID, TOKEN_SECRET, COOKIE, PAYLOAD, PAYLOAD.split(",")[1]):
            self.assertNotIn(marker, exposed)
            self.assertNotIn(marker.encode(), serialized)

    def test_fatal_inference_keeps_diagnostics_without_retries(self) -> None:
        """Keep operation, status, IDs, parameter, and stack locations after one call."""
        client = FakeClient([api_error()])
        error = self.capture(lambda: model_client.call_chat_model(
            client, "synthetic-model", _text_messages(),
        ))
        self.assert_private(error)
        self.assertEqual(len(client.chat.completions.requests), 1)
        for detail in ("HTTP 400", "req-smoke-123", "reasoning_effort", "synthetic-model",
                       "endpoint.invalid", "POST /v1/chat/completions", "unsupported_parameter"):
            self.assertIn(detail, str(error))
        self.assertIn("call_chat_model", " ".join(error.__notes__))

    def test_completion_limit_rejection_keeps_safe_parameter(self) -> None:
        """Identify the rejected cap without retaining credentials or retrying."""
        client = FakeClient([api_error(parameter="max_completion_tokens")])
        error = self.capture(lambda: model_client.call_chat_model(
            client, "synthetic-model", _text_messages(), max_completion_tokens=2048,
        ))
        self.assert_private(error)
        self.assertEqual(error.parameter, "max_completion_tokens")
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_plain_server_reason_survives_transport(self) -> None:
        """Expose an actionable server message while ignoring other body fields."""
        original = api_error()
        message = "The decoder prompt is too long (length 18000, maximum 16384)."
        original.body = {"error": {"message": message, "input": PAYLOAD}}
        error = self.capture(lambda: model_client.call_chat_model(
            FakeClient([original]), "synthetic-model", _text_messages(),
        ))
        self.assert_private(error)
        self.assertIn("server_detail=" + message, str(error))
        self.assertIn(message, str(pickle.loads(pickle.dumps(error))))

    def test_server_detail_redacts_credentials_and_bare_encoded_values(self) -> None:
        """Keep the reason while filtering secrets and opaque payload strings."""
        original = api_error()
        original.body = {"message": (
            "Invalid configuration. " + TOKEN + " " + COOKIE + " "
            + PAYLOAD.split(",")[1] + " https://example.invalid/?key=private"
        )}
        error = self.capture(lambda: model_client.call_chat_model(
            FakeClient([original]), "synthetic-model", _text_messages(),
        ))
        self.assert_private(error)
        self.assertIn("server_detail=Invalid configuration.", str(error))
        self.assertNotIn("key=private", str(error))

    def test_contaminated_detail_is_rejected_before_truncation(self) -> None:
        """Suppress image-bearing or echoed input messages even after a long prefix."""
        for message in (
            "Invalid input: " + PAYLOAD,
            "Error: {'messages': [{'content': 'private input'}]}",
            "Failure. " * 200 + PAYLOAD,
            "x" * 16385,
        ):
            with self.subTest(message_length=len(message)):
                original = api_error()
                original.body = {"message": message}
                error = self.capture(lambda: model_client.call_chat_model(
                    FakeClient([original]), "synthetic-model", _text_messages(),
                ))
                self.assert_private(error)
                self.assertIn("Server detail omitted:", str(error))
                self.assertNotIn("private input", str(error))

    def test_poisoned_metadata_and_unrecognized_body_are_omitted(self) -> None:
        """Drop secret-bearing header values, free text, and unsafe metadata fields."""
        original = api_error(401, poisoned=True)
        client = FakeClient([original])
        error = self.capture(lambda: model_client.call_chat_model(
            client, TOKEN, _text_messages(),
        ))
        self.assert_private(error)
        self.assertIsNone(error.request_id)
        self.assertIsNone(error.code)
        self.assertIsNone(error.parameter)
        self.assertEqual(error.error_type, "AuthenticationError")
        self.assertIn("Raw server message", str(error))

    def test_header_credentials_are_filtered_without_environment_key(self) -> None:
        """Protect explicit client keys and cookie values independently of the env."""
        original = api_error(poisoned=True)
        original.request_id = COOKIE
        with patch.dict(os.environ, OPENAI_API_KEY=""):
            error = self.capture(lambda: model_client.call_chat_model(
                FakeClient([original]), TOKEN_SECRET, _text_messages(),
            ))
        self.assert_private(error)
        self.assertIsNone(error.request_id)

    def test_discovery_and_readiness_preserve_safe_failure(self) -> None:
        """Protect every discovery caller, including the readiness polling boundary."""
        for mode in ("discover", "exact", "preflight", "readiness"):
            with self.subTest(mode=mode):
                client = Mock()
                client.models.list.side_effect = api_error(401)
                client.__enter__ = Mock(return_value=client)
                client.__exit__ = Mock(return_value=False)
                client.with_options.return_value = client
                if mode == "discover":
                    call = lambda: model_client.discover_model_ids(client)
                elif mode == "exact":
                    call = lambda: model_client.require_exact_model(client, "synthetic-model")
                elif mode == "preflight":
                    call = lambda: model_client.discover_models("https://endpoint.invalid/v1")
                else:
                    call = lambda: endpoint_launcher.wait_ready(
                        "https://endpoint.invalid/v1", "synthetic-model", 10,
                    )
                with patch.object(model_client, "create_client", return_value=client), patch.object(
                    endpoint_launcher, "create_client", return_value=client,
                ):
                    error = self.capture(call)
                self.assert_private(error)
                self.assertEqual(client.models.list.call_count, 1)
                self.assertIn("GET /v1/models", str(error))

    def test_sdk_response_validation_error_is_also_sanitized(self) -> None:
        """Protect fatal SDK errors outside APIStatusError without retrying them."""
        original = api_error()
        invalid = APIResponseValidationError(
            response=original.response, body=original.body, message=TOKEN,
        )
        client = FakeClient([invalid])
        error = self.capture(lambda: model_client.call_chat_model(
            client, "synthetic-model", _text_messages(),
        ))
        self.assert_private(error)
        self.assertEqual(error.error_type, "APIResponseValidationError")
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_benchmark_identity_is_available_on_worker_failure(self) -> None:
        """Identify the failed prepared request without copying its private inputs."""
        identifier = "a" * 64
        spec = {"safe_model_input": _text_messages(), "model_input_id": identifier}
        client = FakeClient([api_error()])
        error = self.capture(lambda: _submit_spec(spec, client, "synthetic-model", None))
        self.assert_private(error)
        self.assertIn(f"Benchmark model_input_id={identifier}", error.__notes__)
        self.assertEqual(len(client.chat.completions.requests), 1)

    def test_unrelated_programming_errors_keep_their_identity(self) -> None:
        """Leave non-SDK failures visible and avoid broad exception suppression."""
        original = ValueError("synthetic implementation defect")
        client = FakeClient([original])
        with self.assertRaises(ValueError) as raised:
            model_client.call_chat_model(client, "synthetic-model", _text_messages())
        self.assertIs(raised.exception, original)
        self.assertEqual(len(client.chat.completions.requests), 1)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "Optional Modal SDK is not installed")
    def test_actual_modal_serialization_and_traceback_are_private(self) -> None:
        """Exercise the installed SDK's exception and traceback transport offline."""
        from modal._serialization import deserialize, pickle_exception, pickle_traceback

        error = self.capture(lambda: model_client.call_chat_model(
            FakeClient([api_error()]), "synthetic-model", _text_messages(),
        ))
        self.assert_private(error)
        payload = pickle_exception(error)
        restored = deserialize(payload, None)
        self.assertIsInstance(restored, EndpointRequestError)
        self.assertEqual(restored.__dict__, error.__dict__)
        trace, lines = pickle_traceback(error, "synthetic-task")
        representations = repr(deserialize(trace, None)) + repr(deserialize(lines, None))
        for marker in (TOKEN, TOKEN_ID, TOKEN_SECRET, COOKIE, PAYLOAD, PAYLOAD.split(",")[1]):
            self.assertNotIn(marker.encode(), payload + trace + lines)
            self.assertNotIn(marker, representations)

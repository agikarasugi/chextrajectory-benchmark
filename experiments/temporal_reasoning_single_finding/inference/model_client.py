"""Privacy-safe OpenAI-compatible Chat Completions transport."""

import base64
import os
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from ..artifact_io import safe_relative_path
from ..protocol.prompts import ROLE_INSTRUCTION
from .api_errors import sanitized_api_error


REQUEST_TIMEOUT_SECONDS = 300.0
MAX_ATTEMPTS = 3


class EmptyModelResponseError(RuntimeError):
    """Signal a successful response with no usable assistant result."""


class UnsafeModelResponseError(RuntimeError):
    """Signal returned text that would violate image-payload retention rules."""


def create_client(
    base_url: str, api_key: str | None = None, *,
    timeout: int | float = REQUEST_TIMEOUT_SECONDS,
) -> OpenAI:
    """Create a Chat Completions client with SDK retries disabled."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    selected_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if not selected_key:
        raise EnvironmentError("OPENAI_API_KEY is required")
    return OpenAI(
        api_key=selected_key,
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        max_retries=0,
    )


def discover_model_ids(client: OpenAI, model: str | None = None) -> tuple[str, ...]:
    """Return the sorted unique model IDs advertised by the endpoint."""
    failure = None
    try:
        response = client.models.list()
    except APIError as error:
        failure = sanitized_api_error(error, operation="GET /v1/models", model=model)
    # Raise outside the handler so the raw exception is not retained as context.
    if failure is not None:
        raise failure from None
    data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise ValueError("The model-list response has no data array")
    model_ids = sorted(
        {
            model_id
            for model in data
            if isinstance((model_id := getattr(model, "id", None)), str)
            and model_id
        }
    )
    if not model_ids:
        raise ValueError("The server advertised no model IDs")
    return tuple(model_ids)


def discover_models(
    base_url: str,
    api_key: str | None = None,
) -> tuple[str, ...]:
    """Create a zero-retry client and return its advertised model IDs."""
    return discover_model_ids(create_client(base_url, api_key))


def require_exact_model(client: OpenAI, configured_model: str) -> str:
    """Require the exact configured identifier in endpoint discovery."""
    if not isinstance(configured_model, str) or not configured_model:
        raise ValueError("Configured model must be a non-empty string")
    model_ids = discover_model_ids(client, configured_model)
    if configured_model not in model_ids:
        raise ValueError(
            f"Configured model {configured_model!r} is not advertised exactly; "
            f"server model IDs: {', '.join(model_ids)}"
        )
    return configured_model


def _runtime_png(image_root: Path, image_id: str) -> Path:
    """Resolve one source JPG identity to its runtime PNG file."""
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a non-empty string")
    source_path = Path(image_id)
    if source_path.suffix != ".jpg":
        raise ValueError("image_id must have the exact .jpg source suffix")
    runtime_path = source_path.with_suffix(".png").as_posix()
    resolved = safe_relative_path(image_root, runtime_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Runtime PNG does not exist: {resolved}")
    if resolved.suffix != ".png" or resolved.stat().st_size <= 0:
        raise ValueError(f"Runtime image is not a non-empty PNG: {resolved}")
    return resolved


def _image_data_url(path: Path) -> str:
    """Encode one validated runtime PNG only for the in-memory request."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _materialize_messages(
    safe_messages: list[dict[str, Any]],
    image_root: Path | None,
) -> list[dict[str, Any]]:
    """Replace safe image references with private in-memory data URLs."""
    if not isinstance(safe_messages, list) or len(safe_messages) != 1:
        raise ValueError("Model input must contain exactly one user message")
    user = safe_messages[0]
    if set(user) != {"role", "content"} or user.get("role") != "user":
        raise ValueError("Model input has an invalid user message")

    content = user["content"]
    if isinstance(content, str):
        if not content.startswith(ROLE_INSTRUCTION + "\n\n"):
            raise ValueError("User content must start with the role instruction")
        return [{"role": "user", "content": content}]
    if not isinstance(content, list) or not content:
        raise ValueError("Multimodal user content must be a non-empty list")

    first = content[0]
    if (
        not isinstance(first, dict)
        or first.get("type") != "text"
        or not isinstance(first.get("text"), str)
        or not first["text"].startswith(ROLE_INSTRUCTION + "\n\n")
    ):
        raise ValueError("User content must start with the role instruction")

    materialized: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Every user content part must be an object")
        part_type = part.get("type")
        if part_type == "text" and set(part) == {"type", "text"}:
            if not isinstance(part["text"], str):
                raise ValueError("Text content must be a string")
            materialized.append(dict(part))
            continue
        if part_type == "image_ref" and set(part) == {"type", "image_id"}:
            if image_root is None:
                raise ValueError("An image root is required for image input")
            path = _runtime_png(image_root, part["image_id"])
            materialized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(path)},
                }
            )
            continue
        raise ValueError(f"Unsupported safe model-input part: {part_type}")
    return [{"role": "user", "content": materialized}]


def build_chat_request(
    model: str,
    safe_messages: list[dict[str, Any]],
    image_root: Path | None = None,
    *,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the only approved Chat Completions request fields."""
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    request = {
        "model": model,
        "messages": _materialize_messages(safe_messages, image_root),
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        request["max_completion_tokens"] = max_completion_tokens
    return request


def _clear_image_urls(request: dict[str, Any]) -> None:
    """Remove every private data URL from the request object in place."""
    for message in request.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                part["image_url"]["url"] = ""


def _reject_returned_image_payload(text: str) -> None:
    """Fail closed before returned image data could enter a saved record."""
    if "data:image/" in text.casefold():
        raise UnsafeModelResponseError(
            "Model response unexpectedly contained an image data URL"
        )


def _extract_usage(response: Any) -> dict[str, int | None] | None:
    """Retain only endpoint-reported nonnegative integer token counts."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    counts = {}
    for field, source in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, source, None)
        counts[field] = value if type(value) is int and value >= 0 else None
    return counts


def _extract_response(response: Any) -> dict[str, Any]:
    """Extract text, explicit refusal, or content-filter completion."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or len(choices) != 1:
        raise EmptyModelResponseError(
            "Chat Completions response did not contain exactly one choice"
        )
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise EmptyModelResponseError(
            "Chat Completions response had no assistant message"
        )
    content = getattr(message, "content", None)
    refusal = getattr(message, "refusal", None)
    for value in (content, refusal):
        if isinstance(value, str):
            _reject_returned_image_payload(value)

    if getattr(choice, "finish_reason", None) == "content_filter":
        raw_text = content if isinstance(content, str) and content else None
        return {
            "kind": "content_filter",
            "raw_text": raw_text,
            "error": None,
        }
    if isinstance(refusal, str) and refusal.strip():
        return {
            "kind": "refusal",
            "raw_text": refusal,
            "error": None,
        }
    if isinstance(content, str) and content.strip():
        return {
            "kind": "text",
            "raw_text": content,
            "error": None,
        }
    raise EmptyModelResponseError(
        "Chat Completions response had no usable text or refusal"
    )


def _retryable(error: Exception) -> bool:
    """Apply the exact frozen retry classification."""
    if isinstance(error, EmptyModelResponseError):
        return True
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        status = error.status_code
        return status in {408, 429} or 500 <= status <= 599
    return False


def _sanitized_error(error: Exception) -> dict[str, str]:
    """Return only a safe final technical error type and message."""
    if isinstance(error, EmptyModelResponseError):
        message = "The model endpoint returned no usable assistant response."
    elif isinstance(error, APIConnectionError):
        message = "The model endpoint connection failed or timed out."
    elif isinstance(error, APIStatusError):
        message = f"The model endpoint returned retryable HTTP {error.status_code}."
    else:
        message = "The model request failed without retaining request data."
    return {"type": type(error).__name__, "message": message}


def call_chat_model(
    client: OpenAI,
    model: str,
    safe_messages: list[dict[str, Any]],
    image_root: Path | None = None,
    *,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Submit one request with exactly two manual technical retries."""
    request = build_chat_request(
        model, safe_messages, image_root, reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )
    final_error: Exception | None = None
    attempt_seconds: list[float] = []
    started = time.perf_counter()
    try:
        for _attempt in range(MAX_ATTEMPTS):
            try:
                attempt_started = time.perf_counter()
                try:
                    response = client.chat.completions.create(**request)
                    result = _extract_response(response)
                    result["usage"] = _extract_usage(response)
                finally:
                    attempt_seconds.append(time.perf_counter() - attempt_started)
                result["attempt_count"] = len(attempt_seconds)
                result["attempt_seconds"] = attempt_seconds
                result["request_seconds"] = time.perf_counter() - started
                return result
            except (APIError, EmptyModelResponseError) as error:
                if not _retryable(error):
                    failure = sanitized_api_error(
                        error, operation="POST /v1/chat/completions", model=model,
                    )
                else:
                    final_error = error
                    continue
            # Classification above uses the original SDK type and adds no retries.
            raise failure from None
        if final_error is None:
            raise AssertionError("Retry loop ended without a result or error")
        return {
            "kind": "response_error",
            "raw_text": None,
            "error": _sanitized_error(final_error),
            "usage": None,
            "request_seconds": time.perf_counter() - started,
            "attempt_count": len(attempt_seconds),
            "attempt_seconds": attempt_seconds,
        }
    finally:
        _clear_image_urls(request)


def submit_messages(
    client: OpenAI,
    model: str,
    safe_messages: list[dict[str, Any]],
    image_root: Path | None = None,
    *,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Submit safe messages and return the frozen transport outcome shape."""
    return call_chat_model(
        client, model, safe_messages, image_root, reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
    )

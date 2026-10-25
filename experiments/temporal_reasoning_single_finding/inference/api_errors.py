"""Build useful API diagnostics without retaining HTTP objects or payloads."""

import os
import re
import traceback
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from openai import APIConnectionError, APIError


class EndpointRequestError(RuntimeError):
    """Carry only bounded scalar diagnostics across process and cloud boundaries."""

    error_type: str
    status_code: int | None
    request_id: str | None
    code: str | None
    parameter: str | None
    is_connection_error: bool


def _server_detail(error: APIError, secrets: set[str]) -> str:
    """Extract bounded error text without exposing echoed requests or payloads."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        source = nested if isinstance(nested, dict) else body
        message = source.get("message")
    else:
        message = error.message
    if not isinstance(message, str) or not message.strip():
        return "Server supplied no textual error detail."
    if len(message) > 16384:
        return "Server detail omitted: oversized message may contain echoed input."
    # Reject contaminated messages before truncation. Never retain an excerpt of
    # an image-bearing error or stringify the surrounding error-body structure.
    unsafe = re.search(
        r"data\s*:|base64|image_url|image_ref|data_url|"
        r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|"
        r"[\"'](?:messages|prompt|input|content|image|images|url)[\"']\s*:",
        message, re.IGNORECASE,
    )
    if unsafe:
        return "Server detail omitted: message contains possible echoed input or image data."
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    message = re.sub(r"https?://\S+", "[redacted URL]", message)
    message = re.sub(
        r"(?i)\b(?:bearer\s+|(?:api[_-]?key|token|secret|password)\s*[=:]\s*)\S+",
        "[redacted credential]", message,
    )
    # Also suppress opaque strings such as bare encoded data or unknown tokens.
    message = re.sub(r"[A-Za-z0-9_+/=-]{32,}", "[redacted opaque value]", message)
    message = " ".join(message.split())
    message = "".join(char if 32 <= ord(char) < 127 else "?" for char in message)
    return message[:1000] + (" [truncated]" if len(message) > 1000 else "")


def sanitized_api_error(
    error: APIError, *, operation: str, model: str | None = None,
) -> EndpointRequestError:
    """Keep filtered server diagnostics without retaining private HTTP state."""
    secrets = {os.environ.get("OPENAI_API_KEY", "")}
    request = error.request
    secrets.update(value for _, value in parse_qsl(urlsplit(str(request.url)).query) if value)
    for name in ("authorization", "proxy-authorization", "x-api-key", "cookie"):
        value = request.headers.get(name, "")
        if value:
            secrets.add(value)
            if name in {"authorization", "proxy-authorization"}:
                secrets.add(value.split(" ", 1)[-1])
            if name == "cookie":
                secrets.update(item.partition("=")[2].strip() for item in value.split(";"))
    # Modal proxy credentials combine an ID and secret with a period.
    secrets.update(part for value in tuple(secrets) for part in value.split(".") if len(part) >= 8)
    secrets.discard("")

    def identifier(value: object, pattern: str, limit: int = 160) -> str | None:
        """Accept bounded metadata only when it contains no known credential."""
        if not isinstance(value, str) or not value or len(value) > limit:
            return None
        if not re.fullmatch(pattern, value) or any(secret in value for secret in secrets):
            return None
        return value

    status = getattr(error, "status_code", None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    error_type = type(error).__name__
    # These values are useful selectors, never free-form error-body excerpts.
    codes = {
        "invalid_api_key", "authentication_error", "permission_denied",
        "model_not_found", "invalid_request_error", "invalid_value",
        "unsupported_parameter", "unsupported_value", "context_length_exceeded",
        "rate_limit_exceeded", "insufficient_quota", "server_error",
    }
    code = getattr(error, "code", None)
    code = code if isinstance(code, str) and code in codes else None
    code = identifier(code, r"[a-z_]+")
    parameter = identifier(
        getattr(error, "param", None),
        r"(?:model|messages|reasoning_effort|max_tokens|max_completion_tokens|temperature|"
        r"top_p|response_format|stream)(?:\[\d{1,4}\]|\.(?:content|image_url|url|type|role))*",
    )
    request_id = identifier(getattr(error, "request_id", None), r"[A-Za-z0-9_-]+")
    host = identifier(urlsplit(str(request.url)).hostname, r"[A-Za-z0-9.:-]+")
    model = identifier(model, r"[A-Za-z0-9_./:@+-]+")
    diagnostic = {
        400: "Request rejected; check model support and request settings.",
        401: "Authentication failed; check the endpoint Secret and token.",
        403: "Access denied; check token permissions and endpoint environment.",
        404: "Route or model not found; check the API URL and model identifier.",
        413: "Request too large; check image count and endpoint request limits.",
        415: "Unsupported media type.",
        422: "Request validation failed; check model support and request settings.",
        429: "Rate limit or quota exceeded.",
    }.get(status, "Endpoint request failed.")
    if isinstance(error, APIConnectionError):
        diagnostic = "Endpoint connection failed or timed out."
    # Recognize common causes without copying any server-supplied text.
    message = error.message[:2048].casefold()
    if status in {400, 413, 422} and (
        code == "context_length_exceeded" or "context length" in message
    ):
        diagnostic = "Model context length exceeded; check input and output token budgets."
    elif status in {400, 422} and (
        code == "unsupported_parameter" or "unsupported parameter" in message
    ):
        diagnostic = "Endpoint does not support a supplied request parameter."
    fields = [f"{error_type} (HTTP {status})" if status else error_type, diagnostic]
    for name, value in (("operation", operation), ("host", host), ("model", model),
                        ("code", code), ("parameter", parameter), ("request_id", request_id)):
        if value:
            fields.append(f"{name}={value}")
    fields.append("server_detail=" + _server_detail(error, secrets))
    fields.append("Raw server message and HTTP bodies are not retained.")
    safe = EndpointRequestError(" | ".join(fields))
    safe.error_type = error_type
    safe.status_code = status
    safe.request_id = request_id
    safe.code = code
    safe.parameter = parameter
    safe.is_connection_error = isinstance(error, APIConnectionError)
    locations = []
    for frame, lineno in traceback.walk_tb(error.__traceback__):
        filename = identifier(Path(frame.f_code.co_filename).name, r"[A-Za-z0-9_.-]+")
        function = identifier(frame.f_code.co_name, r"[A-Za-z0-9_<>]+")
        if filename and function:
            locations.append(f"{filename}:{lineno} in {function}")
    if locations:
        safe.add_note("API stack (locations only): " + " -> ".join(locations[-6:]))
    # No request, response, body, original exception, or traceback is attached.
    return safe

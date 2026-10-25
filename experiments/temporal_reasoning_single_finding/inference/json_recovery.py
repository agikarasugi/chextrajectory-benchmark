"""Recover and strictly validate one frozen JSON response object."""

import json
import re
from collections.abc import Callable
from typing import Any

from ..protocol.contracts import (
    CHANGE_LABELS,
    answer_choices,
    interval_keys,
)


class DuplicateJsonKeyError(ValueError):
    """Signal a duplicate object key anywhere in a response."""


class NonStandardJsonConstantError(ValueError):
    """Signal NaN or Infinity in otherwise parseable JSON."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct an object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject non-standard JSON numeric constants."""
    raise NonStandardJsonConstantError(value)


def _loads(value: str) -> Any:
    """Decode strict JSON with duplicate-key and constant rejection."""
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def _success(parsed: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return one accepted parser result."""
    return {
        "valid": True,
        "parsed_output": parsed,
        "parsing_mode": mode,
        "failure_reason": None,
    }


def _failure(reason: str) -> dict[str, Any]:
    """Return one rejected parser result without partially repaired output."""
    return {
        "valid": False,
        "parsed_output": None,
        "parsing_mode": None,
        "failure_reason": reason,
    }


def _object_spans(value: str) -> tuple[list[tuple[int, int]], bool]:
    """Locate balanced top-level object spans while respecting JSON strings."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, character in enumerate(value):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
                start = None

    return spans, start is not None


def _inside_markdown_fence(value: str, start: int, end: int) -> bool:
    """Return whether a candidate is enclosed by a complete triple-backtick fence."""
    markers = [match.start() for match in re.finditer(r"```", value)]
    for marker_index in range(0, len(markers) - 1, 2):
        opening = markers[marker_index]
        closing = markers[marker_index + 1]
        if opening + 3 <= start and end <= closing:
            return True
    return False


def recover_json_object(
    raw_text: str,
    validator: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Recover exactly one unambiguous schema-valid JSON object."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    trimmed = raw_text.strip()
    if not trimmed:
        return _failure("no_json_object")

    try:
        exact = _loads(trimmed)
    except DuplicateJsonKeyError:
        return _failure("duplicate_json_key")
    except (json.JSONDecodeError, NonStandardJsonConstantError):
        pass
    else:
        if isinstance(exact, dict) and validator(exact):
            return _success(exact, "exact")
        return _failure("schema_mismatch")

    spans, unterminated = _object_spans(raw_text)
    if len(spans) > 1:
        return _failure("multiple_json_objects")
    if unterminated:
        return _failure("malformed_json")
    if not spans:
        return _failure("no_json_object")

    start, end = spans[0]
    try:
        candidate = _loads(raw_text[start:end])
    except DuplicateJsonKeyError:
        return _failure("duplicate_json_key")
    except (json.JSONDecodeError, NonStandardJsonConstantError):
        return _failure("malformed_json")
    if not isinstance(candidate, dict) or not validator(candidate):
        return _failure("schema_mismatch")
    mode = (
        "fenced"
        if _inside_markdown_fence(raw_text, start, end)
        else "surrounding_text"
    )
    return _success(candidate, mode)


def _pair_schema(value: dict[str, Any]) -> bool:
    """Validate the exact PAIR response object."""
    return (
        set(value) == {"change_label"}
        and isinstance(value["change_label"], str)
        and value["change_label"] in CHANGE_LABELS
    )


def parse_pair_response(raw_text: str) -> dict[str, Any]:
    """Parse one exact PAIR change-label response."""
    return recover_json_object(raw_text, _pair_schema)


def parse_seq_response(
    raw_text: str,
    timepoint_count: int,
) -> dict[str, Any]:
    """Parse one exact 3TP or 4TP SEQ change-label response."""
    expected_intervals = set(interval_keys(timepoint_count))

    def validate(value: dict[str, Any]) -> bool:
        """Validate the nested interval object for this sequence length."""
        if set(value) != {"change_labels"}:
            return False
        labels = value["change_labels"]
        return (
            isinstance(labels, dict)
            and set(labels) == expected_intervals
            and all(
                isinstance(label, str) and label in CHANGE_LABELS
                for label in labels.values()
            )
        )

    return recover_json_object(raw_text, validate)


def parse_final_response(
    raw_text: str,
    task: str,
    timepoint_count: int,
) -> dict[str, Any]:
    """Parse one exact final-answer response for its task vocabulary."""
    allowed = set(answer_choices(task, timepoint_count))

    def validate(value: dict[str, Any]) -> bool:
        """Validate the exact final-answer object for this task."""
        return (
            set(value) == {"answer"}
            and isinstance(value["answer"], str)
            and value["answer"] in allowed
        )

    return recover_json_object(raw_text, validate)


def parse_response(
    raw_text: str,
    response_type: str,
    *,
    timepoint_count: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Parse one PAIR, SEQ, or final response into saved-record fields."""
    if response_type == "pair":
        if timepoint_count is not None or task is not None:
            raise ValueError("PAIR parsing does not accept task or timepoint_count")
        result = parse_pair_response(raw_text)
    elif response_type == "seq":
        if timepoint_count is None or task is not None:
            raise ValueError("SEQ parsing requires only timepoint_count")
        result = parse_seq_response(raw_text, timepoint_count)
    elif response_type == "final":
        if timepoint_count is None or task is None:
            raise ValueError("Final parsing requires task and timepoint_count")
        result = parse_final_response(raw_text, task, timepoint_count)
    else:
        raise ValueError(f"Unsupported response type: {response_type}")

    return {
        "status": "valid" if result["valid"] else "invalid_response",
        "parsed_output": result["parsed_output"],
        "parsing_mode": result["parsing_mode"],
        "failure_reason": result["failure_reason"],
    }

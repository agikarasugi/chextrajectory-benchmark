"""Deterministic, atomic, and privacy-safe artifact helpers."""

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    """Return the repository root containing the experiments package."""
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without retaining its content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    """Reject JSON numeric constants outside the standard."""
    raise ValueError(f"Non-standard JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a mapping while rejecting duplicate JSON keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    """Read strict UTF-8 JSON, rejecting duplicate keys and numeric constants."""
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict non-empty JSON Lines rows."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}")
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    """Atomically replace one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    """Write deterministic indented UTF-8 JSON atomically."""
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic compact JSON Lines atomically."""
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_text(path, text)


def write_nested_json(path: Path, value: dict[str, Any]) -> None:
    """Write ordered nested JSON atomically, keeping scalar metric rows compact.

    Callers supply deterministic entity/key order. Existing artifact serializers
    retain their original sorted-key formatting.
    """
    def render(node: Any, depth: int) -> str:
        """Indent containers and keep scalar-only objects on a single line."""
        if isinstance(node, dict) and node:
            if all(not isinstance(child, (dict, list)) for child in node.values()):
                return json.dumps(node, ensure_ascii=False, allow_nan=False)
            rows = [
                "  " * (depth + 1)
                + json.dumps(key, ensure_ascii=False)
                + ": "
                + render(child, depth + 1)
                for key, child in node.items()
            ]
            return "{\n" + ",\n".join(rows) + "\n" + "  " * depth + "}"
        return json.dumps(node, ensure_ascii=False, allow_nan=False)

    _atomic_text(path, render(value, 0) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append and durably flush one compact JSON Lines record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_relative_path(root: Path, relative_path: str) -> Path:
    """Resolve a non-traversing relative path beneath an absolute root."""
    if not root.is_absolute():
        raise ValueError(f"Root must be absolute: {root}")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative path: {relative_path}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path leaves configured root: {relative_path}")
    return resolved


def assert_no_image_payload(value: Any) -> None:
    """Reject Base64 image URLs or image-byte fields in a saved artifact."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if lowered in {"image_bytes", "base64", "image_url"}:
                raise ValueError(f"Image payload field cannot be saved: {key}")
            assert_no_image_payload(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_image_payload(child)
    elif isinstance(value, str) and value.lstrip().lower().startswith("data:image/"):
        raise ValueError("Image data URL cannot be saved")

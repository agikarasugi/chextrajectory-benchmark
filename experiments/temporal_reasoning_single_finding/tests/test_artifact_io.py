"""Tests for deterministic artifact helpers."""

import tempfile
import unittest
from pathlib import Path

from experiments.temporal_reasoning_single_finding.artifact_io import (
    append_jsonl,
    assert_no_image_payload,
    read_json,
    read_jsonl,
    safe_relative_path,
    sha256_file,
    write_json,
    write_jsonl,
)


class ArtifactIoTests(unittest.TestCase):
    """Exercise strict JSON, atomic writes, and path privacy."""

    def test_deterministic_json_and_jsonl(self) -> None:
        """Stable writers reproduce identical bytes and rows."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            value = {"z": [2, 1], "a": "text"}
            write_json(first, value)
            write_json(second, value)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(read_json(first), value)
            rows = root / "rows.jsonl"
            write_jsonl(rows, [{"b": 2, "a": 1}])
            append_jsonl(rows, {"c": 3})
            self.assertEqual(read_jsonl(rows), [{"a": 1, "b": 2}, {"c": 3}])
            self.assertEqual(len(sha256_file(first)), 64)

    def test_strict_json_and_safe_paths(self) -> None:
        """Duplicate keys, traversal, and image payloads are rejected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_json(duplicate)
            self.assertEqual(
                safe_relative_path(root, "train/a.png"),
                root / "train/a.png",
            )
            with self.assertRaises(ValueError):
                safe_relative_path(root, "../outside.png")
            with self.assertRaises(ValueError):
                assert_no_image_payload({"value": "data:image/png;base64,AAAA"})


if __name__ == "__main__":
    unittest.main()

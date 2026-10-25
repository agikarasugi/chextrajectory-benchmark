"""Tests for strict experiment and private-path configuration."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.inference.experiment_config import (
    load_experiment,
    load_private_paths,
    require_api_key,
    resolve_benchmark_path,
    run_directory,
    validate_worker_count,
)


class ExperimentConfigTests(unittest.TestCase):
    """Exercise the complete frozen configuration surface."""

    def setUp(self) -> None:
        """Create one isolated repository-like directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {
            "run_name": "temporal-single-v1-qwen-001",
            "model": "unsloth/Qwen3.5-4B-GGUF",
            "base_url": "http://100.64.0.4:8000/v1",
            "benchmark_path": "artifacts/benchmarks/smoke/manifest.json",
        }

    def tearDown(self) -> None:
        """Remove the isolated directory."""
        self.temporary.cleanup()

    def _write(self, name: str, value: object) -> Path:
        """Write one JSON fixture."""
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_request_timeout_is_validated_and_applied(self) -> None:
        """Carry a per-run timeout into the SDK without changing retry policy."""
        from experiments.temporal_reasoning_single_finding.inference import model_client

        value = {**self.config, "request_timeout_seconds": 120}
        loaded = load_experiment(self._write("timeout.json", value))
        with patch.object(model_client, "OpenAI") as sdk:
            model_client.create_client(
                loaded["base_url"], "test-key",
                timeout=loaded["request_timeout_seconds"],
            )
        self.assertEqual(sdk.call_args.kwargs["timeout"], 120)
        self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)
        for invalid in (0, -1, True, "120"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_experiment(self._write(
                    "invalid-timeout.json", {**value, "request_timeout_seconds": invalid},
                ))

    def test_load_and_resolve_exact_configuration(self) -> None:
        """Preserve values and resolve only repository-relative paths."""
        loaded = load_experiment(self._write("experiment.json", self.config))
        self.assertEqual(loaded, self.config)
        self.assertEqual(
            resolve_benchmark_path(loaded, root=self.root),
            (self.root / self.config["benchmark_path"]).resolve(),
        )
        self.assertEqual(
            run_directory(loaded, root=self.root),
            (
                self.root
                / "artifacts/temporal_reasoning_single_finding/runs"
                / self.config["run_name"]
            ).resolve(),
        )

        absolute = dict(self.config, benchmark_path=str(self.root / "manifest.json"))
        self.assertEqual(
            resolve_benchmark_path(absolute, root=self.root / "ignored"),
            (self.root / "manifest.json").resolve(),
        )

    def test_optional_reasoning_effort(self) -> None:
        """Normalize null, preserve endpoint-specific levels, and reject bad types."""
        loaded = load_experiment(self._write(
            "null.json", {**self.config, "reasoning_effort": None}
        ))
        self.assertEqual(loaded, self.config)
        for effort in ("none", "low", "medium", "high", "xhigh", "max", "custom"):
            config = {**self.config, "reasoning_effort": effort}
            self.assertEqual(load_experiment(self._write("effort.json", config)), config)
        for effort in ("", " high", "high ", False, 1, [], {}):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                load_experiment(self._write(
                    "bad-effort.json", {**self.config, "reasoning_effort": effort}
                ))

    def test_optional_completion_limit(self) -> None:
        """Accept only positive integers and normalize null without accepting aliases."""
        for limit in (None, 1, 2048, 10**12):
            config = dict(self.config, reasoning_effort="high", max_completion_tokens=limit)
            expected = dict(config)
            if limit is None:
                expected.pop("max_completion_tokens")
            self.assertEqual(load_experiment(self._write("limit.json", config)), expected)
        for limit in (0, -1, True, False, 1.0, "2048", [], {}):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integer"):
                load_experiment(self._write(
                    "bad-limit.json", dict(self.config, max_completion_tokens=limit)
                ))
        with self.assertRaises(ValueError):
            load_experiment(self._write("alias.json", dict(self.config, max_tokens=2048)))

    def test_missing_extra_and_non_object_configs_fail(self) -> None:
        """Reject every departure from the exact four-field object."""
        for name, value in (
            (
                "missing",
                {
                    key: value
                    for key, value in self.config.items()
                    if key != "model"
                },
            ),
            ("extra", {**self.config, "temperature": 0}),
            ("array", [self.config]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_experiment(self._write(f"{name}.json", value))

    def test_run_name_strings_and_base_url_are_strict(self) -> None:
        """Reject unsafe names, padding, empty values, and non-API URLs."""
        replacements = {
            "uppercase": ("run_name", "Uppercase"),
            "slash": ("run_name", "bad/name"),
            "empty_model": ("model", ""),
            "padded_model": ("model", " model"),
            "bad_url": ("base_url", "localhost:8000/v1"),
            "query_url": ("base_url", "http://localhost:8000/v1?x=1"),
            "padded_path": ("benchmark_path", " manifest.json"),
        }
        for name, (field, value) in replacements.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_experiment(
                    self._write(f"{name}.json", {**self.config, field: value})
                )

    def test_duplicate_keys_and_nonstandard_constants_fail(self) -> None:
        """Use strict JSON decoding for human-authored configuration."""
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"run_name":"a","run_name":"b","model":"m",'
            '"base_url":"http://localhost/v1","benchmark_path":"x"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            load_experiment(duplicate)

        constant = self.root / "constant.json"
        constant.write_text(
            json.dumps(self.config)[:-1] + ',"extra":NaN}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Non-standard JSON constant"):
            load_experiment(constant)

    def test_load_private_image_root(self) -> None:
        """Require exactly one absolute existing image directory."""
        images = self.root / "PNG"
        images.mkdir()
        loaded = load_private_paths(
            self._write(
                "paths.json",
                {"chexpert_plus_image_root": str(images)},
            )
        )
        self.assertEqual(loaded, {"chexpert_plus_image_root": images.resolve()})

        for name, value in (
            ("relative", {"chexpert_plus_image_root": "PNG"}),
            ("missing", {}),
            (
                "extra",
                {
                    "chexpert_plus_image_root": str(images),
                    "mimic_cxr_image_root": str(images),
                },
            ),
            ("not_found", {"chexpert_plus_image_root": str(self.root / "none")}),
        ):
            with self.subTest(name=name), self.assertRaises(
                (ValueError, FileNotFoundError)
            ):
                load_private_paths(self._write(f"paths-{name}.json", value))

    def test_api_key_and_worker_count_are_runtime_only(self) -> None:
        """Read credentials from the environment and reject bad worker counts."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=True):
            self.assertEqual(require_api_key(), "secret")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                require_api_key()
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_worker_count(value)  # type: ignore[arg-type]
        validate_worker_count(1)


if __name__ == "__main__":
    unittest.main()

"""Regression coverage for model-independent job identity and CLI arguments."""

import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    repository_root,
)
from experiments.temporal_reasoning_single_finding.modal.config import (
    server_command,
    source_files,
    validate_job,
    validate_profile,
    validate_serving,
)


def example(name: str) -> dict:
    """Load an actual shipped configuration example."""
    return read_json(
        repository_root()
        / "experiments/temporal_reasoning_single_finding/examples"
        / name
    )


class ConfigurationTests(unittest.TestCase):
    """Keep custom arguments flexible while refusing identity and resource conflicts."""

    def setUp(self) -> None:
        """Use the checked-in examples as the baseline job."""
        self.serving = example("modal-serving.example.json")
        self.experiment = example("modal-experiment.example.json")
        self.profile = example("modal-profile.example.json")

    def test_examples_and_unknown_model_options(self) -> None:
        """Preserve unfamiliar long options and JSON as exact argv tokens."""
        self.serving["vllm_args"] += [
            "--custom-option",
            '{"value": "a b"}',
            "--enforce-eager",
        ]
        validated, _, _ = validate_job(self.serving, self.experiment, self.profile)
        self.assertEqual(server_command(validated)[-3:], self.serving["vllm_args"][-3:])
        self.assertEqual(server_command(validated)[2], self.serving["model_path"])

    def test_completion_limit_validation(self) -> None:
        """Preserve request limits independently from serving arguments."""
        for limit in (None, 2048):
            value = dict(self.experiment, max_completion_tokens=limit)
            serving, validated, profile = validate_job(self.serving, value, self.profile)
            expected = self.experiment if limit is None else value
            self.assertEqual(validated, expected)
            self.assertEqual((serving, profile), (self.serving, self.profile))
        for limit in (0, -1, True, 1.0, "2048", [], {}):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                validate_job(
                    self.serving, dict(self.experiment, max_completion_tokens=limit),
                    self.profile,
                )

    def test_logging_environment_validation(self) -> None:
        """Allow logging controls without accepting model or nonfinite settings."""
        settings = {"VLLM_LOGGING_LEVEL": "INFO", "VLLM_LOG_STATS_INTERVAL": "60"}
        validated = validate_serving(dict(self.serving, vllm_env=settings))
        self.assertEqual(validated["vllm_env"], settings)
        self.assertEqual(server_command(validated), server_command(self.serving))
        for invalid in (
            {"VLLM_LOG_STATS_INTERVAL": "0"},
            {"VLLM_LOG_STATS_INTERVAL": "-1"},
            {"VLLM_LOG_STATS_INTERVAL": "nan"},
            {"VLLM_LOG_STATS_INTERVAL": "inf"},
            {"VLLM_LOG_STATS_INTERVAL": 60},
            {"VLLM_LOGGING_LEVEL": "invalid"},
            {"VLLM_ATTENTION_BACKEND": "anything"},
            [],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_serving(dict(self.serving, vllm_env=invalid))

    def test_reserved_aliases_abbreviations_and_equals(self) -> None:
        """Block alternate spellings and parser shortcuts for managed settings."""
        for args in (
            ["--host=0.0.0.0"],
            ["--served_model_name", "other"],
            ["--ho", "0.0.0.0"],
            ["--config", "file.yaml"],
            ["-tp", "2"],
            ["--"],
            ["--hf-token", "secret"],
            ["--enable-log-requests"],
            ["--api-key=secret"],
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_serving({**self.serving, "vllm_args": args})

    def test_invalid_token_shapes(self) -> None:
        """Reject duplicate options, positional overrides, and malformed lists."""
        for args in (
            ["other-model"],
            ["--foo", "x", "other-model"],
            ["--foo=x", "other-model"],
            ["--foo", "--foo"],
            ["--foo", "\n"],
            [1],
            "--foo",
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_serving({**self.serving, "vllm_args": args})

    def test_path_and_endpoint_boundaries(self) -> None:
        """Require paths that resolve predictably on the mounted cloud layout."""
        for path in ("model", "/models", "/models/../secret", "/models//x"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_serving({**self.serving, "model_path": path})
        for changes in (
            {"model": "other"},
            {"base_url": "https://example.com/v1"},
            {"benchmark_path": "/local/manifest.json"},
            {
                "benchmark_path": "artifacts/temporal_reasoning_single_finding/benchmarks/../x"
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_job(self.serving, {**self.experiment, **changes}, self.profile)

    def test_tensor_parallelism_and_gpu_count(self) -> None:
        """Reject insufficient GPU allocations before a cloud invocation."""
        for args in (
            ["--tensor-parallel-size", "2"],
            ["--tensor_parallel_size=2"],
            ["--tensor-parallel-size"],
            ["--tensor-parallel-size=0"],
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_job(
                    {**self.serving, "vllm_args": args}, self.experiment, self.profile
                )
        validate_job(
            {**self.serving, "vllm_args": ["--tensor-parallel-size=2"]},
            self.experiment,
            {**self.profile, "gpu": "H100:2"},
        )

    def test_profile_requires_explicit_environment_and_limits(self) -> None:
        """Catch mutable image aliases and invalid resource settings."""
        for changes in (
            {"image": "vllm/vllm-openai:latest"},
            {"image": "vllm/vllm-openai"},
            {"timeout": True},
            {"startup_timeout": 86400},
            {"pip_packages": ["transformers"]},
            {"gpu": "H100:0"},
            {"memory_mib": 0},
            {"memory_mib": True},
            {"memory_mib": 16.5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_profile({**self.profile, **changes})

    def test_profile_accepts_memory_request(self) -> None:
        """Preserve an explicit host RAM request for the GPU function."""
        profile = validate_profile({**self.profile, "memory_mib": 16384})
        self.assertEqual(profile["memory_mib"], 16384)

    def test_source_packaging_excludes_data_and_retired_code(self) -> None:
        """Ship only active runtime code and the non-Python checksum fixture."""
        paths = source_files()
        self.assertTrue(all(path.is_file() for path in paths))
        names = [path.relative_to(repository_root()).as_posix() for path in paths]
        self.assertIn(
            "experiments/temporal_reasoning_single_finding/tests/fixtures/golden_prompts/sha256.json",
            names,
        )
        self.assertFalse(
            any(
                name.startswith(("artifacts/", "data/", "benchmark/", "tmp/"))
                for name in names
            )
        )
        self.assertFalse(any("construction/" in name for name in names))


class ExplicitProfileTests(unittest.TestCase):
    """Protect launches from stale shell profile selections."""

    def test_local_profile_requires_argument_and_ignores_environment(self) -> None:
        """Both resource types require an argument even with old variables set."""
        from experiments.temporal_reasoning_single_finding.modal import config, endpoint_config

        for module, filename in (
            (config, "modal-profile.example.json"),
            (endpoint_config, "modal-endpoint-profile.example.json"),
        ):
            path = repository_root() / "experiments/temporal_reasoning_single_finding/examples" / filename
            with self.subTest(module=module.__name__), patch.dict(os.environ, {
                "CHEXTEMPORAL_MODAL_PROFILE": "/stale/profile.json",
                "CHEXTEMPORAL_MODAL_ENDPOINT_PROFILE": "/stale/endpoint.json",
            }):
                with patch.object(sys, "argv", ["launch"]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.runtime_profile(True)
                    self.assertEqual(error.exception.code, 2)
                for args in (["--profile", str(path)], ["--profile=" + str(path)]):
                    with patch.object(sys, "argv", ["launch", *args]):
                        self.assertEqual(module.runtime_profile(True), example(filename))

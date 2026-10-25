"""Check endpoint configuration before any SDK import or paid submission."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import write_json
from experiments.temporal_reasoning_single_finding.modal import endpoint_config as config
from experiments.temporal_reasoning_single_finding.modal import endpoint_job
from .test_config import example


def experiment() -> dict:
    """Return an inert HTTPS configuration with no real endpoint or credentials."""
    return {
        **example("modal-endpoint-experiment.example.json"),
        "model": "synthetic-model", "base_url": "https://endpoint.invalid/v1",
    }


def profile() -> dict:
    """Return the documented non-secret CPU execution profile."""
    return example("modal-endpoint-profile.example.json")


class EndpointConfigTests(unittest.TestCase):
    """Validate exact identifiers and reject unsafe or GPU-specific settings."""

    def test_direct_and_gateway_keep_exact_identity(self) -> None:
        """Keep the selected model and reasoning option for either endpoint route."""
        for url in ("https://endpoint.invalid/v1", "https://inference.us-west.modal.direct/v1"):
            value = dict(experiment(), base_url=url, reasoning_effort="low", max_completion_tokens=2048)
            self.assertEqual(config.validate_job(value, profile(), 2), (value, profile()))

    def test_completion_limit_validation(self) -> None:
        """Allow integer limits while retaining validation of string settings."""
        self.assertEqual(config.validate_job(
            dict(experiment(), max_completion_tokens=None), profile()
        )[0], experiment())
        for limit in (0, -1, True, 1.0, "2048", [], {}):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                config.validate_job(dict(experiment(), max_completion_tokens=limit), profile())
        for field, value in (("model", "<model>"), ("reasoning_effort", "high\ncustom")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                value = dict(experiment(), max_completion_tokens=2048, **{field: value})
                config.validate_job(value, profile())

    def test_invalid_urls_and_placeholders(self) -> None:
        """Reject credentials, unsupported paths, malformed ports, and templates."""
        for url in (
            "http://localhost:8000/v1", "https://user:secret@host/v1",
            "https://host/v1?token=secret", "https://host/v1#secret", "https://host/v1?",
            "https://host/v1/chat/completions", "https://<host>/v1",
            "https://host:bad/v1", "https://host:0/v1", "https://ho st/v1",
            "https://host/\\v1", "https://host\n/v1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                config.validate_job(dict(experiment(), base_url=url), profile())
        with self.assertRaises(ValueError):
            config.validate_job(example("modal-endpoint-experiment.example.json"), profile())

    def test_invalid_resources_and_paths(self) -> None:
        """Reject GPU fields, embedded credentials, invalid resources, and path escapes."""
        for changes in (
            {"gpu": "H100"}, {"OPENAI_API_KEY": "secret"}, {"commit_interval": 60},
            {"cpu": True}, {"memory": 0}, {"startup_timeout": 21600},
            {"secret_name": "<secret>"}, {"deployment_identity": {"token": "secret"}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config.validate_job(experiment(), dict(profile(), **changes))
        with self.assertRaises(ValueError):
            config.validate_job(dict(experiment(), benchmark_path="../manifest.json"), profile())
        for workers in (0, True, 1.5):
            with self.assertRaises(ValueError):
                config.validate_job(experiment(), profile(), workers)

    def test_optional_deployment_metadata(self) -> None:
        """Allow explicit operator attribution without adding experiment fields."""
        selected = dict(profile(), deployment_identity={"checkpoint_revision": "abc123"})
        self.assertEqual(config.validate_job(experiment(), selected)[1], selected)

    def test_bad_arguments_do_not_submit(self) -> None:
        """Reject configuration before reaching the function that imports the App."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            write_json(path, experiment())
            with (
                patch.object(config, "runtime_profile", return_value=profile()),
                patch.object(sys, "argv", [
                    "endpoint_job", "--profile", "profile.json", "--experiment", str(path), "--workers", "0",
                ]),
                patch.object(endpoint_job, "submit") as submit,
                self.assertRaises(ValueError),
            ):
                endpoint_job.main()
            submit.assert_not_called()

    def test_check_does_not_import_sdk_or_read_payloads(self) -> None:
        """Run the real CLI validation while blocking SDK imports and image reads."""
        code = '''
import builtins
import sys
from unittest.mock import patch
from experiments.temporal_reasoning_single_finding.modal.endpoint_job import main
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "modal" or name.startswith("modal."):
        raise AssertionError("Modal SDK imported during local check")
    return original(name, *args, **kwargs)
with patch("builtins.__import__", side_effect=guarded):
    main()
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "profile.json", profile())
            write_json(root / "experiment.json", dict(experiment(), max_completion_tokens=2048))
            result = subprocess.run(
                [sys.executable, "-c", code, "--experiment",
                 str(root / "experiment.json"), "--profile", str(root / "profile.json"), "--check"],
                env={**os.environ, config.PROFILE_PATH_ENV: str(root / "profile.json")},
                capture_output=True, text=True, check=True,
            )
            self.assertIn("no API calls made", result.stdout)

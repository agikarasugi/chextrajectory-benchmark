"""Verify checkpoint submission is independent of inference-image construction."""

import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import repository_root


@unittest.skipUnless(importlib.util.find_spec("modal"), "Optional Modal SDK not installed")
class DownloadAppTests(unittest.TestCase):
    """Check the CPU App, storage identity, and credential injection offline."""

    def setUp(self) -> None:
        """Load the CPU App using the documented resource profile."""
        self.profile = str(repository_root() / (
            "experiments/temporal_reasoning_single_finding/examples/"
            "modal-profile.example.json"
        ))
        with patch.object(sys, "argv", ["test", "--profile", self.profile]):
            self.module = importlib.import_module(
                "experiments.temporal_reasoning_single_finding.modal.download_app"
            )

    def test_import_does_not_construct_inference_image(self) -> None:
        """A registry-image failure cannot block loading the downloader App."""
        code = '''
import sys
import modal
from unittest.mock import patch
with patch.object(modal.Image, "from_registry", side_effect=AssertionError("inference image")):
    from experiments.temporal_reasoning_single_finding.modal.download_app import app
    assert set(app.registered_functions) == {"fetch_model"}
    assert "experiments.temporal_reasoning_single_finding.modal.app" not in sys.modules
'''
        subprocess.run(
            [sys.executable, "-c", code, "--profile", self.profile], check=True,
            env={**os.environ, "CHEXTEMPORAL_MODAL_PROFILE": self.profile},
        )

    def test_local_and_remote_imports_have_identical_dependencies(self) -> None:
        """Reproduce container imports and compare the SDK hydration dependency list."""
        code = r'''
import importlib
import json
import os
import sys
from unittest.mock import patch
import modal
from experiments.temporal_reasoning_single_finding.modal.config import PROFILE_ENV
profile_path = os.environ["CHEXTEMPORAL_MODAL_PROFILE"]
with open(profile_path) as stream:
    profile = json.load(stream)
original_function = modal.App.function
definitions = []
def capture_function(self, **kwargs):
    """Record resource declarations while retaining real SDK construction."""
    definitions.append(kwargs)
    return original_function(self, **kwargs)
for short_name, function_name, expected in [
    ("download_app", "fetch_model", ["Image", "Volume"]),
    ("app", "run_benchmark", ["Image", "Volume", "Volume", "Volume", "Volume"]),
]:
    name = "experiments.temporal_reasoning_single_finding.modal." + short_name
    with patch.object(modal.App, "function", capture_function):
        local = importlib.import_module(name)
    local_definition = definitions[-1]
    local_deps = [type(dep).__name__ for dep in getattr(local, function_name)._deps_(only_explicit_mounts=True)]
    mounts_name = "DOWNLOAD_MOUNTS" if short_name == "download_app" else "MOUNTS"
    mount_paths = list(getattr(local, mounts_name))
    del sys.modules[name]
    with patch.object(modal.App, "function", capture_function), patch.object(
        modal, "is_local", return_value=False
    ), patch.dict(os.environ, {
        PROFILE_ENV: json.dumps(profile),
        "CHEXTEMPORAL_MODAL_PROFILE": "/nonexistent-local-profile.json",
    }), patch(
        "experiments.temporal_reasoning_single_finding.modal.config.source_files",
        side_effect=AssertionError("Remote import tried to package local files"),
    ):
        remote = importlib.import_module(name)
        remote_deps = [type(dep).__name__ for dep in getattr(remote, function_name)._deps_(only_explicit_mounts=True)]
        assert remote.PROFILE == profile
        assert list(getattr(remote, mounts_name)) == mount_paths
    assert local_deps == remote_deps == expected, (local_deps, remote_deps)
    if short_name == "app":
        for definition in (local_definition, definitions[-1]):
            assert definition["gpu"] == profile["gpu"]
            assert definition["timeout"] == profile["timeout"]
            assert definition["region"] == profile.get("region")
'''
        subprocess.run(
            [sys.executable, "-c", code, "--profile", self.profile], check=True,
            env={**os.environ, "CHEXTEMPORAL_MODAL_PROFILE": self.profile},
        )

    def test_download_passes_secret_and_region_without_gpu(self) -> None:
        """Only the CPU call receives the Secret and optional placement."""
        function = Mock()
        secret = Mock()
        profile = {"volume_prefix": "selected", "region": "us-east"}
        with (
            patch.object(self.module, "fetch_model", function),
            patch.object(self.module, "PROFILE", profile),
            patch.object(self.module.modal.Secret, "from_name", return_value=secret),
        ):
            self.module.download("owner/model", "a" * 40, "/models/model", "hf")
        function.with_options.assert_called_once_with(
            secrets=[secret], region="us-east"
        )
        function.with_options.return_value.spawn.assert_called_once_with(
            "owner/model", "a" * 40, "/models/model", "selected"
        )

    def test_interrupted_wait_does_not_cancel_or_resubmit(self) -> None:
        """Keep a spawned download independent of an interrupted local wait."""
        function = Mock()
        configured = function.with_options.return_value
        call = configured.spawn.return_value
        call.get.side_effect = KeyboardInterrupt
        with patch.object(self.module, "fetch_model", function), self.assertRaises(KeyboardInterrupt):
            self.module.download("owner/model", "a" * 40, "/models/model")
        configured.spawn.assert_called_once()
        configured.remote.assert_not_called()
        call.get.assert_called_once_with()
        call.cancel.assert_not_called()

    def test_failure_releases_model_lock(self) -> None:
        """The independent App retains cleanup and the same model Volume."""
        for resumed in (False, True):
            with self.subTest(resumed=resumed):
                volume = Mock()
                locks = Mock()
                locks.put.return_value = not resumed
                locks.get.return_value = "call"
                with (
                    patch.object(self.module.modal.Volume, "from_name", return_value=volume) as lookup,
                    patch.object(self.module.modal.Dict, "from_name", return_value=locks),
                    patch.object(self.module.modal, "current_function_call_id", return_value="call"),
                    patch(
                        "experiments.temporal_reasoning_single_finding.modal.download_model.download_checkpoint",
                        side_effect=RuntimeError("synthetic failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "synthetic failure"),
                ):
                    self.module.fetch_model.local("owner/model", "a" * 40, "/models/model", "selected")
                lookup.assert_called_once_with("selected-models", create_if_missing=True)
                volume.reload.assert_called_once()
                locks.pop.assert_called_once_with("model:/models/model")


class DownloadArgumentTests(unittest.TestCase):
    """Reject malformed local inputs before entering the App context."""

    def test_omitted_revision_is_forwarded_to_worker(self) -> None:
        """The CLI accepts omission without resolving a revision locally."""
        from experiments.temporal_reasoning_single_finding.modal import download_job

        fake_app = MagicMock()
        fake_modal = MagicMock()
        with (
            patch.dict(sys.modules, {
                "modal": fake_modal,
                "experiments.temporal_reasoning_single_finding.modal.download_app": fake_app,
            }),
            patch.object(sys, "argv", [
                "download_job", "--profile", "profile.json", "--repo-id", "owner/model",
                "--model-path", "/models/model", "--secret", "hf",
            ]),
        ):
            download_job.main()
        fake_app.app.run.assert_called_once_with(detach=True)
        fake_app.download.assert_called_once_with(
            "owner/model", None, "/models/model", "hf"
        )

    def test_invalid_arguments_do_not_start_app(self) -> None:
        """Invalid revisions and destinations fail before cloud preparation."""
        from experiments.temporal_reasoning_single_finding.modal import download_job

        for revision, destination in [("abc123", "/models/model"),
                                      ("a" * 40, "/outside/model")]:
            with self.subTest(revision=revision, destination=destination):
                fake_app = Mock()
                with (
                    patch.dict(sys.modules, {
                        "experiments.temporal_reasoning_single_finding.modal.download_app": fake_app,
                    }),
                    patch.object(sys, "argv", [
                        "download_job", "--profile", "profile.json", "--repo-id", "owner/model",
                        "--revision", revision, "--model-path", destination,
                    ]),
                    self.assertRaises(ValueError),
                ):
                    download_job.main()
                fake_app.app.run.assert_not_called()
                fake_app.download.assert_not_called()

"""Verify CPU resource wiring and durable submission without authenticating to Modal."""

import importlib
import importlib.util
import os
import subprocess
import sys
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import repository_root
from experiments.temporal_reasoning_single_finding.modal import endpoint_job
from experiments.temporal_reasoning_single_finding.modal.endpoint_config import PROFILE_PATH_ENV
from .test_endpoint_config import experiment, profile


@unittest.skipUnless(importlib.util.find_spec("modal"), "Optional Modal SDK is not installed")
class EndpointAppTests(unittest.TestCase):
    """Inspect the actual decorated Function and mock its remote dependencies."""

    def setUp(self) -> None:
        """Import the endpoint App using its example non-secret resource profile."""
        self.profile_path = str(repository_root() / (
            "experiments/temporal_reasoning_single_finding/examples/"
            "modal-endpoint-profile.example.json"
        ))
        with patch.object(sys, "argv", ["test", "--profile", self.profile_path]):
            self.module = importlib.import_module(
                "experiments.temporal_reasoning_single_finding.modal.endpoint_app"
            )

    def test_cpu_resources_secret_and_remote_hydration(self) -> None:
        """Construct both SDK definitions and prove equal dependencies without a GPU App."""
        code = '''
import importlib
import json
import os
import sys
from collections import Counter
from unittest.mock import patch
import modal
from experiments.temporal_reasoning_single_finding.modal.endpoint_config import (
    PROFILE_ENV, PROFILE_PATH_ENV,
)
with open(os.environ[PROFILE_PATH_ENV]) as stream:
    profile = json.load(stream)
name = "experiments.temporal_reasoning_single_finding.modal.endpoint_app"
original = modal.App.function
original_mount = modal.Volume.with_mount_options
definitions = []
mount_options = []
def capture(self, **kwargs):
    definitions.append(kwargs)
    return original(self, **kwargs)
def capture_mount(self, **kwargs):
    mount_options.append((self, kwargs))
    return original_mount(self, **kwargs)
with patch.object(modal.App, "function", capture), patch.object(
    modal.Image, "from_registry", side_effect=AssertionError("GPU registry image")
), patch.object(
    modal.Volume, "with_mount_options", capture_mount
):
    local = importlib.import_module(name)
assert "experiments.temporal_reasoning_single_finding.modal.app" not in sys.modules
definition, = definitions
assert "gpu" not in definition
assert definition["cpu"] == profile["cpu"]
assert definition["memory"] == profile["memory"]
assert definition["timeout"] == profile["timeout"]
assert definition["retries"] == 0
assert definition["include_source"] is False
assert definition["secrets"] == [local.SECRET]
assert set(local.stores) == {"images", "benchmarks", "results"}
assert "/models" not in local.MOUNTS
assert mount_options == [
    (local.stores["images"], {"read_only": True}),
    (local.stores["benchmarks"], {"read_only": True}),
]
local_deps = [
    type(dep).__name__ for dep in local.run_benchmark._deps_(only_explicit_mounts=True)
]
del sys.modules[name]
with patch.object(modal, "is_local", return_value=False), patch.dict(os.environ, {
    PROFILE_ENV: json.dumps(profile), PROFILE_PATH_ENV: "/no-local-profile",
}), patch("experiments.temporal_reasoning_single_finding.modal.config.package_source",
          side_effect=AssertionError("Remote source packaging")):
    remote = importlib.import_module(name)
remote_deps = [
    type(dep).__name__ for dep in remote.run_benchmark._deps_(only_explicit_mounts=True)
]
assert remote.PROFILE == local.PROFILE == profile
assert list(remote.MOUNTS) == list(local.MOUNTS)
assert local_deps == remote_deps, (local_deps, remote_deps)
assert Counter(local_deps) == Counter({"Image": 1, "Volume": 3, "Secret": 1})
'''
        subprocess.run(
            [sys.executable, "-c", code, "--profile", self.profile_path], check=True,
            env={**os.environ, PROFILE_PATH_ENV: self.profile_path},
        )

    def test_spawn_detach_and_optional_wait(self) -> None:
        """Submit once, print the call ID, and never cancel after a local interruption."""
        for mode in ("wait", "no-wait", "interrupted"):
            with self.subTest(mode=mode):
                function = Mock()
                call = function.spawn.return_value
                call.object_id = "fc-synthetic"
                if mode == "interrupted":
                    call.get.side_effect = KeyboardInterrupt
                with (
                    patch.object(self.module, "run_benchmark", function),
                    patch.object(self.module.app, "run", return_value=nullcontext()) as run,
                    patch.object(self.module.modal, "enable_output", return_value=nullcontext()),
                    patch("builtins.print") as output,
                ):
                    if mode == "interrupted":
                        with self.assertRaises(KeyboardInterrupt):
                            endpoint_job.submit(
                                experiment(), profile(), "source", 3, True, retry_failed=True,
                            )
                    else:
                        endpoint_job.submit(experiment(), profile(), "source", 3, mode == "wait")
                run.assert_called_once_with(name=experiment()["run_name"], detach=True)
                function.spawn.assert_called_once_with(
                    experiment(), profile(), "source", 3, retry_failed=mode == "interrupted",
                )
                output.assert_any_call("Submitted endpoint benchmark: fc-synthetic", flush=True)
                function.remote.assert_not_called()
                call.cancel.assert_not_called()
                self.assertEqual(call.get.call_count, int(mode != "no-wait"))

    def test_lock_lifecycle_and_mount_reload(self) -> None:
        """Reject competing writers and release owned locks after execution or reload failure."""
        for outcome in ("success", "resumed", "locked", "reload", "pipeline", "cleanup"):
            with self.subTest(outcome=outcome):
                locks = Mock()
                locks.put.return_value = outcome not in ("locked", "resumed")
                locks.get.return_value = "fc-synthetic" if outcome == "resumed" else "fc-other"
                stores = {name: Mock() for name in ("images", "benchmarks", "results")}
                error = RuntimeError("original failure")
                if outcome == "reload":
                    stores["images"].reload.side_effect = error
                if outcome == "cleanup":
                    locks.pop.side_effect = OSError("lock cleanup failure")
                with (
                    patch.object(self.module, "stores", stores),
                    patch.object(self.module.modal.Dict, "from_name", return_value=locks),
                    patch.object(
                        self.module.modal, "current_function_call_id", return_value="fc-synthetic",
                    ),
                    patch(
                        "experiments.temporal_reasoning_single_finding.modal."
                        "endpoint_launcher.execute_job"
                    ) as execute,
                ):
                    if outcome in ("pipeline", "cleanup"):
                        execute.side_effect = error
                    if outcome in ("success", "resumed"):
                        self.module.run_benchmark.local(
                            experiment(), profile(), "code", 2, retry_failed=True,
                        )
                    else:
                        with self.assertRaises(RuntimeError) as raised:
                            self.module.run_benchmark.local(experiment(), profile(), "code", 2)
                        if outcome != "locked":
                            self.assertIs(raised.exception, error)
                    if outcome in ("locked", "reload"):
                        execute.assert_not_called()
                    else:
                        execute.assert_called_once_with(
                            experiment(), profile(), "code", stores["results"].commit, workers=2,
                            retry_failed=outcome in ("success", "resumed"),
                        )
                locks.put.assert_called_once_with(
                    "run:endpoint-smoke-001", "fc-synthetic", skip_if_exists=True,
                )
                if outcome == "locked":
                    locks.pop.assert_not_called()
                    for volume in stores.values():
                        volume.reload.assert_not_called()
                else:
                    locks.pop.assert_called_once_with("run:endpoint-smoke-001")

    def test_profile_mismatch_precedes_lock_or_mount_access(self) -> None:
        """Refuse to record resources that differ from the Function's actual mounts."""
        with (
            patch.object(self.module.modal.Dict, "from_name") as locks,
            self.assertRaisesRegex(ValueError, "mounted resources"),
        ):
            self.module.run_benchmark.local(
                experiment(), dict(profile(), volume_prefix="different"), "code",
            )
        locks.assert_not_called()

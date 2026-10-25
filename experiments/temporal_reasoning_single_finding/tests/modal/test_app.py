"""Test SDK wiring and selected-source execution without authenticating to Modal."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import repository_root
from experiments.temporal_reasoning_single_finding.modal.config import package_source, source_files


class PackagingTests(unittest.TestCase):
    """Verify runtime packaging independently of the full repository."""

    def test_packaged_prompt_preflight(self) -> None:
        """The selected source files suffice to validate all prompt snapshots."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in source_files():
                target = root / path.relative_to(repository_root())
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            command = (
                "from experiments.temporal_reasoning_single_finding.validation.preflight "
                "import validate_prompt_snapshots; "
                "validate_prompt_snapshots(); print('packaged prompts passed')"
            )
            result = subprocess.run(
                [sys.executable, "-c", command],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("packaged prompts passed", result.stdout)


@unittest.skipUnless(
    importlib.util.find_spec("modal"), "Optional Modal SDK is not installed"
)
class AppTests(unittest.TestCase):
    """Exercise actual decorated entrypoints with cloud handles mocked."""

    def setUp(self) -> None:
        """Import the App using its explicit resource profile without cloud calls."""
        profile = (
            repository_root()
            / "experiments/temporal_reasoning_single_finding/examples/modal-profile.example.json"
        )
        with patch.object(sys, "argv", ["test", "--profile", str(profile)]):
            self.module = importlib.import_module(
                "experiments.temporal_reasoning_single_finding.modal.app"
            )

    def test_single_copy_contains_exact_source_allowlist(self) -> None:
        """Inspect the SDK build definition and file selection without cloud calls."""
        from modal._image import _Image

        base = _Image.debian_slim(python_version="3.12")
        with patch.object(_Image, "_from_args") as build:
            package_source(base)
        build.assert_called_once()
        arguments = build.call_args.kwargs
        self.assertEqual(
            arguments["dockerfile_function"](None).commands,
            ["FROM base", "COPY . /"],
        )
        mount = arguments["context_mount_function"]()
        self.assertEqual(len(mount._entries), 1)
        entry = mount._entries[0]
        self.assertTrue(entry.ignore.can_prune_directories())
        for directory in ("data", "artifacts", "tmp", ".git", ".venv"):
            self.assertTrue(entry.ignore(Path(directory)))
        selected = set(entry.get_files_to_upload())
        expected = {
            (path, Path("/app") / path.relative_to(repository_root()))
            for path in source_files()
        }
        self.assertEqual(selected, expected)

    def test_submission_survives_interrupted_wait(self) -> None:
        """Submit once asynchronously and never cancel when the local wait ends."""
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                function = Mock()
                call = function.spawn.return_value
                call.object_id = "fc-synthetic"
                call.get.return_value = {"status": "complete"}
                if interrupted:
                    call.get.side_effect = KeyboardInterrupt
                profile = {"gpu": "H100", "timeout": 21600, "region": "us-east"}
                with (
                    patch.object(self.module, "run_benchmark", function),
                    patch.object(self.module, "load_job", return_value=({"serving": 1}, {"experiment": 1}, profile)),
                    patch.object(self.module, "source_digest", return_value="code"),
                    patch("builtins.print") as output,
                ):
                    if interrupted:
                        with self.assertRaises(KeyboardInterrupt):
                            self.module.main("serving.json", "experiment.json", 1, retry_failed=True)
                    else:
                        self.module.main("serving.json", "experiment.json", 1)
                        output.assert_any_call({"status": "complete"})
                    output.assert_any_call("Submitted benchmark: fc-synthetic", flush=True)
                function.with_options.assert_not_called()
                function.spawn.assert_called_once_with(
                    {"serving": 1}, {"experiment": 1}, profile, "code", 1,
                    retry_failed=interrupted,
                )
                function.remote.assert_not_called()
                call.get.assert_called_once_with()
                call.cancel.assert_not_called()

    def test_python_alias_preserves_registry_interpreter(self) -> None:
        """Support python3-only images without replacing an existing python."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "link-arguments"
            # Stub ln records the requested link without changing system files.
            link = root / "ln"
            link.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$LINK_LOG"\n')
            link.chmod(0o755)
            python3 = root / "python3"
            python3.write_text("#!/bin/sh\nexit 0\n")
            python3.chmod(0o755)
            env = {"PATH": str(root), "LINK_LOG": str(log)}
            subprocess.run(
                ["/bin/sh", "-c", self.module.PYTHON_ALIAS_COMMAND],
                env=env, check=True,
            )
            self.assertEqual(log.read_text().splitlines(), [
                "-s", str(python3), "/usr/local/bin/python"
            ])
            log.unlink()
            (root / "python").symlink_to(python3)
            subprocess.run(
                ["/bin/sh", "-c", self.module.PYTHON_ALIAS_COMMAND],
                env=env, check=True,
            )
            self.assertFalse(log.exists())
            (root / "python").unlink()
            python3.unlink()
            result = subprocess.run(
                ["/bin/sh", "-c", self.module.PYTHON_ALIAS_COMMAND], env=env
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists())

    def test_cloud_lock_rejects_concurrent_writer(self) -> None:
        """Never invoke the pipeline when another call owns the run key."""
        locks = Mock()
        locks.put.return_value = False
        with (
            patch.object(self.module.modal.Dict, "from_name", return_value=locks),
            patch.object(self.module, "volumes", return_value={}),
            patch.object(
                self.module.modal,
                "current_function_call_id",
                return_value="synthetic-call",
            ),
            self.assertRaisesRegex(RuntimeError, "locked"),
        ):
            self.module.run_benchmark.local(
                {}, {"run_name": "run"}, {"volume_prefix": "test"}, "code", 1
            )
        locks.pop.assert_not_called()

    def test_cloud_failure_releases_owned_lock(self) -> None:
        """Release a handled failed call's lock after the supervisor cleans up."""
        for resumed in (False, True):
            with self.subTest(resumed=resumed):
                locks = Mock()
                locks.put.return_value = not resumed
                locks.get.return_value = "synthetic-call"
                stores = {
                    name: Mock() for name in ("images", "benchmarks", "models", "results")
                }
                with (
                    patch.object(self.module.modal.Dict, "from_name", return_value=locks),
                    patch.object(self.module, "volumes", return_value=stores),
                    patch.object(
                        self.module.modal,
                        "current_function_call_id",
                        return_value="synthetic-call",
                    ),
                    patch(
                        "experiments.temporal_reasoning_single_finding.modal.launcher.execute_job",
                        side_effect=RuntimeError("synthetic"),
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    self.module.run_benchmark.local(
                        {}, {"run_name": "run"}, {"volume_prefix": "test"}, "code", 1
                    )
                locks.pop.assert_called_once_with("run:run")
                for store in stores.values():
                    store.reload.assert_called_once()

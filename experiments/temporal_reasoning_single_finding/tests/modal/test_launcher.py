"""Exercise lifecycle failures, persistence, and compatible code-fix resume offline."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    read_jsonl,
    write_json,
)
from experiments.temporal_reasoning_single_finding.modal import launcher
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
    execute_specs,
)

from .test_config import example


class LifecycleTests(unittest.TestCase):
    """Verify cleanup under readiness, child failure, and storage failure."""

    def test_readiness_rejects_exited_server(self) -> None:
        """Do not wait out the startup timeout after an early server crash."""
        server = Mock()
        server.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "before readiness"):
            launcher.wait_ready(server, "model", 10)

    def test_readiness_timeout(self) -> None:
        """Enforce the deadline when the expected model never appears."""
        server = Mock()
        server.poll.return_value = None
        with (
            patch.object(launcher.time, "monotonic", side_effect=[0, 0, 2]),
            patch.object(launcher.time, "sleep"),
            patch.object(launcher, "endpoint_ready", return_value=False),
            self.assertRaises(TimeoutError),
        ):
            launcher.wait_ready(server, "model", 1)

    def test_readiness_requires_exact_model(self) -> None:
        """Model discovery accepts only the configured API identifier."""
        response = Mock()
        response.read.return_value = b'{"data":[{"id":"other"}]}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(launcher, "urlopen", return_value=context):
            self.assertFalse(launcher.endpoint_ready("model"))
            self.assertTrue(launcher.endpoint_ready("other"))

    def test_real_subprocess_success_and_failure(self) -> None:
        """Run synthetic CPU children and enforce stage exit codes."""
        server = Mock()
        server.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            commit = Mock()
            launcher.run_stage(
                [sys.executable, "-c", "pass"],
                server,
                commit,
                60,
                Path(directory),
                dict(launcher.os.environ),
            )
            commit.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "subprocess failed"):
                launcher.run_stage(
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    server,
                    commit,
                    60,
                    Path(directory),
                    dict(launcher.os.environ),
                )

    def test_stage_forwards_stdout_and_stderr(self) -> None:
        """Expose runner progress and error diagnostics through inherited streams."""
        server = Mock()
        server.poll.return_value = None
        child = Mock()
        child.poll.return_value = 0
        child.returncode = 0
        with (
            patch.object(launcher.subprocess, "Popen", return_value=child) as spawn,
            patch.object(launcher, "stop_process"),
        ):
            launcher.run_stage(["fake"], server, Mock(), 60, Path("/tmp"), {})
        self.assertIsNone(spawn.call_args.kwargs["stdout"])
        self.assertIsNone(spawn.call_args.kwargs["stderr"])

    def test_midstage_server_death_stops_child(self) -> None:
        """Stop a running pipeline when its model process disappears."""
        server = Mock()
        server.poll.return_value = 1
        child = Mock()
        child.poll.return_value = None
        with (
            patch.object(launcher.subprocess, "Popen", return_value=child),
            patch.object(launcher, "stop_process") as stop,
            self.assertRaisesRegex(RuntimeError, "during the experiment"),
        ):
            launcher.run_stage(["fake"], server, Mock(), 60, Path("/tmp"), {})
        stop.assert_called_once_with(child)

    def test_checkpoint_commit_failure_stops_child(self) -> None:
        """A persistence failure aborts further inference instead of being ignored."""
        server = Mock()
        server.poll.return_value = None
        child = Mock()
        child.poll.return_value = None
        with (
            patch.object(launcher.subprocess, "Popen", return_value=child),
            patch.object(launcher.time, "monotonic", side_effect=[0, 2]),
            patch.object(launcher, "stop_process") as stop,
            self.assertRaises(OSError),
        ):
            launcher.run_stage(
                ["fake"], server, Mock(side_effect=OSError), 1, Path("/tmp"), {}
            )
        stop.assert_called_once_with(child)

    def test_process_group_cleanup_escalates(self) -> None:
        """Kill worker descendants even when the server parent does not exit."""
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 10), 0]
        with patch.object(launcher.os, "killpg") as kill:
            launcher.stop_process(process)
        self.assertEqual(kill.call_count, 2)

    def test_provenance_resume_and_conflict(self) -> None:
        """Preserve saved identity and reject changed model or command settings."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            launcher.require_provenance(path, {"model": "a"})
            original = path.read_bytes()
            launcher.require_provenance(path, {"model": "a"})
            with self.assertRaises(ValueError) as error:
                launcher.require_provenance(path, {"model": "b"})
            self.assertIn('model: saved="a", current="b"', str(error.exception))
            self.assertIn(str(path), str(error.exception))
            self.assertEqual(path.read_bytes(), original)

    def test_provenance_error_details(self) -> None:
        """Show nested changes, additions, removals, and lists without compatible fields."""
        identity = {
            "checkpoint": {"revision": "old", "unchanged": "same"},
            "experiment": {"removed": None, "max_completion_tokens": None},
            "serving": {"vllm_args": ["--dtype", "float16"]},
            "workers": 1,
        }
        updated = deepcopy(identity)
        updated["checkpoint"]["revision"] = "new"
        updated["experiment"] = {"added": None}
        updated["serving"]["vllm_args"] = ["--dtype", "bfloat16"]
        updated["workers"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            launcher.require_provenance(path, identity)
            original = path.read_bytes()
            with self.assertRaises(ValueError) as error:
                launcher.require_provenance(path, updated)
            self.assertEqual(str(error.exception).splitlines()[2:], [
                '  checkpoint.revision: saved="old", current="new"',
                '  experiment.added: saved=<missing>, current=null',
                '  experiment.removed: saved=null, current=<missing>',
                '  serving.vllm_args: saved=["--dtype", "float16"], '
                'current=["--dtype", "bfloat16"]',
            ])
            self.assertEqual(path.read_bytes(), original)

    def test_request_timeout_can_change_on_resume(self) -> None:
        """Allow default and explicit timeout changes while preserving attempt history."""
        options = ({}, {"request_timeout_seconds": None},
                   {"request_timeout_seconds": 120}, {"request_timeout_seconds": 600})
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)
            path = metadata / "provenance.json"
            for saved in options:
                identity = {"experiment": {"model": "model", **saved}}
                write_json(path, identity)
                original = path.read_bytes()
                for current in options:
                    with self.subTest(saved=saved, current=current):
                        updated = {"experiment": {"model": "model", **current}}
                        launcher.require_provenance(path, updated)
                        attempt = launcher.record_attempt(metadata, updated)
                        self.assertEqual(
                            read_json(attempt / "provenance.json")["provenance"], updated,
                        )
                        self.assertEqual(path.read_bytes(), original)
                        updated["experiment"]["model"] = "other"
                        with self.assertRaisesRegex(ValueError, "experiment.model"):
                            launcher.require_provenance(path, updated)

    def test_context_resume_rejects_other_identity_changes(self) -> None:
        """Permit context changes while retaining generation and model identity."""
        serving = example("modal-serving.example.json")
        serving["vllm_args"] = ["--max-model-len", "28000"]
        original = {
            "serving": serving,
            "server_command": launcher.server_command(serving),
            "experiment": {"max_completion_tokens": 512},
            "checkpoint": {"revision": "original"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            write_json(path, original)
            expanded = deepcopy(original)
            expanded["serving"]["vllm_args"][1] = "32768"
            expanded["server_command"] = launcher.server_command(expanded["serving"])
            launcher.require_provenance(path, expanded)
            for limit in ("16000", "auto"):
                changed = deepcopy(expanded)
                changed["serving"]["vllm_args"][1] = limit
                changed["server_command"] = launcher.server_command(changed["serving"])
                launcher.require_provenance(path, changed)
            for change in ("sampling", "cap", "checkpoint", "command"):
                with self.subTest(change=change):
                    changed = deepcopy(expanded)
                    if change == "sampling":
                        changed["serving"]["vllm_args"] += ["--generation-config", "vllm"]
                    elif change == "cap":
                        changed["experiment"]["max_completion_tokens"] = 1024
                    elif change == "checkpoint":
                        changed["checkpoint"]["revision"] = "different"
                    changed["server_command"] = launcher.server_command(changed["serving"])
                    if change == "command":
                        changed["server_command"].append("--enforce-eager")
                    with self.assertRaisesRegex(ValueError, "provenance changed"):
                        launcher.require_provenance(path, changed)
            uncapped = deepcopy(original)
            uncapped["experiment"] = {}
            write_json(path, uncapped)
            expanded["experiment"] = {}
            launcher.require_provenance(path, expanded)

    def test_execution_arguments_can_change_without_replacing_history(self) -> None:
        """Allow scheduler and parallelism tuning, including removing old limits."""
        serving = example("modal-serving.example.json")
        protected = ["--dtype", "bfloat16", "--generation-config", "vllm"]
        execution = [
            "--gpu-memory-utilization", "0.9", "--max-model-len", "auto",
            "--max-num-seqs", "1", "--max-num-batched-tokens", "7168",
            "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
        ]
        serving["vllm_args"] = protected + execution
        identity = {
            "serving": serving, "server_command": launcher.server_command(serving),
            "experiment": {"model": "model"},
        }
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)
            path = metadata / "provenance.json"
            launcher.require_provenance(path, identity)
            original = path.read_bytes()
            for args in (
                protected,
                protected + ["--max-model-len=9639", "--max_num_seqs=256",
                             "--max-num-batched-tokens=32768",
                             "--gpu-memory-utilization=0.95",
                             "--tensor-parallel-size=2", "--pipeline-parallel-size=2"],
            ):
                updated = deepcopy(identity)
                updated["serving"]["vllm_args"] = args
                updated["server_command"] = launcher.server_command(updated["serving"])
                launcher.require_provenance(path, updated)
                attempt = launcher.record_attempt(metadata, updated)
                self.assertEqual(read_json(attempt / "provenance.json")["provenance"], updated)
                self.assertEqual(path.read_bytes(), original)
            for option in (
                ["--quantization", "fp8"],
                ["--override-generation-config", '{"temperature":0.6}'],
                ["--default-chat-template-kwargs", '{"enable_thinking":false}'],
                ["--unknown-future-option", "value"],
            ):
                changed = deepcopy(identity)
                changed["serving"]["vllm_args"] += option
                changed["server_command"] = launcher.server_command(changed["serving"])
                with self.subTest(option=option), self.assertRaisesRegex(
                    ValueError, "provenance changed",
                ):
                    launcher.require_provenance(path, changed)

    def test_legacy_provenance_preserves_request_identity(self) -> None:
        """Allow execution metadata changes while retaining other identity checks."""
        identity = {
            "source_sha256": "old-code",
            "serving": {"model_path": "/models/model", "vllm_args": []},
            "experiment": {"model": "model", "benchmark_path": "manifest.json"},
            "profile": {"gpu": "H100", "commit_interval": 60},
            "checkpoint": {"revision": "pinned"},
            "versions": {"vllm": "pinned"},
            "workers": 1,
            "enable_short_explanation": True,
            "server_command": ["vllm", "serve", "/models/model"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            write_json(path, identity)
            original = path.read_bytes()
            updated = dict(
                identity, source_sha256="fixed-code",
                profile={"gpu": "A100", "commit_interval": 300},
                versions={"vllm": "different"}, workers=4,
            )
            launcher.require_provenance(path, updated)
            updated.pop("enable_short_explanation")
            launcher.require_provenance(path, updated)
            for key in identity.keys() - {"source_sha256", "profile", "versions", "workers"}:
                with self.subTest(field=key):
                    changed = deepcopy(updated)
                    changed[key] = {"incompatible": True}
                    with self.assertRaisesRegex(ValueError, "provenance changed"):
                        launcher.require_provenance(path, changed)
            self.assertEqual(path.read_bytes(), original)

    def test_logging_changes_resume_legacy_and_new_runs(self) -> None:
        """Keep logging mutable, attempts auditable, and model settings strict."""
        identity = {"serving": example("modal-serving.example.json")}
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)
            path = metadata / "provenance.json"
            launcher.require_provenance(path, identity)
            original = path.read_bytes()
            updated = deepcopy(identity)
            for interval in ("60", "120"):
                updated["serving"]["vllm_env"] = {
                    "VLLM_LOGGING_LEVEL": "INFO", "VLLM_LOG_STATS_INTERVAL": interval,
                }
                launcher.require_provenance(path, updated)
                attempt = launcher.record_attempt(metadata, updated)
                self.assertEqual(read_json(attempt / "provenance.json")["provenance"], updated)
            self.assertEqual(path.read_bytes(), original)
            write_json(path, updated)
            launcher.require_provenance(path, identity)
            changed = deepcopy(updated)
            changed["serving"]["vllm_args"] += ["--dtype", "float16"]
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                launcher.require_provenance(path, changed)
            changed = deepcopy(updated)
            changed["serving"]["vllm_env"]["UNKNOWN"] = "value"
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                launcher.require_provenance(path, changed)

    def test_answer_only_provenance_is_rejected(self) -> None:
        """The retired false flag cannot be normalized into a different prompt."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            for value in (False, None, 1, "true"):
                with self.subTest(value=value):
                    write_json(path, {"model": "model", "enable_short_explanation": value})
                    with self.assertRaisesRegex(ValueError, "saved prompt mode"):
                        launcher.require_provenance(path, {"model": "model"})

    def test_every_profile_field_can_change_on_resume(self) -> None:
        """Accept each validated infrastructure change without changing saved provenance."""
        profile = example("modal-profile.example.json")
        changes = {
            "gpu": "A100", "timeout": 43200, "startup_timeout": 1200,
            "commit_interval": 300, "image": "vllm/vllm-openai:v0.28.0",
            "volume_prefix": "relocated", "region": "eu-west",
            "pip_packages": ["transformers==4.57.1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            identity = {"profile": profile, "source_sha256": "code"}
            launcher.require_provenance(path, identity)
            original = path.read_bytes()
            for field, value in changes.items():
                with self.subTest(field=field):
                    updated = dict(profile, **{field: value})
                    from experiments.temporal_reasoning_single_finding.modal.config import (
                        validate_profile,
                    )

                    validate_profile(updated)
                    launcher.require_provenance(path, dict(identity, profile=updated))
            self.assertEqual(path.read_bytes(), original)

    def test_submitted_source_must_still_match_container(self) -> None:
        """Reject a wrongly packaged attempt before reading inputs or starting processes."""
        with (
            patch.object(launcher, "source_digest", return_value="different"),
            patch.object(launcher, "read_json") as read,
            patch.object(launcher.subprocess, "Popen") as spawn,
            patch.object(launcher, "record_attempt") as record,
        ):
            commit = Mock()
            with self.assertRaisesRegex(ValueError, "Packaged source fingerprint"):
                launcher.execute_job(
                    example("modal-serving.example.json"),
                    example("modal-experiment.example.json"),
                    example("modal-profile.example.json"),
                    "submitted", commit,
                )
            read.assert_not_called()
            spawn.assert_not_called()
            record.assert_not_called()
            commit.assert_not_called()

    def test_inventory_checks_metadata_without_reading_payloads(self) -> None:
        """Accept legacy hashes without reading files and reject missing or resized files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.bin"
            path.write_bytes(b"synthetic fixture")
            files = {"synthetic.bin": {"bytes": 17, "sha256": "legacy-hash"}}
            with patch.object(Path, "open", side_effect=AssertionError("payload read")):
                launcher.check_file_metadata(root, files)
                with self.assertRaises(ValueError):
                    launcher.check_file_metadata(root, {"../elsewhere": files["synthetic.bin"]})
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                launcher.check_file_metadata(root, files)
            path.unlink()
            with self.assertRaises(ValueError):
                launcher.check_file_metadata(root, files)

    def _execute(
        self, directory: str, stage_error: Exception | None = None,
        *, code: str = "code", stage_action=None,
        profile_changes: dict | None = None, versions: dict | None = None,
        logging_env: dict | None = None, experiment_changes: dict | None = None,
        context_limit: int | None = None,
        max_num_seqs: int | None = None,
        retry_failed: bool = False,
    ) -> tuple:
        """Run the supervisor with synthetic metadata and mocked GPU processes."""
        root = Path(directory)
        serving = example("modal-serving.example.json")
        if context_limit is not None:
            args = serving["vllm_args"]
            args[args.index("--max-model-len") + 1] = str(context_limit)
        if logging_env is not None:
            serving["vllm_env"] = logging_env
        if max_num_seqs is not None:
            serving["vllm_args"] += ["--max-num-seqs", str(max_num_seqs)]
        experiment = example("modal-experiment.example.json")
        experiment.update(experiment_changes or {})
        profile = example("modal-profile.example.json")
        profile.update(profile_changes or {})
        benchmark = root / experiment["benchmark_path"]
        write_json(benchmark, {"synthetic": True})
        write_json(
            benchmark.parent / "modal_images.json",
            {
                "benchmark_sha256": launcher.sha256_file(benchmark),
                "images": {"fake": {}},
            },
        )
        original_read = read_json

        def fake_read(path: Path) -> dict:
            """Replace only the absent cloud checkpoint's metadata."""
            if path.name == "modal_checkpoint.json":
                return {"revision": "abc", "files": {"fake": {}}}
            return original_read(path)

        commit = Mock()
        with (
            patch.object(launcher, "repository_root", return_value=root),
            patch.object(launcher, "source_digest", return_value=code),
            patch.object(
                launcher, "package_versions",
                return_value=versions or {"vllm": "original"},
            ),
            patch.object(launcher.Path, "is_dir", return_value=True),
            patch.object(launcher.Path, "is_file", return_value=True),
            patch.object(launcher, "read_json", side_effect=fake_read),
            patch.object(launcher, "check_file_metadata"),
            patch("socket.socket") as sock,
            patch.object(launcher.subprocess, "Popen") as popen,
            patch.object(launcher, "wait_ready"),
            patch.object(
                launcher, "run_stage", side_effect=stage_action or stage_error
            ) as stage,
            patch.object(launcher, "stop_process") as stop,
        ):
            sock.return_value.__enter__.return_value.connect_ex.return_value = 1
            if stage_error:
                with self.assertRaises(type(stage_error)):
                    launcher.execute_job(
                        serving, experiment, profile, code, commit,
                    )
            else:
                launcher.execute_job(
                    serving, experiment, profile, code, commit,
                    retry_failed=retry_failed,
                )
            env = popen.call_args.kwargs["env"]
            self.assertEqual(
                env["VLLM_LOGGING_LEVEL"],
                (logging_env or {}).get("VLLM_LOGGING_LEVEL", "WARNING"),
            )
            for name, value in (logging_env or {}).items():
                self.assertEqual(env[name], value)
            self.assertIsNone(popen.call_args.kwargs["stdout"])
            self.assertIsNone(popen.call_args.kwargs["stderr"])
            stop.assert_called_once_with(popen.return_value)
        metadata = root / launcher.RUN_ROOT / ".modal" / experiment["run_name"]
        return read_json(metadata / "status.json"), stage, commit

    def test_completion_limit_forwarded_to_pipeline(self) -> None:
        """Read the capped configuration from the actual subprocess argument path."""
        seen = []

        def stage_action(command, *args) -> None:
            """Inspect only the synthetic experiment file while it exists."""
            if "inference.run_pipeline" in command[2]:
                path = Path(command[command.index("--experiment") + 1])
                seen.append(read_json(path)["max_completion_tokens"])

        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                self._execute(
                    directory, stage_action=stage_action,
                    experiment_changes={"max_completion_tokens": 2048},
                )
        self.assertEqual(seen, [2048, 2048])

    def test_completion_limit_provenance(self) -> None:
        """Normalize null and keep added, changed, or removed caps incompatible."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            experiment = example("modal-experiment.example.json")
            write_json(path, {"experiment": experiment})
            omitted = {"experiment": experiment}
            null_limit = {"experiment": dict(experiment, max_completion_tokens=None)}
            launcher.require_provenance(path, null_limit)
            write_json(path, null_limit)
            launcher.require_provenance(path, omitted)
            with self.assertRaisesRegex(ValueError, "new run_name"):
                launcher.require_provenance(path, {
                    "experiment": dict(experiment, max_completion_tokens=2048),
                })
            capped = dict(experiment, max_completion_tokens=2048)
            write_json(path, {"experiment": capped})
            launcher.require_provenance(path, {"experiment": capped})
            for limit in (None, 1024):
                with self.assertRaisesRegex(ValueError, "new run_name"):
                    launcher.require_provenance(path, {
                        "experiment": dict(experiment, max_completion_tokens=limit),
                    })

    def test_job_resumes_with_new_logging_environment(self) -> None:
        """Pass changed logging settings to the server on the same saved run."""
        with tempfile.TemporaryDirectory() as directory:
            self._execute(directory)
            for interval in ("60", "120"):
                self._execute(directory, logging_env={
                    "VLLM_LOGGING_LEVEL": "INFO", "VLLM_LOG_STATS_INTERVAL": interval,
                })

    def test_complete_job_runs_pipeline_then_evaluation(self) -> None:
        """Use the authoritative module entrypoints and persist completion."""
        with tempfile.TemporaryDirectory() as directory:
            status, stage, commit = self._execute(directory, retry_failed=True)
            self.assertEqual(status, {"status": "complete"})
            self.assertIn("inference.run_pipeline", stage.call_args_list[0].args[0][2])
            self.assertIn("evaluation.evaluate", stage.call_args_list[1].args[0][2])
            self.assertGreaterEqual(commit.call_count, 2)
            self.assertNotIn("--enable-short-explanation", stage.call_args_list[0].args[0])
            self.assertIn("--retry-failed", stage.call_args_list[0].args[0])
            self.assertNotIn("--retry-failed", stage.call_args_list[1].args[0])

    def test_failed_job_preserves_outputs_and_skips_evaluation(self) -> None:
        """A failed pipeline still commits and cleans up, without scoring gaps."""
        with tempfile.TemporaryDirectory() as directory:
            status, stage, commit = self._execute(
                directory, RuntimeError("synthetic failure")
            )
            self.assertEqual(status, {"status": "failed"})
            self.assertEqual(stage.call_count, 1)
            self.assertGreaterEqual(commit.call_count, 2)

    def test_changed_source_and_profile_resume_only_missing_requests(self) -> None:
        """Reuse responses after context expansion and infrastructure changes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "original"
            experiment = example("modal-experiment.example.json")
            run_dir = root / launcher.RUN_ROOT / experiment["run_name"]
            output = run_dir / "responses" / "synthetic.jsonl"
            metadata = run_dir.parent / ".modal" / experiment["run_name"]

            def finalize(record: dict) -> None:
                """Retain the parsed synthetic answer for saved-row validation."""
                record["answer"] = record["parsed_output"]["answer"]

            specs = [
                {
                    "result_id": name, "model_input_id": f"input-{name}",
                    "safe_model_input": [{"role": "user", "content": name}],
                    "response_type": "final", "timepoint_count_for_parser": 3,
                    "task_for_parser": "trajectory_shape",
                    "identity_fields": ("result_id", "model_input_id"),
                    "finalize": finalize,
                }
                for name in ("first", "second")
            ]
            response = {"kind": "text", "raw_text": '{"answer":"constant"}'}
            client = Mock()
            client.submit.side_effect = [response, RuntimeError("logic bug")]
            summaries = []

            def run_pipeline(command, *args) -> None:
                """Run the real response persistence and reuse path with a mock transport."""
                if "inference.run_pipeline" not in command[2]:
                    return
                summaries.append(execute_specs(
                    specs, output, "result_id", client=client, model="model",
                    image_root=None, workers=1, progress=False,
                ))

            with patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "run_change_labels.submit_messages", side_effect=client.submit,
            ):
                self._execute(
                    str(root), RuntimeError(), code="old-code",
                    stage_action=run_pipeline,
                    context_limit=28000,
                    max_num_seqs=1,
                    experiment_changes={"max_completion_tokens": 512},
                )
                legacy = read_json(metadata / "provenance.json")
                legacy["enable_short_explanation"] = True
                write_json(metadata / "provenance.json", legacy)
                original_provenance = (metadata / "provenance.json").read_bytes()
                saved_response = output.read_bytes()
                self.assertEqual(len(read_jsonl(output)), 1)
                # Simulate operator-provided destination Volume contents using
                # only small synthetic metadata and text responses on local disk.
                destination = Path(directory) / "relocated"
                shutil.copytree(root, destination)
                output = destination / output.relative_to(root)
                metadata = destination / metadata.relative_to(root)
                profile_changes = {
                    "gpu": "A100", "timeout": 43200,
                    "startup_timeout": 1200, "commit_interval": 300,
                    "image": "vllm/vllm-openai:v0.28.0",
                    "volume_prefix": "relocated",
                }
                client.submit.reset_mock(side_effect=True)
                client.submit.return_value = response
                status, _, _ = self._execute(
                    str(destination), code="fixed-code", stage_action=run_pipeline,
                    profile_changes=profile_changes, versions={"vllm": "updated"},
                    context_limit=32768,
                    max_num_seqs=128,
                    experiment_changes={"max_completion_tokens": 512},
                )
                self.assertEqual(status, {"status": "complete"})
                client.submit.assert_called_once()
                self.assertEqual(client.submit.call_args.args[2], specs[1]["safe_model_input"])
                self.assertEqual(summaries[-1]["reused"], 1)
                self.assertEqual(summaries[-1]["submitted"], 1)
                self.assertEqual(len(read_jsonl(output)), 2)
                self.assertTrue(output.read_bytes().startswith(saved_response))

                client.submit.reset_mock()
                specs[0]["model_input_id"] = "changed-input"
                with self.assertRaisesRegex(ValueError, "identity conflict"):
                    run_pipeline(["python", "-m", "inference.run_pipeline"])
                client.submit.assert_not_called()

            self.assertEqual((metadata / "provenance.json").read_bytes(), original_provenance)
            attempts = list((metadata / "attempts").iterdir())
            self.assertEqual(len(attempts), 2)
            history = {
                read_json(path / "provenance.json")["provenance"]["source_sha256"]:
                read_json(path / "status.json")["status"]
                for path in attempts
            }
            self.assertEqual(history, {"old-code": "failed", "fixed-code": "complete"})
            resumed = next(
                read_json(path / "provenance.json")["provenance"]
                for path in attempts
                if read_json(path / "provenance.json")["provenance"]["source_sha256"]
                == "fixed-code"
            )
            for key, value in profile_changes.items():
                self.assertEqual(resumed["profile"][key], value)
            self.assertEqual(resumed["versions"], {"vllm": "updated"})
            self.assertIn("32768", resumed["serving"]["vllm_args"])
            self.assertIn("128", resumed["serving"]["vllm_args"])

    def test_partial_response_tail_is_backed_up_and_recovered(self) -> None:
        """Retain complete responses and archive a killed writer's partial tail."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "run/responses"
            responses.mkdir(parents=True)
            path = responses / "pair.jsonl"
            original = b'{"complete":true}\n{"incomplete":'
            path.write_bytes(original)
            launcher.recover_response_tails(root / "run", root / "metadata")
            self.assertEqual(path.read_bytes(), b'{"complete":true}\n')
            backup = next((root / "metadata/recovery").iterdir())
            self.assertEqual(backup.read_bytes(), original)

    def test_valid_unterminated_record_is_preserved(self) -> None:
        """A completed JSON object missing its newline must not be resubmitted."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "responses").mkdir()
            path = root / "responses/pair.jsonl"
            path.write_bytes(b'{"complete":true}')
            launcher.recover_response_tails(root, root / "metadata")
            self.assertEqual(path.read_bytes(), b'{"complete":true}\n')

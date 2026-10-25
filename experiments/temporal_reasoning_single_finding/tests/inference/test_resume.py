"""Tests for deterministic registries and run preparation resume checks."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import (
    build_registries,
    prepare_run,
)
from experiments.temporal_reasoning_single_finding.inference.run_change_labels import (
    execute_specs,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS


def _manifest() -> dict[str, str]:
    """Return a minimal compatible benchmark manifest."""
    return {
        "benchmark_version": "temporal-reasoning-smoke-v1",
        "pair_prompt_version": "pair-change-v1",
        "seq_prompt_version": "sequence-change-v1",
        "final_prompt_version": "final-tasks-v1",
        "json_schema_version": "v1",
    }


def _item(identifier: str, count: int, labels: list[str]) -> dict:
    """Create one all-task-eligible frozen benchmark item."""
    from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
        derive_answer,
    )

    images = [
        {
            "timepoint": f"T{index}",
            "image_id": f"train/patient/study{index}/view1_frontal.jpg",
            "relative_path": f"train/patient/study{index}/view1_frontal.png",
        }
        for index in range(1, count + 1)
    ]
    return {
        "benchmark_version": "temporal-reasoning-smoke-v1",
        "trajectory_id": f"chexpert_plus:{identifier}",
        "dataset": "chexpert_plus",
        "annotation_source": "silver",
        "source_sequence_id": identifier,
        "patient_id": "patient",
        "timepoint_count": count,
        "target_finding": {"canonical_name": "edema", "display_name": "edema"},
        "images": images,
        "source_interval_labels": ["Worse"] * (count - 1),
        "reference_change_labels": labels,
        "tasks": {
            task: {
                "eligible": True,
                "reference_answer": derive_answer(task, labels),
                "exclusion_reason": None,
            }
            for task in TASKS
        },
    }


class RegistryAndResumeTests(unittest.TestCase):
    """Verify reuse, privacy, preparation, and conflict behavior."""

    def test_registry_reuses_overlapping_pair_and_separates_results(self) -> None:
        """Overlapping PAIR input is reused while all final settings remain distinct."""
        items = [
            _item("three", 3, ["deterioration", "deterioration"]),
            _item("four", 4, ["deterioration", "deterioration", "deterioration"]),
        ]
        registries = build_registries({"model": "model"}, _manifest(), items)
        self.assertEqual(len(registries["pair_inputs"]), 3)
        self.assertEqual(len(registries["seq_inputs"]), 2)
        self.assertEqual(len(registries["final_cases"]), 8)
        for case in registries["final_cases"]:
            self.assertEqual(len(set(case["result_ids"].values())), 6)
        serialized = str(registries["model_inputs"])
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("reference_answer", serialized)

    def test_pair_reference_conflict_is_fatal(self) -> None:
        """One source-specific PAIR input cannot have conflicting references."""
        first = _item("one", 3, ["deterioration", "deterioration"])
        second = _item("two", 3, ["improvement", "deterioration"])
        with self.assertRaisesRegex(ValueError, "PAIR model input has conflicting"):
            build_registries({"model": "model"}, _manifest(), [first, second])

    def test_preflight_accepts_current_and_legacy_prepared_manifests(self) -> None:
        """Preflight uses the same portable identity and null effort as preparation."""
        from experiments.temporal_reasoning_single_finding.validation.preflight import (
            ValidatedBenchmark, _check_run_target,
        )

        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path = root / "benchmark.json"
                write_json(manifest_path, _manifest())
                experiment = {
                    "run_name": "resume-test", "model": "model",
                    "base_url": "http://unused/v1", "benchmark_path": str(manifest_path),
                }
                items = [_item("one", 3, ["deterioration", "improvement"])]
                with patch(
                    "experiments.temporal_reasoning_single_finding.inference."
                    "prepare_run.repository_root", return_value=root,
                ):
                    prepared = prepare_run(experiment, manifest_path, _manifest(), items)
                run_dir = prepared["run_dir"]
                saved_path = run_dir / "run_manifest.json"
                if legacy:
                    saved = read_json(saved_path)
                    saved["benchmark_manifest_path"] = "/old-machine/benchmark.json"
                    saved["experiment"]["reasoning_effort"] = None
                    saved["experiment"]["max_completion_tokens"] = None
                    write_json(saved_path, saved)
                before = saved_path.read_bytes()
                benchmark = ValidatedBenchmark(manifest_path, _manifest(), {3: items}, items)
                _check_run_target(run_dir, experiment, benchmark)
                _check_run_target(
                    run_dir, dict(experiment, reasoning_effort=None, max_completion_tokens=None),
                    benchmark,
                )
                self.assertEqual(saved_path.read_bytes(), before)
                for field, value in (
                    ("model", "different"), ("reasoning_effort", "high"),
                    ("max_completion_tokens", 2048),
                ):
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        _check_run_target(run_dir, dict(experiment, **{field: value}), benchmark)
                write_json(manifest_path, dict(_manifest(), changed=True))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    _check_run_target(run_dir, experiment, benchmark)

    def test_matching_resume_reuses_preparation_and_conflict_stops(self) -> None:
        """A run resumes only under its exact run manifest."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "benchmark" / "manifest.json"
            write_json(manifest_path, _manifest())
            experiment = {
                "run_name": "smoke-001",
                "model": "model",
                "base_url": "http://server/v1",
                "benchmark_path": str(manifest_path),
            }
            with patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "prepare_run.repository_root",
                return_value=root,
            ):
                items = [_item("one", 3, ["deterioration", "improvement"])]
                first = prepare_run(
                    experiment,
                    manifest_path,
                    _manifest(),
                    items,
                )
                second = prepare_run(
                    experiment,
                    manifest_path,
                    _manifest(),
                    items,
                )
                self.assertEqual(first["run_dir"], second["run_dir"])
                changed = dict(experiment, base_url="http://other/v1")
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prepare_run(changed, manifest_path, _manifest(), items)

    def test_relocation_preserves_new_and_legacy_manifests_on_resume(self) -> None:
        """A relative benchmark path resumes after moving the repository."""
        for legacy in (False, True):
            with (
                self.subTest(legacy=legacy),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                original = root / "original"
                relative_benchmark = Path("benchmark/manifest.json")
                manifest_path = original / relative_benchmark
                write_json(manifest_path, _manifest())
                experiment = {
                    "run_name": "smoke-001",
                    "model": "model",
                    "base_url": "http://server/v1",
                    "benchmark_path": str(relative_benchmark),
                }
                items = [_item("one", 3, ["deterioration", "improvement"])]
                module = (
                    "experiments.temporal_reasoning_single_finding.inference."
                    "prepare_run.repository_root"
                )
                with patch(module, return_value=original):
                    result = prepare_run(experiment, manifest_path, _manifest(), items)
                relative_run = Path(result["run_dir"]).relative_to(original)
                saved_path = original / relative_run / "run_manifest.json"
                saved = read_json(saved_path)
                self.assertNotIn("benchmark_manifest_path", saved)
                self.assertEqual(
                    saved["experiment"]["benchmark_path"], str(relative_benchmark)
                )
                if legacy:
                    saved["benchmark_manifest_path"] = str(manifest_path)
                    write_json(saved_path, saved)
                digest = sha256_file(saved_path)
                moved = root / "moved"
                original.rename(moved)
                with patch(module, return_value=moved):
                    prepare_run(
                        experiment, moved / relative_benchmark, _manifest(), items
                    )
                    self.assertEqual(
                        sha256_file(moved / relative_run / "run_manifest.json"), digest
                    )
                    write_json(
                        moved / relative_benchmark, {**_manifest(), "changed": True}
                    )
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        prepare_run(
                            experiment, moved / relative_benchmark, _manifest(), items
                        )

    def test_response_resume_requires_exact_model_input_id(self) -> None:
        """A completed response is reused only under the same safe model input."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            calls: list[str] = []

            def finalize(record: dict) -> None:
                """Extract the test answer."""
                record["predicted_answer"] = record["parsed_output"]["answer"]
                record["correct"] = True

            spec = {
                "result_id": "result",
                "model_input_id": "input-one",
                "safe_model_input": [],
                "response_type": "final",
                "timepoint_count_for_parser": 3,
                "task_for_parser": "trajectory_shape",
                "identity_fields": ("result_id", "model_input_id"),
                "finalize": finalize,
            }

            def submit(*_args, **_kwargs):
                """Return one valid terminal response and count submissions."""
                calls.append("called")
                return {
                    "kind": "text",
                    "raw_text": '{"answer":"constant"}',
                    "error": None,
            }

            with patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "run_change_labels.submit_messages",
                side_effect=submit,
            ):
                first = execute_specs(
                    [spec],
                    path,
                    "result_id",
                    client=object(),
                    model="model",
                    image_root=None,
                    workers=1,
                    progress=False,
                )
                second = execute_specs(
                    [spec],
                    path,
                    "result_id",
                    client=object(),
                    model="model",
                    image_root=None,
                    workers=1,
                    progress=False,
                )
            self.assertEqual(first["submitted"], 1)
            self.assertEqual(second["reused"], 1)
            self.assertEqual(len(calls), 1)
            changed = dict(spec, model_input_id="input-two")
            with self.assertRaisesRegex(ValueError, "model_input_id"):
                execute_specs(
                    [changed],
                    path,
                    "result_id",
                    client=object(),
                    model="model",
                    image_root=None,
                    workers=1,
                    progress=False,
                )
            self.assertEqual(len(read_jsonl(path)), 1)

    def test_response_resume_rejects_malformed_saved_row(self) -> None:
        """A known ID is not complete unless its whole response row is valid."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"

            def finalize(record: dict) -> None:
                """Extract the test answer."""
                output = record["parsed_output"]
                record["predicted_answer"] = (
                    output["answer"] if record["status"] == "valid" else None
                )
                record["correct"] = record["predicted_answer"] == "constant"

            spec = {
                "result_id": "result",
                "model_input_id": "input-one",
                "safe_model_input": [],
                "response_type": "final",
                "timepoint_count_for_parser": 3,
                "task_for_parser": "trajectory_shape",
                "identity_fields": ("result_id", "model_input_id"),
                "finalize": finalize,
            }
            write_jsonl(
                path,
                [{
                    "result_id": "result",
                    "model_input_id": "input-one",
                    "status": "upstream_invalid",
                }],
            )
            with self.assertRaisesRegex(ValueError, "invalid status"):
                execute_specs(
                    [spec],
                    path,
                    "result_id",
                    client=object(),
                    model="model",
                    image_root=None,
                    workers=1,
                    progress=False,
                )

    def test_concurrent_fatal_error_drains_and_persists_running_successes(self) -> None:
        """A fatal worker error does not discard another submitted call's result."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            success_started = threading.Event()
            release_success = threading.Event()
            attempts: list[str] = []

            def make_spec(identifier: str) -> dict:
                """Create one independently identified final-response spec."""
                def finalize(record: dict) -> None:
                    """Extract the test answer."""
                    output = record["parsed_output"]
                    record["predicted_answer"] = (
                        output["answer"] if record["status"] == "valid" else None
                    )
                    record["correct"] = record["predicted_answer"] == "constant"

                return {
                    "result_id": identifier,
                    "model_input_id": f"input-{identifier}",
                    "safe_model_input": [{"identifier": identifier}],
                    "response_type": "final",
                    "timepoint_count_for_parser": 3,
                    "task_for_parser": "trajectory_shape",
                    "identity_fields": ("result_id", "model_input_id"),
                    "finalize": finalize,
                }

            fatal_attempts = 0

            def submit(_client, _model, safe_messages, _image_root):
                """Fail one first attempt while the other call is already running."""
                nonlocal fatal_attempts
                identifier = safe_messages[0]["identifier"]
                attempts.append(identifier)
                if identifier == "success":
                    success_started.set()
                    self.assertTrue(release_success.wait(timeout=2))
                elif fatal_attempts == 0:
                    fatal_attempts += 1
                    self.assertTrue(success_started.wait(timeout=2))
                    release_success.set()
                    raise RuntimeError("fatal setup error")
                return {
                    "kind": "text",
                    "raw_text": '{"answer":"constant"}',
                    "error": None,
                }

            specs = [make_spec("fatal"), make_spec("success")]
            with patch(
                "experiments.temporal_reasoning_single_finding.inference."
                "run_change_labels.submit_messages",
                side_effect=submit,
            ):
                with self.assertRaisesRegex(RuntimeError, "fatal setup error"):
                    execute_specs(
                        specs,
                        path,
                        "result_id",
                        client=object(),
                        model="model",
                        image_root=None,
                        workers=2,
                        progress=False,
                    )
                self.assertEqual(
                    [row["result_id"] for row in read_jsonl(path)],
                    ["success"],
                )
                resumed = execute_specs(
                    specs,
                    path,
                    "result_id",
                    client=object(),
                    model="model",
                    image_root=None,
                    workers=2,
                    progress=False,
                )

            self.assertEqual(resumed["reused"], 1)
            self.assertEqual(resumed["submitted"], 1)
            self.assertEqual(attempts.count("success"), 1)
            self.assertEqual(attempts.count("fatal"), 2)
            self.assertEqual(len(read_jsonl(path)), 2)


if __name__ == "__main__":
    unittest.main()

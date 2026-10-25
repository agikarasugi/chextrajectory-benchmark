"""Verify run-wide request option forwarding and resume isolation."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import read_json, write_json
from experiments.temporal_reasoning_single_finding.inference import (
    run_change_labels, run_lc, run_seq_dir,
)
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    derive_fixed_code,
)
from experiments.temporal_reasoning_single_finding.inference.experiment_config import (
    require_matching_request_options,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import prepare_run
from experiments.temporal_reasoning_single_finding.protocol.contracts import interval_keys
from experiments.temporal_reasoning_single_finding.tests.inference.test_resume import (
    _item, _manifest,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    _check_run_target,
)


class RequestOptionTests(unittest.TestCase):
    """Exercise all inference stages using synthetic responses only."""

    def test_timeout_changes_pass_preparation_and_preflight(self) -> None:
        """Resume across timeout defaults and values without rewriting frozen artifacts."""
        options = ({}, {"request_timeout_seconds": None},
                   {"request_timeout_seconds": 120}, {"request_timeout_seconds": 600})
        for saved in options:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = _manifest()
                manifest_path = root / "benchmark.json"
                write_json(manifest_path, manifest)
                benchmark = SimpleNamespace(manifest=manifest, manifest_path=manifest_path)
                experiment = {
                    "run_name": "test", "model": "model", "base_url": "http://unused/v1",
                    "benchmark_path": str(manifest_path),
                }
                items = [_item("one", 3, ["deterioration", "improvement"])]
                with patch(
                    "experiments.temporal_reasoning_single_finding.inference."
                    "prepare_run.repository_root", return_value=root,
                ):
                    run_dir = prepare_run(
                        {**experiment, **saved}, manifest_path, manifest, items,
                    )["run_dir"]
                    original = {
                        path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()
                    }
                    for current in options:
                        with self.subTest(saved=saved, current=current):
                            updated = {**experiment, **current}
                            _check_run_target(run_dir, updated, benchmark)
                            prepare_run(updated, manifest_path, manifest, items)
                            require_matching_request_options(run_dir, updated)
                            for path, contents in original.items():
                                self.assertEqual(path.read_bytes(), contents)
                            changed = dict(updated, max_completion_tokens=1024)
                            with self.assertRaisesRegex(ValueError, "does not match"):
                                _check_run_target(run_dir, changed, benchmark)

    def test_all_stages_forward_options_and_reject_changed_resume(self) -> None:
        """Keep null equivalent to omission and freeze options across stages."""
        for effort, limit, workers in (
            (None, None, 1), ("none", None, 1), ("high", None, 2),
            (None, 2048, 1), (None, 2048, 2), ("high", 2048, 2),
        ):
            with (
                self.subTest(effort=effort, limit=limit, workers=workers),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                manifest = _manifest()
                manifest_path = root / "benchmark.json"
                write_json(manifest_path, manifest)
                experiment = {
                    "run_name": "test", "model": "model",
                    "base_url": "http://unused/v1",
                    "benchmark_path": str(manifest_path), "reasoning_effort": effort,
                    "max_completion_tokens": limit,
                }
                items = [_item("one", 3, ["deterioration", "improvement"])]
                prefix = "experiments.temporal_reasoning_single_finding.inference."
                with patch(prefix + "prepare_run.repository_root", return_value=root):
                    prepared = prepare_run(experiment, manifest_path, manifest, items)
                    run_dir = prepared["run_dir"]
                    saved = read_json(run_dir / "run_manifest.json")
                    self.assertEqual(
                        saved["experiment"].get("max_completion_tokens"), limit,
                    )
                    if limit is None:
                        self.assertNotIn("max_completion_tokens", saved["experiment"])
                    changed_limit = {**experiment, "max_completion_tokens": 1024}
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        prepare_run(changed_limit, manifest_path, manifest, items)
                    if effort is None:
                        self.assertNotIn("reasoning_effort", saved["experiment"])
                        omitted = {k: v for k, v in experiment.items() if k != "reasoning_effort"}
                        prepare_run(omitted, manifest_path, manifest, items)
                        # A manually retained null in a manifest is equivalent too.
                        saved["experiment"]["reasoning_effort"] = None
                        write_json(run_dir / "run_manifest.json", saved)
                        prepare_run(omitted, manifest_path, manifest, items)
                    else:
                        self.assertEqual(saved["experiment"]["reasoning_effort"], effort)
                    changed = {**experiment, "reasoning_effort": "low"}
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        prepare_run(changed, manifest_path, manifest, items)

                answers = {}
                for row in read_json(run_dir / "prepared/pair_inputs.json"):
                    answers[json.dumps(row["safe_model_input"])] = {
                        "change_label": row["reference_change_label"]
                    }
                for row in read_json(run_dir / "prepared/seq_inputs.json"):
                    answers[json.dumps(row["safe_model_input"])] = {
                        "change_labels": dict(zip(
                            interval_keys(row["timepoint_count"]),
                            row["reference_change_labels"], strict=True,
                        ))
                    }
                cases = read_json(run_dir / "prepared/final_cases.json")
                for spec in (
                    run_seq_dir.build_seq_dir_specs(run_dir, cases, "model")
                    + run_lc.build_ref_lc_specs(run_dir, cases, "model")
                ):
                    answers[json.dumps(spec["safe_model_input"])] = {
                        "answer": spec["reference_answer"]
                    }
                calls = []
                expected_options = {
                    key: value for key, value in {
                        "reasoning_effort": effort, "max_completion_tokens": limit,
                    }.items() if value is not None
                }

                def submit(_client, _model, messages, _image_root, **kwargs):
                    """Check transport options and return reference-derived test text."""
                    self.assertEqual(kwargs, expected_options)
                    calls.append(kwargs)
                    return {
                        "kind": "text", "raw_text": json.dumps(answers[json.dumps(messages)]),
                        "error": None,
                    }

                def stages(config):
                    """Provide each independently resumable model stage."""
                    return [
                        lambda: run_change_labels.run_change_labels(
                            run_dir, config, root / "unused", client=object(), workers=workers
                        ),
                        lambda: run_seq_dir.run_seq_dir(
                            run_dir, config, root / "unused", client=object(), workers=workers
                        ),
                        lambda: run_lc.run_ref_lc(run_dir, config, client=object(), workers=workers),
                        lambda: run_lc.run_predicted_lc(
                            run_dir, config, "PAIR", client=object(), workers=workers
                        ),
                        lambda: run_lc.run_predicted_lc(
                            run_dir, config, "SEQ", client=object(), workers=workers
                        ),
                    ]

                with ExitStack() as stack:
                    discoveries = [
                        stack.enter_context(patch.object(module, "require_exact_model"))
                        for module in (run_change_labels, run_seq_dir, run_lc)
                    ]
                    transport = stack.enter_context(patch(
                        prefix + "run_change_labels.submit_messages", side_effect=submit
                    ))
                    selected = stages(experiment)
                    selected[0]()
                    derive_fixed_code(run_dir)
                    for stage in selected[1:]:
                        before = len(calls)
                        stage()
                        self.assertGreater(len(calls), before)
                    before = transport.call_count
                    for stage in selected:
                        stage()
                    self.assertEqual(transport.call_count, before)
                    for timeout in (120, 600, None):
                        for stage in stages(dict(experiment, request_timeout_seconds=timeout)):
                            stage()
                    self.assertEqual(transport.call_count, before)
                    for changed_effort in ("low", "none" if effort is None else None):
                        for stage in stages({**experiment, "reasoning_effort": changed_effort}):
                            with self.assertRaisesRegex(ValueError, "new run_name"):
                                stage()
                    self.assertEqual(transport.call_count, before)
                    for discovery in discoveries:
                        discovery.reset_mock()
                    for changed_limit in (1024, None if limit is not None else 2048):
                        changed_config = dict(
                            experiment, max_completion_tokens=changed_limit,
                        )
                        for stage in stages(changed_config):
                            with self.assertRaisesRegex(ValueError, "new run_name"):
                                stage()
                    self.assertEqual(transport.call_count, before)
                    for discovery in discoveries:
                        discovery.assert_not_called()

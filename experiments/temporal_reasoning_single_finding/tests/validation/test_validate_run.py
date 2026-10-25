"""Tests for complete six-setting run coverage validation."""

import json
from functools import partial
import tempfile
import unittest
from pathlib import Path

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.temporal_reasoning_single_finding.inference.derive_fixed_code import (
    build_fixed_code_records,
)
from experiments.temporal_reasoning_single_finding.inference.prepare_run import (
    _prompt_preview,
    build_registries,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    FINAL_PROMPT_VERSION,
    JSON_SCHEMA_VERSION,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
)
from experiments.temporal_reasoning_single_finding.protocol.identities import (
    model_input_id,
)
from experiments.temporal_reasoning_single_finding.protocol.prompts import (
    render_lc_messages,
)
from experiments.temporal_reasoning_single_finding.validation.legacy_prompts import render_answer_only
from experiments.temporal_reasoning_single_finding.validation.validate_run import (
    validate_run,
)

from experiments.temporal_reasoning_single_finding.tests.validation import (
    test_preflight,
)


def _valid_pair(spec: dict[str, object]) -> dict[str, object]:
    """Create the terminal valid PAIR row produced by the runner."""
    row = {key: value for key, value in spec.items() if key != "safe_model_input"}
    label = spec["reference_change_label"]
    row.update(
        {
            "status": "valid",
            "raw_text": json.dumps({"change_label": label}, separators=(",", ":")),
            "parsed_output": {"change_label": label},
            "parsing_mode": "exact",
            "failure_reason": None,
            "response_error": None,
            "predicted_change_label": label,
            "correct": True,
        }
    )
    return row


def _valid_seq(spec: dict[str, object]) -> dict[str, object]:
    """Create the terminal valid SEQ row produced by the runner."""
    row = {key: value for key, value in spec.items() if key != "safe_model_input"}
    labels = list(spec["reference_change_labels"])
    fields = {
        f"T{index}_to_T{index + 1}": label
        for index, label in enumerate(labels, start=1)
    }
    row.update(
        {
            "status": "valid",
            "raw_text": json.dumps({"change_labels": fields}, separators=(",", ":")),
            "parsed_output": {"change_labels": fields},
            "parsing_mode": "exact",
            "failure_reason": None,
            "response_error": None,
            "predicted_change_labels": labels,
            "label_correctness": [True] * len(labels),
            "correct": True,
        }
    )
    return row


def _late_input(
    case: dict[str, object],
    setting: str,
    labels: list[str],
    model: str,
    *,
    legacy_answer_only: bool = False,
) -> dict[str, object]:
    """Create one exact safe predicted-label LC input record."""
    renderer = render_lc_messages
    if legacy_answer_only:
        renderer = partial(render_answer_only, renderer)
    messages = renderer(
        case["target_finding"]["display_name"], labels, case["task"],
    )
    identifier = model_input_id(
        safe_model_input=messages,
        model=model,
        user_prompt_version=FINAL_PROMPT_VERSION,
        json_schema_version=JSON_SCHEMA_VERSION,
        benchmark_version=case["benchmark_version"],
    )
    return {
        "record_type": setting,
        "record_id": case["result_ids"][setting],
        "model_input_id": identifier,
        "model": model,
        "user_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "benchmark_version": case["benchmark_version"],
        "safe_model_input": messages,
    }


def _valid_final(
    case: dict[str, object],
    setting: str,
    input_record: dict[str, object],
    labels: list[str] | None,
    dependencies: list[str],
) -> dict[str, object]:
    """Create one terminal valid model-backed final-answer row."""
    answer = case["reference_answer"]
    return {
        "result_id": case["result_ids"][setting],
        "final_case_id": case["final_case_id"],
        "setting": setting,
        "model_input_id": input_record["model_input_id"],
        "benchmark_version": case["benchmark_version"],
        "trajectory_id": case["trajectory_id"],
        "dataset": case["dataset"],
        "annotation_source": case["annotation_source"],
        "patient_id": case["patient_id"],
        "timepoint_count": case["timepoint_count"],
        "target_finding": case["target_finding"],
        "task": case["task"],
        "reference_answer": answer,
        "model": input_record["model"],
        "prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "dependency_ids": dependencies,
        "dependency_statuses": [],
        "consumed_change_labels": labels,
        "status": "valid",
        "raw_text": json.dumps({"answer": answer}, separators=(",", ":")),
        "parsed_output": {"answer": answer},
        "parsing_mode": "exact",
        "failure_reason": None,
        "response_error": None,
        "predicted_answer": answer,
        "correct": True,
    }


class ValidateRunTests(unittest.TestCase):
    """Build a complete metadata-only run and exercise coverage failures."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        for name in ("prepared", "responses", "derived", "validation"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        self.experiment = {
            "run_name": "fixture-run",
            "model": "fixture-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "benchmark_path": "/fixture/manifest.json",
        }
        self.manifest = {
            "benchmark_version": "fixture-compatible-v1",
            "pair_prompt_version": PAIR_PROMPT_VERSION,
            "seq_prompt_version": SEQ_PROMPT_VERSION,
            "final_prompt_version": FINAL_PROMPT_VERSION,
            "json_schema_version": JSON_SCHEMA_VERSION,
        }
        self.items = [test_preflight._item(3, "silver")]
        registries = build_registries(self.experiment, self.manifest, self.items)
        self.registries = registries

        benchmark_manifest_path = self.run_dir / "fixture_benchmark_manifest.json"
        write_json(benchmark_manifest_path, {"fixture": True})
        self.experiment["benchmark_path"] = str(benchmark_manifest_path)
        prompt_fixture = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "golden_prompts"
            / "sha256.json"
        )
        write_json(
            self.run_dir / "run_manifest.json",
            {
                "experiment": self.experiment,
                "benchmark_manifest_path": str(benchmark_manifest_path.resolve()),
                "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
                "benchmark_version": self.manifest["benchmark_version"],
                "pair_prompt_version": PAIR_PROMPT_VERSION,
                "seq_prompt_version": SEQ_PROMPT_VERSION,
                "final_prompt_version": FINAL_PROMPT_VERSION,
                "json_schema_version": JSON_SCHEMA_VERSION,
                "prompt_fixture_checksums": {
                    "golden_prompts/sha256.json": sha256_file(prompt_fixture)
                },
            },
        )
        prepared = self.run_dir / "prepared"
        write_json(prepared / "pair_inputs.json", registries["pair_inputs"])
        write_json(prepared / "seq_inputs.json", registries["seq_inputs"])
        write_json(prepared / "final_cases.json", registries["final_cases"])
        write_json(prepared / "prompt_preview.json", _prompt_preview())

        pair_rows = [_valid_pair(spec) for spec in registries["pair_inputs"]]
        seq_rows = [_valid_seq(spec) for spec in registries["seq_inputs"]]
        pair_by_id = {row["prediction_id"]: row for row in pair_rows}
        seq_by_id = {row["prediction_id"]: row for row in seq_rows}
        pair_fc, seq_fc = build_fixed_code_records(
            registries["final_cases"], pair_by_id, seq_by_id
        )
        write_jsonl(self.run_dir / "responses" / "pair.jsonl", pair_rows)
        write_jsonl(self.run_dir / "responses" / "seq.jsonl", seq_rows)
        write_json(self.run_dir / "derived" / "pair_fc.json", pair_fc)
        write_json(self.run_dir / "derived" / "seq_fc.json", seq_fc)

        initial_lookup = {
            (row["record_type"], row["record_id"]): row
            for row in registries["model_inputs"]
        }
        pair_fc_by_case = {row["final_case_id"]: row for row in pair_fc}
        seq_fc_by_case = {row["final_case_id"]: row for row in seq_fc}
        final_rows = {
            "SEQ-DIR": [],
            "REF-LC": [],
            "PAIR-LC": [],
            "SEQ-LC": [],
        }
        all_inputs = list(registries["model_inputs"])
        for case in registries["final_cases"]:
            seq_dir_input = initial_lookup[("SEQ-DIR", case["result_ids"]["SEQ-DIR"])]
            ref_input = initial_lookup[("REF-LC", case["result_ids"]["REF-LC"])]
            final_rows["SEQ-DIR"].append(
                _valid_final(case, "SEQ-DIR", seq_dir_input, None, [])
            )
            final_rows["REF-LC"].append(
                _valid_final(
                    case,
                    "REF-LC",
                    ref_input,
                    list(case["reference_change_labels"]),
                    [],
                )
            )
            for source, setting, fc_by_case in (
                ("PAIR", "PAIR-LC", pair_fc_by_case),
                ("SEQ", "SEQ-LC", seq_fc_by_case),
            ):
                fc = fc_by_case[case["final_case_id"]]
                labels = list(fc["consumed_change_labels"])
                input_record = _late_input(
                    case, setting, labels, self.experiment["model"]
                )
                all_inputs.append(input_record)
                final_rows[setting].append(
                    _valid_final(
                        case,
                        setting,
                        input_record,
                        labels,
                        list(fc["dependency_ids"]),
                    )
                )
        write_jsonl(prepared / "model_inputs.jsonl", all_inputs)
        filenames = {
            "SEQ-DIR": "seq_dir.jsonl",
            "REF-LC": "ref_lc.jsonl",
            "PAIR-LC": "pair_lc.jsonl",
            "SEQ-LC": "seq_lc.jsonl",
        }
        for setting, filename in filenames.items():
            write_jsonl(self.run_dir / "responses" / filename, final_rows[setting])

        prepared_files = (
            "pair_inputs.json",
            "seq_inputs.json",
            "final_cases.json",
            "model_inputs.jsonl",
            "prompt_preview.json",
        )
        write_json(
            prepared / "manifest.json",
            {
                "call_plan": registries["call_plan"],
                "counts": {
                    "pair_inputs": len(registries["pair_inputs"]),
                    "seq_inputs": len(registries["seq_inputs"]),
                    "final_cases": len(registries["final_cases"]),
                    "initial_model_inputs": len(registries["model_inputs"]),
                    "model_inputs": len(all_inputs),
                },
                "files": {
                    filename: sha256_file(prepared / filename)
                    for filename in prepared_files
                },
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_writes_complete_six_setting_coverage(self) -> None:
        coverage = validate_run(self.run_dir)

        self.assertEqual(coverage["status"], "passed")
        self.assertEqual(coverage["expected"]["final_cases"], 4)
        self.assertEqual(coverage["expected"]["final_results"], 24)
        self.assertEqual(
            coverage,
            read_json(self.run_dir / "validation" / "coverage.json"),
        )

    def test_rejects_missing_expected_final_result(self) -> None:
        path = self.run_dir / "responses" / "seq_lc.jsonl"
        saved = read_jsonl(path)
        write_jsonl(path, saved[:-1])

        with self.assertRaisesRegex(ValueError, "coverage differs"):
            validate_run(self.run_dir)

    def test_rejects_conflicting_fixed_code_result(self) -> None:
        path = self.run_dir / "derived" / "pair_fc.json"
        rows = read_json(path)
        rows[0]["consumed_change_labels"] = ["deterioration", "deterioration"]
        write_json(path, rows)

        with self.assertRaisesRegex(ValueError, "fixed code"):
            validate_run(self.run_dir)

    def test_rejects_tampered_model_input_fingerprint(self) -> None:
        path = self.run_dir / "prepared" / "model_inputs.jsonl"
        rows = read_jsonl(path)
        rows[0]["model_input_id"] = "0" * 64
        write_jsonl(path, rows)
        self._refresh_model_inputs_checksum_only()

        with self.assertRaisesRegex(ValueError, "fingerprint|conflicts"):
            validate_run(self.run_dir)

    def _refresh_model_inputs_checksum_only(self) -> None:
        """Refresh mutable prepared input metadata after a deliberate test edit."""
        prepared_manifest_path = self.run_dir / "prepared" / "manifest.json"
        prepared_manifest = read_json(prepared_manifest_path)
        prepared_manifest["files"]["model_inputs.jsonl"] = sha256_file(
            self.run_dir / "prepared" / "model_inputs.jsonl"
        )
        write_json(prepared_manifest_path, prepared_manifest)


if __name__ == "__main__":
    unittest.main()

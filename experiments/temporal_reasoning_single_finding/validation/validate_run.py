"""Validate complete prediction and six-setting result coverage."""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifact_io import (
    assert_no_image_payload,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from ..protocol.prompt_compatibility import normalize_prompt_fixture
from ..protocol.contracts import (
    CHANGE_LABELS,
    FAILURE_REASONS,
    FINAL_PROMPT_VERSION,
    JSON_SCHEMA_VERSION,
    NO_IDENTIFIABLE,
    LC_INFERENCE_POLICIES,
    lc_skip_status,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
    SETTINGS,
    STATUSES,
    answer_choices,
    interval_keys,
    require_exact_keys,
)
from ..protocol.identities import (
    final_case_id,
    model_input_id,
    pair_prediction_id,
    result_id,
    seq_prediction_id,
)
from ..protocol.task_rules import derive_answer
from ..protocol.task_selection import GOLD_PILOT_BENCHMARK_VERSION
from .preflight import _validate_registries, load_and_validate_benchmark


_MODEL_TERMINAL_STATUSES = {"valid", "invalid_response", "response_error"}
_PROPAGATED_STATUSES = {
    "upstream_invalid",
    "upstream_response_error",
    NO_IDENTIFIABLE,
}
_INITIAL_INPUT_TYPES = {"PAIR", "SEQ", "SEQ-DIR", "REF-LC"}
_MODEL_INPUT_FIELDS = {
    "record_type",
    "record_id",
    "model_input_id",
    "model",
    "user_prompt_version",
    "json_schema_version",
    "benchmark_version",
    "safe_model_input",
}
_RUN_MANIFEST_FIELDS = {
    "experiment",
    "benchmark_manifest_sha256",
    "benchmark_version",
    "pair_prompt_version",
    "seq_prompt_version",
    "final_prompt_version",
    "json_schema_version",
    "prompt_fixture_checksums",
}
_RESPONSE_NULL_FIELDS = {
    "raw_text",
    "parsed_output",
    "parsing_mode",
    "response_error",
    "predicted_answer",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a string-keyed mapping or reject the value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _array(value: Any, name: str) -> list[Any]:
    """Return an array or reject the value."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _identifier(value: Any, name: str) -> str:
    """Return one non-empty identifier string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _rows_by_id(
    path: Path,
    identifier_field: str,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load a JSON/JSONL result collection with exact ID coverage."""
    if not path.is_file():
        raise ValueError(f"Required run artifact is not a regular file: {path}")
    raw_rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    rows = _array(raw_rows, str(path))
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{path.name}[{index}]")
        identifier = _identifier(
            row.get(identifier_field), f"{path.name}[{index}].{identifier_field}"
        )
        if identifier in by_id:
            raise ValueError(f"Duplicate {identifier_field} in {path}: {identifier}")
        by_id[identifier] = row
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        extra = len(actual_ids - expected_ids)
        raise ValueError(
            f"{path.name} coverage differs: missing={missing}, extra={extra}"
        )
    return by_id


def _match_fields(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Sequence[str],
    name: str,
) -> None:
    """Require approved identity fields to match their prepared source."""
    for field in fields:
        if row.get(field) != expected.get(field):
            raise ValueError(f"{name} identity conflict: {field}")


def _validate_model_outcome(row: Mapping[str, Any], name: str) -> str:
    """Validate common terminal model response and parsing fields."""
    status = row.get("status")
    if status not in _MODEL_TERMINAL_STATUSES:
        raise ValueError(f"{name} has an invalid model-backed status: {status}")
    _identifier(row.get("model_input_id"), f"{name}.model_input_id")
    if type(row.get("correct")) is not bool:
        raise ValueError(f"{name}.correct must be a Boolean")
    if status == "valid":
        if not isinstance(row.get("raw_text"), str):
            raise ValueError(f"{name} valid response lacks raw text")
        if row.get("parsing_mode") not in {"exact", "fenced", "surrounding_text"}:
            raise ValueError(f"{name} has an invalid parsing mode")
        if not isinstance(row.get("parsed_output"), Mapping):
            raise ValueError(f"{name} valid response lacks parsed output")
        if (
            row.get("failure_reason") is not None
            or row.get("response_error") is not None
        ):
            raise ValueError(f"{name} valid response retains failure fields")
    elif status == "invalid_response":
        if row.get("failure_reason") not in FAILURE_REASONS:
            raise ValueError(f"{name} has an invalid failure reason")
        if row.get("parsed_output") is not None or row.get("parsing_mode") is not None:
            raise ValueError(f"{name} invalid response retains parsed output")
        if row.get("response_error") is not None:
            raise ValueError(f"{name} invalid response retains a technical error")
        if row.get("correct") is not False:
            raise ValueError(f"{name} invalid response must be incorrect")
    else:
        for field in ("raw_text", "parsed_output", "parsing_mode", "failure_reason"):
            if row.get(field) is not None:
                raise ValueError(f"{name} response_error has non-null {field}")
        error = _mapping(row.get("response_error"), f"{name}.response_error")
        require_exact_keys(error, {"type", "message"}, f"{name}.response_error")
        _identifier(error["type"], f"{name}.response_error.type")
        _identifier(error["message"], f"{name}.response_error.message")
        if row.get("correct") is not False:
            raise ValueError(f"{name} response_error must be incorrect")
    assert_no_image_payload(dict(row))
    return str(status)


def _validate_pair_prediction(
    row: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Validate one complete PAIR prediction record."""
    fields = (
        "prediction_id",
        "prediction_type",
        "model_input_id",
        "benchmark_version",
        "dataset",
        "annotation_source",
        "earlier_image_id",
        "later_image_id",
        "target_finding",
        "model",
        "prompt_version",
        "json_schema_version",
        "reference_change_label",
    )
    _match_fields(row, spec, fields, f"PAIR {spec['prediction_id']}")
    expected_id = pair_prediction_id(
        dataset=spec["dataset"],
        annotation_source=spec["annotation_source"],
        earlier_image_id=spec["earlier_image_id"],
        later_image_id=spec["later_image_id"],
        target_finding=spec["target_finding"]["canonical_name"],
        model=spec["model"],
        pair_prompt_version=spec["prompt_version"],
        json_schema_version=spec["json_schema_version"],
        benchmark_version=spec["benchmark_version"],
    )
    if expected_id != spec["prediction_id"]:
        raise ValueError("Prepared PAIR prediction ID is not canonical")
    status = _validate_model_outcome(row, f"PAIR {expected_id}")
    predicted = row.get("predicted_change_label")
    if status == "valid":
        parsed = _mapping(row["parsed_output"], f"PAIR {expected_id}.parsed_output")
        require_exact_keys(parsed, {"change_label"}, "PAIR parsed_output")
        if (
            parsed["change_label"] not in CHANGE_LABELS
            or predicted != parsed["change_label"]
        ):
            raise ValueError("PAIR valid output has an invalid change label")
        if row["correct"] != (predicted == row["reference_change_label"]):
            raise ValueError("PAIR correctness does not match its labels")
    elif predicted is not None:
        raise ValueError("Failed PAIR prediction retains a change label")


def _validate_seq_prediction(
    row: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Validate one complete SEQ prediction record."""
    fields = (
        "prediction_id",
        "prediction_type",
        "model_input_id",
        "benchmark_version",
        "source_sequence_id",
        "dataset",
        "annotation_source",
        "patient_id",
        "timepoint_count",
        "target_finding",
        "model",
        "prompt_version",
        "json_schema_version",
        "reference_change_labels",
    )
    _match_fields(row, spec, fields, f"SEQ {spec['prediction_id']}")
    expected_id = seq_prediction_id(
        source_sequence_id=spec["source_sequence_id"],
        model=spec["model"],
        seq_prompt_version=spec["prompt_version"],
        json_schema_version=spec["json_schema_version"],
        benchmark_version=spec["benchmark_version"],
    )
    if expected_id != spec["prediction_id"]:
        raise ValueError("Prepared SEQ prediction ID is not canonical")
    status = _validate_model_outcome(row, f"SEQ {expected_id}")
    predicted = row.get("predicted_change_labels")
    label_correctness = row.get("label_correctness")
    if status == "valid":
        parsed = _mapping(row["parsed_output"], f"SEQ {expected_id}.parsed_output")
        require_exact_keys(parsed, {"change_labels"}, "SEQ parsed_output")
        parsed_values = _mapping(parsed["change_labels"], "SEQ parsed change_labels")
        keys = interval_keys(spec["timepoint_count"])
        require_exact_keys(parsed_values, set(keys), "SEQ parsed change_labels")
        chronological = [parsed_values[key] for key in keys]
        if predicted != chronological or any(
            label not in CHANGE_LABELS for label in chronological
        ):
            raise ValueError(
                "SEQ valid output has an invalid chronological label vector"
            )
        expected_correctness = [
            value == reference
            for value, reference in zip(
                chronological, spec["reference_change_labels"], strict=True
            )
        ]
        if label_correctness != expected_correctness:
            raise ValueError("SEQ label correctness is inconsistent")
        if row["correct"] != all(expected_correctness):
            raise ValueError("SEQ vector correctness is inconsistent")
    elif predicted is not None or label_correctness is not None:
        raise ValueError("Failed SEQ prediction retains parsed label data")


def _expected_fc(
    case: Mapping[str, Any],
    source: str,
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the semantic outcome of one fixed-code setting."""
    dependency_ids = (
        list(case["pair_dependency_ids"])
        if source == "PAIR"
        else [case["seq_dependency_id"]]
    )
    failures = [
        {"dependency_id": identifier, "status": predictions[identifier]["status"]}
        for identifier in dependency_ids
        if predictions[identifier]["status"] != "valid"
    ]
    if failures:
        status = (
            "upstream_response_error"
            if any(value["status"] == "response_error" for value in failures)
            else "upstream_invalid"
        )
        return {
            "status": status,
            "dependency_ids": dependency_ids,
            "dependency_statuses": failures,
            "consumed_change_labels": None,
            "predicted_answer": None,
            "correct": False,
        }
    if source == "PAIR":
        labels = [
            predictions[value]["predicted_change_label"]
            for value in dependency_ids
        ]
    else:
        labels = list(predictions[dependency_ids[0]]["predicted_change_labels"])
    answer = derive_answer(case["task"], labels)
    return {
        "status": "valid" if answer != NO_IDENTIFIABLE else NO_IDENTIFIABLE,
        "dependency_ids": dependency_ids,
        "dependency_statuses": [],
        "consumed_change_labels": labels,
        "predicted_answer": answer if answer != NO_IDENTIFIABLE else None,
        "correct": answer != NO_IDENTIFIABLE and answer == case["reference_answer"],
    }


def _validate_fixed_result(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    setting: str,
    expected: Mapping[str, Any],
) -> None:
    """Validate one no-call fixed-code result against canonical dependencies."""
    _validate_final_identity(row, case, setting)
    if row.get("status") not in {"valid", *_PROPAGATED_STATUSES}:
        raise ValueError(f"{setting} has a status impossible for fixed code")
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{setting} result conflicts with fixed code: {field}")
    if row.get("model_input_id") is not None:
        raise ValueError(f"{setting} must not have a model-input ID")
    for field in (
        "raw_text",
        "parsed_output",
        "parsing_mode",
        "failure_reason",
        "response_error",
    ):
        if row.get(field) is not None:
            raise ValueError(f"{setting} fixed result has non-null {field}")
    assert_no_image_payload(dict(row))


def _validate_final_identity(
    row: Mapping[str, Any], case: Mapping[str, Any], setting: str
) -> None:
    """Match one final-result row to its exact prepared case and setting."""
    fields = (
        "final_case_id",
        "benchmark_version",
        "trajectory_id",
        "dataset",
        "annotation_source",
        "patient_id",
        "timepoint_count",
        "target_finding",
        "task",
        "reference_answer",
    )
    _match_fields(row, case, fields, f"{setting} {case['final_case_id']}")
    if row.get("setting") != setting:
        raise ValueError("Final result has the wrong setting")
    if row.get("result_id") != case["result_ids"][setting]:
        raise ValueError("Final result has the wrong result ID")
    if case["final_case_id"] != final_case_id(
        source_sequence_id=case["trajectory_id"], task=case["task"]
    ):
        raise ValueError("Prepared final-case ID is not canonical")
    if row["result_id"] != result_id(
        case_id=case["final_case_id"], setting=setting
    ):
        raise ValueError("Final result ID is not canonical")


def _validate_model_final(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    setting: str,
    fc_result: Mapping[str, Any] | None,
    *,
    lc_inference_policy: str | None = None,
) -> bool:
    """Validate one direct or LC result and return whether a call was submitted."""
    _validate_final_identity(row, case, setting)
    skip_status = None
    if fc_result is not None:
        skip_status = lc_skip_status(
            fc_result["status"], fc_result.get("predicted_answer"),
            case["reference_answer"], lc_inference_policy,
        )
    if skip_status is not None:
        if row.get("status") != skip_status:
            raise ValueError(f"{setting} does not match the required LC skip status")
        for field in (
            "dependency_ids",
            "dependency_statuses",
            "consumed_change_labels",
        ):
            if row.get(field) != fc_result.get(field):
                raise ValueError(f"{setting} propagation conflicts on {field}")
        if row.get("model_input_id") is not None or row.get("correct") is not False:
            raise ValueError(f"Skipped {setting} has model input or correctness")
        if row.get("failure_reason") is not None:
            raise ValueError(f"Skipped {setting} has a failure reason")
        for field in _RESPONSE_NULL_FIELDS:
            if row.get(field) is not None:
                raise ValueError(f"Skipped {setting} has non-null {field}")
        assert_no_image_payload(dict(row))
        return False

    if setting == "SEQ-DIR":
        expected_dependencies: list[str] = []
        expected_labels = None
    elif setting == "REF-LC":
        expected_dependencies = []
        expected_labels = list(case["reference_change_labels"])
    else:
        if fc_result is None:
            raise AssertionError("Predicted LC validation lacks its FC result")
        expected_dependencies = list(fc_result["dependency_ids"])
        expected_labels = list(fc_result["consumed_change_labels"])
    if row.get("dependency_ids") != expected_dependencies:
        raise ValueError(f"{setting} has the wrong dependencies")
    if row.get("dependency_statuses") != []:
        raise ValueError(f"Submitted {setting} has dependency failures")
    if row.get("consumed_change_labels") != expected_labels:
        raise ValueError(f"{setting} consumed labels differ from its source")
    status = _validate_model_outcome(row, f"{setting} {case['final_case_id']}")
    predicted = row.get("predicted_answer")
    if status == "valid":
        parsed = _mapping(row["parsed_output"], f"{setting}.parsed_output")
        require_exact_keys(parsed, {"answer"}, f"{setting}.parsed_output")
        if parsed["answer"] not in answer_choices(
            case["task"], case["timepoint_count"]
        ):
            raise ValueError(f"{setting} answer is outside its task vocabulary")
        if predicted != parsed["answer"]:
            raise ValueError(f"{setting} predicted answer differs from parsed output")
        if row["correct"] != (predicted == case["reference_answer"]):
            raise ValueError(f"{setting} correctness is inconsistent")
    elif predicted is not None:
        raise ValueError(f"Failed {setting} retains a predicted answer")
    return True


def _validate_model_inputs(
    rows: Sequence[dict[str, Any]],
    run_manifest: Mapping[str, Any],
    expected_called: set[tuple[str, str]],
    response_input_ids: Mapping[tuple[str, str], str],
) -> None:
    """Validate exact safe input fingerprints and submitted-call coverage."""
    experiment = _mapping(run_manifest.get("experiment"), "run manifest experiment")
    prompt_versions = {
        "PAIR": run_manifest["pair_prompt_version"],
        "SEQ": run_manifest["seq_prompt_version"],
        "SEQ-DIR": run_manifest["final_prompt_version"],
        "REF-LC": run_manifest["final_prompt_version"],
        "PAIR-LC": run_manifest["final_prompt_version"],
        "SEQ-LC": run_manifest["final_prompt_version"],
    }
    keys: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"model_inputs[{index}]")
        require_exact_keys(row, _MODEL_INPUT_FIELDS, f"model_inputs[{index}]")
        assert_no_image_payload(row)
        key = (row["record_type"], row["record_id"])
        if key in keys:
            raise ValueError("Model-input registry contains duplicate keys")
        keys.add(key)
        if row["model"] != experiment.get("model"):
            raise ValueError("Model-input registry has the wrong model")
        if row["record_type"] not in prompt_versions:
            raise ValueError("Model-input registry has an unsupported record type")
        if (
            row["user_prompt_version"] != prompt_versions[row["record_type"]]
            or row["json_schema_version"] != run_manifest["json_schema_version"]
            or row["benchmark_version"] != run_manifest["benchmark_version"]
        ):
            raise ValueError("Model-input registry has conflicting frozen versions")
        expected = model_input_id(
            safe_model_input=row["safe_model_input"],
            model=row["model"],
            user_prompt_version=row["user_prompt_version"],
            json_schema_version=row["json_schema_version"],
            benchmark_version=row["benchmark_version"],
        )
        if row["model_input_id"] != expected:
            raise ValueError("Model-input registry has a non-canonical fingerprint")
        if response_input_ids.get(key) != row["model_input_id"]:
            raise ValueError("Model-input registry conflicts with a response record")
    if keys != expected_called:
        missing = len(expected_called - keys)
        extra = len(keys - expected_called)
        raise ValueError(
            f"Model-input coverage differs: missing={missing}, extra={extra}"
        )


def _validate_run_manifest(path: Path) -> dict[str, Any]:
    """Validate the immutable run identity and its current benchmark provenance."""
    manifest = _mapping(read_json(path), "run manifest")
    # Accept the redundant path in legacy manifests without using it for
    # lookup or identity. Keep the saved artifact unchanged.
    required_manifest = {
        key: value for key, value in manifest.items()
        if key not in {"benchmark_manifest_path", "lc_inference_policy"}
    }
    if (
        "lc_inference_policy" in manifest
        and manifest["lc_inference_policy"] not in LC_INFERENCE_POLICIES
    ):
        raise ValueError("Unsupported LC inference policy")
    require_exact_keys(required_manifest, _RUN_MANIFEST_FIELDS, "run manifest")
    experiment = _mapping(manifest["experiment"], "run manifest experiment")
    from ..inference.experiment_config import validate_experiment

    validate_experiment(dict(experiment))
    for field in ("run_name", "model", "base_url", "benchmark_path"):
        _identifier(experiment[field], f"run manifest experiment.{field}")
    from ..inference.experiment_config import resolve_benchmark_path

    benchmark_path = resolve_benchmark_path(experiment)
    if not benchmark_path.is_file():
        raise ValueError(
            f"Run benchmark manifest is not a regular file: {benchmark_path}"
        )
    digest = _identifier(
        manifest["benchmark_manifest_sha256"], "benchmark manifest checksum"
    )
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Run benchmark manifest checksum is not lowercase SHA-256")
    if sha256_file(benchmark_path) != digest:
        raise ValueError("Run benchmark manifest checksum no longer matches")
    expected_versions = {
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "seq_prompt_version": SEQ_PROMPT_VERSION,
        "final_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
    }
    for field, expected in expected_versions.items():
        if manifest[field] != expected:
            raise ValueError(f"Run manifest has unsupported {field}")
    _identifier(manifest["benchmark_version"], "run benchmark_version")
    normalize_prompt_fixture(manifest)
    return manifest


def _load_prepared(run_dir: Path) -> dict[str, Any]:
    """Load and checksum the immutable and append-updated prepared registries."""
    prepared = run_dir / "prepared"
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run lacks prepared/manifest.json")
    manifest = _mapping(read_json(manifest_path), "prepared manifest")
    require_exact_keys(manifest, {"call_plan", "counts", "files"}, "prepared manifest")
    files = _mapping(manifest["files"], "prepared manifest files")
    expected_files = {
        "pair_inputs.json",
        "seq_inputs.json",
        "final_cases.json",
        "model_inputs.jsonl",
        "prompt_preview.json",
    }
    require_exact_keys(files, expected_files, "prepared manifest files")
    for filename, digest in files.items():
        path = prepared / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Prepared artifact checksum mismatch: {filename}")

    pair_inputs = _array(read_json(prepared / "pair_inputs.json"), "pair_inputs")
    seq_inputs = _array(read_json(prepared / "seq_inputs.json"), "seq_inputs")
    final_cases = _array(read_json(prepared / "final_cases.json"), "final_cases")
    model_inputs = read_jsonl(prepared / "model_inputs.jsonl")
    prompt_preview = _mapping(
        read_json(prepared / "prompt_preview.json"), "prompt preview"
    )
    if len(prompt_preview) != 19:
        raise ValueError("Prompt preview does not contain all 19 variants")
    assert_no_image_payload(prompt_preview)

    initial_inputs = [
        row for row in model_inputs if row.get("record_type") in _INITIAL_INPUT_TYPES
    ]
    registries = {
        "pair_inputs": pair_inputs,
        "seq_inputs": seq_inputs,
        "final_cases": final_cases,
        "model_inputs": initial_inputs,
        "call_plan": manifest["call_plan"],
    }
    _validate_registries(registries, seq_inputs)
    counts = _mapping(manifest["counts"], "prepared manifest counts")
    expected_counts = {
        "pair_inputs": len(pair_inputs),
        "seq_inputs": len(seq_inputs),
        "final_cases": len(final_cases),
        "initial_model_inputs": len(initial_inputs),
        "model_inputs": len(model_inputs),
    }
    if counts != expected_counts:
        raise ValueError("Prepared manifest counts disagree with current registries")
    return {
        "pair_inputs": pair_inputs,
        "seq_inputs": seq_inputs,
        "final_cases": final_cases,
        "model_inputs": model_inputs,
        "call_plan": manifest["call_plan"],
    }


@dataclass
class ValidatedRun:
    """Complete saved inputs and outcomes loaded without changing the run."""

    manifest: dict[str, Any]
    prepared: dict[str, Any]
    pair_predictions: dict[str, dict[str, Any]]
    seq_predictions: dict[str, dict[str, Any]]
    results: dict[str, dict[str, dict[str, Any]]]
    coverage: dict[str, Any]


def load_validated_run(run_dir: Path) -> ValidatedRun:
    """Require complete validly identified artifacts without writing files."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory is not a directory: {run_dir}")
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError("Run lacks run_manifest.json")
    run_manifest = _validate_run_manifest(run_manifest_path)
    prepared = _load_prepared(run_dir)
    from ..inference.experiment_config import resolve_benchmark_path
    from ..inference.prepare_run import build_registries
    from .legacy_prompts import build_answer_only_registries

    benchmark_path = resolve_benchmark_path(run_manifest["experiment"])
    benchmark_manifest = read_json(benchmark_path)
    if (
        "task_selection" in benchmark_manifest
        or run_manifest["benchmark_version"] == GOLD_PILOT_BENCHMARK_VERSION
    ):
        benchmark = load_and_validate_benchmark(benchmark_path)
        if benchmark.manifest["benchmark_version"] != run_manifest["benchmark_version"]:
            raise ValueError("Run benchmark version conflicts with its manifest")
        # The prompt mode is recorded in exact saved inputs, not a manifest flag.
        # Require one complete matching mode, never a mixture of registry modes.
        keys = ("pair_inputs", "seq_inputs", "final_cases", "call_plan")
        for builder in (build_registries, build_answer_only_registries):
            expected = builder(
                run_manifest["experiment"], benchmark.manifest, benchmark.items,
            )
            conflicts = [key for key in keys if prepared[key] != expected[key]]
            if not conflicts:
                break
        else:
            raise ValueError(
                f"Prepared {conflicts[0]} conflicts with frozen task assignments "
                "or supported prompt modes"
            )

    pair_specs = {
        row["prediction_id"]: row for row in prepared["pair_inputs"]
    }
    seq_specs = {row["prediction_id"]: row for row in prepared["seq_inputs"]}
    cases = {row["final_case_id"]: row for row in prepared["final_cases"]}
    if len(pair_specs) != len(prepared["pair_inputs"]):
        raise ValueError("Prepared PAIR registry contains duplicate IDs")
    if len(seq_specs) != len(prepared["seq_inputs"]):
        raise ValueError("Prepared SEQ registry contains duplicate IDs")
    if len(cases) != len(prepared["final_cases"]):
        raise ValueError("Prepared final-case registry contains duplicate IDs")

    experiment = run_manifest["experiment"]
    for spec in pair_specs.values():
        if (
            spec.get("model") != experiment["model"]
            or spec.get("benchmark_version") != run_manifest["benchmark_version"]
            or spec.get("prompt_version") != run_manifest["pair_prompt_version"]
            or spec.get("json_schema_version") != run_manifest["json_schema_version"]
        ):
            raise ValueError("Prepared PAIR registry conflicts with run identity")
    for spec in seq_specs.values():
        if (
            spec.get("model") != experiment["model"]
            or spec.get("benchmark_version") != run_manifest["benchmark_version"]
            or spec.get("prompt_version") != run_manifest["seq_prompt_version"]
            or spec.get("json_schema_version") != run_manifest["json_schema_version"]
        ):
            raise ValueError("Prepared SEQ registry conflicts with run identity")
    if any(
        case.get("benchmark_version") != run_manifest["benchmark_version"]
        for case in cases.values()
    ):
        raise ValueError("Prepared final case conflicts with run benchmark version")

    pair_rows = _rows_by_id(
        run_dir / "responses" / "pair.jsonl", "prediction_id", set(pair_specs)
    )
    seq_rows = _rows_by_id(
        run_dir / "responses" / "seq.jsonl", "prediction_id", set(seq_specs)
    )
    for identifier, spec in pair_specs.items():
        _validate_pair_prediction(pair_rows[identifier], spec)
    for identifier, spec in seq_specs.items():
        _validate_seq_prediction(seq_rows[identifier], spec)

    paths = {
        "SEQ-DIR": run_dir / "responses" / "seq_dir.jsonl",
        "PAIR-FC": run_dir / "derived" / "pair_fc.json",
        "REF-LC": run_dir / "responses" / "ref_lc.jsonl",
        "SEQ-FC": run_dir / "derived" / "seq_fc.json",
        "PAIR-LC": run_dir / "responses" / "pair_lc.jsonl",
        "SEQ-LC": run_dir / "responses" / "seq_lc.jsonl",
    }
    expected_result_ids = {
        setting: {case["result_ids"][setting] for case in cases.values()}
        for setting in SETTINGS
    }
    result_rows = {
        setting: _rows_by_id(path, "result_id", expected_result_ids[setting])
        for setting, path in paths.items()
    }
    for setting in ("SEQ-DIR", "REF-LC", "PAIR-LC", "SEQ-LC"):
        for row in result_rows[setting].values():
            if (
                row.get("model") != experiment["model"]
                or row.get("prompt_version") != run_manifest["final_prompt_version"]
                or row.get("json_schema_version") != run_manifest["json_schema_version"]
            ):
                raise ValueError(f"{setting} response conflicts with run identity")

    expected_called: set[tuple[str, str]] = {
        *(('PAIR', identifier) for identifier in pair_rows),
        *(('SEQ', identifier) for identifier in seq_rows),
    }
    response_input_ids: dict[tuple[str, str], str] = {
        **{
            ("PAIR", identifier): row["model_input_id"]
            for identifier, row in pair_rows.items()
        },
        **{
            ("SEQ", identifier): row["model_input_id"]
            for identifier, row in seq_rows.items()
        },
    }

    for case in cases.values():
        pair_expected = _expected_fc(case, "PAIR", pair_rows)
        seq_expected = _expected_fc(case, "SEQ", seq_rows)
        pair_fc = result_rows["PAIR-FC"][case["result_ids"]["PAIR-FC"]]
        seq_fc = result_rows["SEQ-FC"][case["result_ids"]["SEQ-FC"]]
        _validate_fixed_result(pair_fc, case, "PAIR-FC", pair_expected)
        _validate_fixed_result(seq_fc, case, "SEQ-FC", seq_expected)

        for setting, fc in (
            ("SEQ-DIR", None),
            ("REF-LC", None),
            ("PAIR-LC", pair_fc),
            ("SEQ-LC", seq_fc),
        ):
            row = result_rows[setting][case["result_ids"][setting]]
            submitted = _validate_model_final(
                row, case, setting, fc,
                lc_inference_policy=run_manifest.get("lc_inference_policy"),
            )
            if submitted:
                key = (setting, row["result_id"])
                expected_called.add(key)
                response_input_ids[key] = row["model_input_id"]

    _validate_model_inputs(
        prepared["model_inputs"],
        run_manifest,
        expected_called,
        response_input_ids,
    )

    prediction_statuses = {
        "PAIR": dict(
            sorted(Counter(row["status"] for row in pair_rows.values()).items())
        ),
        "SEQ": dict(
            sorted(Counter(row["status"] for row in seq_rows.values()).items())
        ),
    }
    result_statuses = {
        setting: dict(
            sorted(
                Counter(row["status"] for row in result_rows[setting].values()).items()
            )
        )
        for setting in SETTINGS
    }
    if any(
        status not in STATUSES
        for counts in result_statuses.values()
        for status in counts
    ):
        raise ValueError("Final result contains a status outside the frozen vocabulary")
    coverage = {
        "status": "passed",
        "expected": {
            "pair_predictions": len(pair_specs),
            "seq_predictions": len(seq_specs),
            "final_cases": len(cases),
            "final_results": len(cases) * len(SETTINGS),
            "results_by_setting": {setting: len(cases) for setting in SETTINGS},
        },
        "observed": {
            "pair_predictions": len(pair_rows),
            "seq_predictions": len(seq_rows),
            "final_results": sum(len(rows) for rows in result_rows.values()),
            "results_by_setting": {
                setting: len(result_rows[setting]) for setting in SETTINGS
            },
        },
        "status_counts": {
            "predictions": prediction_statuses,
            "results": result_statuses,
        },
    }
    assert_no_image_payload(coverage)
    return ValidatedRun(
        manifest=run_manifest,
        prepared=prepared,
        pair_predictions=pair_rows,
        seq_predictions=seq_rows,
        results=result_rows,
        coverage=coverage,
    )


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Require complete validly identified artifacts and write coverage.json."""
    run_dir = Path(run_dir).resolve()
    coverage = load_validated_run(run_dir).coverage
    write_json(run_dir / "validation" / "coverage.json", coverage)
    return coverage


def _parse_args() -> argparse.Namespace:
    """Parse the completed-run validation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate one run and print its coverage summary."""
    args = _parse_args()
    print(json.dumps(validate_run(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

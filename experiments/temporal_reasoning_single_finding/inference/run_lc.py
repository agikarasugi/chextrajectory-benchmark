"""Run fresh text-only REF-LC, PAIR-LC, and SEQ-LC requests."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifact_io import (
    append_jsonl,
    assert_no_image_payload,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from ..protocol.contracts import (
    FINAL_PROMPT_VERSION, JSON_SCHEMA_VERSION, LC_INFERENCE_POLICY,
    NO_IDENTIFIABLE, UPSTREAM_ANSWER_MISMATCH, lc_skip_status,
)
from ..protocol.task_rules import derive_answer
from ..protocol.identities import model_input_id
from ..protocol.prompts import render_lc_messages
from .experiment_config import (
    load_experiment,
    require_api_key,
    require_matching_request_options,
    run_directory,
)
from .model_client import create_client, require_exact_model
from .run_change_labels import execute_specs
from .run_seq_dir import final_response_spec


def _by_id(path: Path, identifier: str) -> dict[str, dict[str, Any]]:
    """Load an array or JSONL file as a duplicate-free ID mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected an array at {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(identifier)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"Invalid or duplicate {identifier} in {path}")
        result[value] = row
    return result


def _input_record(
    *,
    setting: str,
    result_identifier: str,
    messages: list[dict[str, Any]],
    model: str,
    benchmark_version: str,
) -> dict[str, Any]:
    """Create one safe late-stage LC model-input registry record."""
    record = {
        "record_type": setting,
        "record_id": result_identifier,
        "model_input_id": model_input_id(
            safe_model_input=messages,
            model=model,
            user_prompt_version=FINAL_PROMPT_VERSION,
            json_schema_version=JSON_SCHEMA_VERSION,
            benchmark_version=benchmark_version,
        ),
        "model": model,
        "user_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "safe_model_input": messages,
    }
    assert_no_image_payload(record)
    return record


def _append_model_inputs(run_dir: Path, records: list[dict[str, Any]]) -> None:
    """Append new LC inputs once and refresh prepared registry metadata."""
    path = run_dir / "prepared" / "model_inputs.jsonl"
    existing_rows = read_jsonl(path)
    existing = {
        (row.get("record_type"), row.get("record_id")): row
        for row in existing_rows
    }
    if len(existing) != len(existing_rows):
        raise ValueError("Prepared model-input registry contains duplicate entries")
    for record in records:
        key = (record["record_type"], record["record_id"])
        if key in existing:
            if existing[key] != record:
                raise ValueError(f"Prepared LC model input conflicts: {key}")
            continue
        append_jsonl(path, record)
        existing[key] = record
    prepared_manifest_path = run_dir / "prepared" / "manifest.json"
    prepared_manifest = read_json(prepared_manifest_path)
    prepared_manifest.setdefault("counts", {})["model_inputs"] = len(existing)
    prepared_manifest["files"]["model_inputs.jsonl"] = sha256_file(path)
    write_json(prepared_manifest_path, prepared_manifest)


def _input_lookup(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the model-input registry after rejecting duplicate keys."""
    rows = read_jsonl(run_dir / "prepared" / "model_inputs.jsonl")
    result = {
        (row.get("record_type"), row.get("record_id")): row
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("Prepared model-input registry contains duplicate entries")
    return result


def build_ref_lc_specs(
    run_dir: Path,
    final_cases: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Recompute every REF-LC input and verify its prepared fingerprint."""
    lookup = _input_lookup(run_dir)
    specs: list[dict[str, Any]] = []
    for case in final_cases:
        result_identifier = case["result_ids"]["REF-LC"]
        labels = list(case["reference_change_labels"])
        messages = render_lc_messages(
            case["target_finding"]["display_name"], labels, case["task"],
        )
        record = _input_record(
            setting="REF-LC",
            result_identifier=result_identifier,
            messages=messages,
            model=model,
            benchmark_version=case["benchmark_version"],
        )
        if lookup.get(("REF-LC", result_identifier)) != record:
            raise ValueError(
                f"Prepared REF-LC model input conflicts: {result_identifier}"
            )
        specs.append(
            final_response_spec(
                case,
                "REF-LC",
                messages,
                record["model_input_id"],
                model,
                consumed_change_labels=labels,
                dependency_ids=[],
            )
        )
    return specs


def _propagated_lc_record(
    case: dict[str, Any],
    setting: str,
    fc_record: dict[str, Any],
    model: str,
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Create one complete skipped-LC result from its fixed-code outcome."""
    status = status or fc_record["status"]
    if status not in {
        "upstream_invalid",
        "upstream_response_error",
        "no_identifiable_most_severe_visit_or_plateau",
        UPSTREAM_ANSWER_MISMATCH,
    }:
        raise ValueError(f"Cannot propagate fixed-code status to LC: {status}")
    return {
        "result_id": case["result_ids"][setting],
        "final_case_id": case["final_case_id"],
        "setting": setting,
        "model_input_id": None,
        "benchmark_version": case["benchmark_version"],
        "trajectory_id": case["trajectory_id"],
        "dataset": case["dataset"],
        "annotation_source": case["annotation_source"],
        "patient_id": case["patient_id"],
        "timepoint_count": case["timepoint_count"],
        "target_finding": case["target_finding"],
        "task": case["task"],
        "reference_answer": case["reference_answer"],
        "model": model,
        "prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
        "dependency_ids": list(fc_record["dependency_ids"]),
        "dependency_statuses": list(fc_record["dependency_statuses"]),
        "consumed_change_labels": fc_record["consumed_change_labels"],
        "status": status,
        "raw_text": None,
        "parsed_output": None,
        "parsing_mode": None,
        "failure_reason": None,
        "response_error": None,
        "predicted_answer": None,
        "correct": False,
    }


def _canonical_predicted_labels(
    case: dict[str, Any],
    fc_record: dict[str, Any],
    source: str,
    predictions: dict[str, dict[str, Any]],
) -> list[str]:
    """Verify FC dependency IDs and labels against canonical predictions."""
    expected_ids = (
        list(case["pair_dependency_ids"])
        if source == "PAIR"
        else [case["seq_dependency_id"]]
    )
    if fc_record.get("dependency_ids") != expected_ids:
        raise ValueError(f"{source}-FC dependency IDs conflict with final case")
    if source == "PAIR":
        labels: list[str] = []
        for dependency_id in expected_ids:
            prediction = predictions.get(dependency_id)
            if prediction is None or prediction.get("status") != "valid":
                raise ValueError("PAIR-LC requires valid canonical predictions")
            labels.append(prediction["predicted_change_label"])
    else:
        prediction = predictions.get(expected_ids[0])
        if prediction is None or prediction.get("status") != "valid":
            raise ValueError("SEQ-LC requires valid canonical predictions")
        labels = list(prediction["predicted_change_labels"])
    if labels != fc_record.get("consumed_change_labels"):
        raise ValueError(f"{source}-FC consumed labels conflict with predictions")
    return labels


def build_predicted_lc_specs(
    run_dir: Path,
    final_cases: list[dict[str, Any]],
    model: str,
    source: str,
    *,
    lc_inference_policy: str | None = LC_INFERENCE_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build LC requests from usable labels under the specified run policy."""
    if source not in {"PAIR", "SEQ"}:
        raise ValueError(f"Unsupported LC label source: {source}")
    setting = f"{source}-LC"
    fc_setting = f"{source}-FC"
    fc_filename = "pair_fc.json" if source == "PAIR" else "seq_fc.json"
    prediction_filename = "pair.jsonl" if source == "PAIR" else "seq.jsonl"
    fc_records = _by_id(run_dir / "derived" / fc_filename, "result_id")
    predictions = _by_id(
        run_dir / "responses" / prediction_filename, "prediction_id"
    )
    specs: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for case in final_cases:
        fc_identifier = case["result_ids"][fc_setting]
        fc_record = fc_records.get(fc_identifier)
        if fc_record is None:
            raise ValueError(f"Missing expected {fc_setting} result: {fc_identifier}")
        if fc_record.get("final_case_id") != case["final_case_id"]:
            raise ValueError(f"{fc_setting} final-case identity conflict")
        if fc_record["status"] in {"valid", NO_IDENTIFIABLE}:
            labels = _canonical_predicted_labels(case, fc_record, source, predictions)
            answer = derive_answer(case["task"], labels)
            expected_answer = None if answer == NO_IDENTIFIABLE else answer
            expected_status = NO_IDENTIFIABLE if answer == NO_IDENTIFIABLE else "valid"
            if (
                fc_record.get("predicted_answer") != expected_answer
                or fc_record["status"] != expected_status
            ):
                raise ValueError(f"{fc_setting} answer conflicts with predicted labels")
        skip_status = lc_skip_status(
            fc_record["status"], fc_record.get("predicted_answer"),
            case["reference_answer"], lc_inference_policy,
        )
        if skip_status is not None:
            propagated = _propagated_lc_record(
                case, setting, fc_record, model, status=skip_status
            )
            specs.append(
                {
                    **propagated,
                    "identity_fields": tuple(propagated),
                    "precomputed_record": propagated,
                }
            )
            continue
        messages = render_lc_messages(
            case["target_finding"]["display_name"], labels, case["task"],
        )
        input_record = _input_record(
            setting=setting,
            result_identifier=case["result_ids"][setting],
            messages=messages,
            model=model,
            benchmark_version=case["benchmark_version"],
        )
        input_records.append(input_record)
        specs.append(
            final_response_spec(
                case,
                setting,
                messages,
                input_record["model_input_id"],
                model,
                consumed_change_labels=labels,
                dependency_ids=list(fc_record["dependency_ids"]),
            )
        )
    return specs, input_records


def _client(experiment: Mapping[str, Any], client: Any | None) -> Any:
    """Create and validate the configured model client once."""
    selected = client or create_client(
        str(experiment["base_url"]), require_api_key(),
        timeout=experiment.get("request_timeout_seconds", 300),
    )
    require_exact_model(selected, str(experiment["model"]))
    return selected


def run_ref_lc(
    run_dir: Path,
    experiment: Mapping[str, Any],
    *,
    workers: int = 1,
    progress: bool = False,
    client: Any | None = None,
) -> dict[str, int]:
    """Run or resume the independent reference-label LC setting."""
    require_matching_request_options(run_dir, dict(experiment))
    cases = read_json(run_dir / "prepared" / "final_cases.json")
    specs = build_ref_lc_specs(
        run_dir, cases, str(experiment["model"]),
    )
    selected = _client(experiment, client)
    return execute_specs(
        specs,
        run_dir / "responses" / "ref_lc.jsonl",
        "result_id",
        client=selected,
        model=str(experiment["model"]),
        reasoning_effort=experiment.get("reasoning_effort"),
        max_completion_tokens=experiment.get("max_completion_tokens"),
        image_root=None,
        workers=workers,
        progress=progress,
    )


def run_predicted_lc(
    run_dir: Path,
    experiment: Mapping[str, Any],
    source: str,
    *,
    workers: int = 1,
    progress: bool = False,
    client: Any | None = None,
) -> dict[str, int]:
    """Run or propagate one predicted-label LC setting."""
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("lc_inference_policy") != LC_INFERENCE_POLICY:
        raise ValueError(
            "Run lacks the current LC inference policy. Prepare a new run name "
            "or explicitly prepare the saved run before resuming under a new policy."
        )
    require_matching_request_options(run_dir, dict(experiment))
    cases = read_json(run_dir / "prepared" / "final_cases.json")
    specs, input_records = build_predicted_lc_specs(
        run_dir, cases, str(experiment["model"]), source,
    )
    _append_model_inputs(run_dir, input_records)
    selected = (
        _client(experiment, client)
        if any("precomputed_record" not in spec for spec in specs)
        else client
    )
    filename = "pair_lc.jsonl" if source == "PAIR" else "seq_lc.jsonl"
    return execute_specs(
        specs,
        run_dir / "responses" / filename,
        "result_id",
        client=selected,
        model=str(experiment["model"]),
        reasoning_effort=experiment.get("reasoning_effort"),
        max_completion_tokens=experiment.get("max_completion_tokens"),
        image_root=None,
        workers=workers,
        progress=progress,
    )


def parse_args() -> argparse.Namespace:
    """Parse options for the separately resumable LC module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all LC settings after fixed-code dependencies are available."""
    args = parse_args()
    experiment = load_experiment(args.experiment)
    run_dir = run_directory(experiment)
    selected = _client(experiment, None)
    print(
        {
            "ref_lc": run_ref_lc(
                run_dir,
                experiment,
                workers=args.workers,
                progress=args.progress,
                client=selected,
            ),
            "pair_lc": run_predicted_lc(
                run_dir,
                experiment,
                "PAIR",
                workers=args.workers,
                progress=args.progress,
                client=selected,
            ),
            "seq_lc": run_predicted_lc(
                run_dir,
                experiment,
                "SEQ",
                workers=args.workers,
                progress=args.progress,
                client=selected,
            ),
        }
    )


if __name__ == "__main__":
    main()

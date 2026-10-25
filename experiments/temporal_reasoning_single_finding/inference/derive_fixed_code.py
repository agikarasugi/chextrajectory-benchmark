"""Derive PAIR-FC and SEQ-FC final results from canonical predictions."""

import argparse
from pathlib import Path
from typing import Any

from ..artifact_io import read_json, read_jsonl, write_json
from ..protocol.contracts import NO_IDENTIFIABLE
from ..protocol.task_rules import derive_answer


def _records_by_id(
    path: Path,
    identifier_field: str,
) -> dict[str, dict[str, Any]]:
    """Load response rows while rejecting duplicate identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"Required response file not found: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        identifier = row.get(identifier_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Response lacks {identifier_field}: {path}")
        if identifier in result:
            raise ValueError(f"Duplicate {identifier_field} in {path}: {identifier}")
        result[identifier] = row
    return result


def _base_record(case: dict[str, Any], setting: str) -> dict[str, Any]:
    """Create the common fixed-code final-result fields."""
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
        "raw_text": None,
        "parsed_output": None,
        "parsing_mode": None,
        "failure_reason": None,
        "response_error": None,
    }


def _upstream_status(
    dependency_ids: list[str],
    predictions: dict[str, dict[str, Any]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Return the propagated status and every failed dependency."""
    failures: list[dict[str, str]] = []
    for dependency_id in dependency_ids:
        if dependency_id not in predictions:
            raise ValueError(f"Missing expected prediction: {dependency_id}")
        status = predictions[dependency_id].get("status")
        if status not in {"valid", "invalid_response", "response_error"}:
            raise ValueError(f"Invalid prediction status for {dependency_id}: {status}")
        if status != "valid":
            failures.append(
                {"dependency_id": dependency_id, "status": str(status)}
            )
    if not failures:
        return None, []
    if any(failure["status"] == "response_error" for failure in failures):
        return "upstream_response_error", failures
    return "upstream_invalid", failures


def _compose_one(
    case: dict[str, Any],
    setting: str,
    dependency_ids: list[str],
    predictions: dict[str, dict[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    """Compose one fixed-code record with exact failure propagation."""
    record = _base_record(case, setting)
    status, dependency_statuses = _upstream_status(dependency_ids, predictions)
    record["dependency_ids"] = dependency_ids
    record["dependency_statuses"] = dependency_statuses
    record["consumed_change_labels"] = None
    record["predicted_answer"] = None
    record["correct"] = False
    if status is not None:
        record["status"] = status
        return record

    labels: list[str] = []
    for dependency_id in dependency_ids:
        value = predictions[dependency_id].get(value_field)
        if value_field == "predicted_change_labels":
            if not isinstance(value, list):
                raise ValueError(
                    f"Valid SEQ prediction lacks label vector: {dependency_id}"
                )
            labels.extend(value)
        else:
            if not isinstance(value, str):
                raise ValueError(
                    f"Valid PAIR prediction lacks change label: {dependency_id}"
                )
            labels.append(value)
    if len(labels) != case["timepoint_count"] - 1:
        raise ValueError("Prediction label count conflicts with final case")
    record["consumed_change_labels"] = labels
    answer = derive_answer(case["task"], labels)
    if answer == NO_IDENTIFIABLE:
        record["status"] = NO_IDENTIFIABLE
        return record
    record["status"] = "valid"
    record["predicted_answer"] = answer
    record["correct"] = answer == case["reference_answer"]
    return record


def build_fixed_code_records(
    final_cases: list[dict[str, Any]],
    pair_predictions: dict[str, dict[str, Any]],
    seq_predictions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build both fixed-code setting result collections."""
    pair_records: list[dict[str, Any]] = []
    seq_records: list[dict[str, Any]] = []
    for case in final_cases:
        pair_records.append(
            _compose_one(
                case,
                "PAIR-FC",
                list(case["pair_dependency_ids"]),
                pair_predictions,
                "predicted_change_label",
            )
        )
        seq_records.append(
            _compose_one(
                case,
                "SEQ-FC",
                [case["seq_dependency_id"]],
                seq_predictions,
                "predicted_change_labels",
            )
        )
    return pair_records, seq_records


def derive_fixed_code(run_dir: Path) -> dict[str, int]:
    """Derive or exactly validate the two atomic fixed-code artifacts."""
    final_cases = read_json(run_dir / "prepared" / "final_cases.json")
    if not isinstance(final_cases, list):
        raise ValueError("prepared/final_cases.json must contain an array")
    pair_predictions = _records_by_id(
        run_dir / "responses" / "pair.jsonl", "prediction_id"
    )
    seq_predictions = _records_by_id(
        run_dir / "responses" / "seq.jsonl", "prediction_id"
    )
    pair_records, seq_records = build_fixed_code_records(
        final_cases,
        pair_predictions,
        seq_predictions,
    )
    for filename, records in (
        ("pair_fc.json", pair_records),
        ("seq_fc.json", seq_records),
    ):
        path = run_dir / "derived" / filename
        if path.exists():
            if read_json(path) != records:
                raise ValueError(f"Existing derived artifact conflicts: {filename}")
        else:
            write_json(path, records)
    return {"pair_fc": len(pair_records), "seq_fc": len(seq_records)}


def parse_args() -> argparse.Namespace:
    """Parse the separately resumable fixed-code stage arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run fixed-code composition as a module entry point."""
    args = parse_args()
    print(derive_fixed_code(args.run_dir))


if __name__ == "__main__":
    main()

"""Run the complete-sequence direct final-answer setting."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifact_io import read_json, read_jsonl
from ..protocol.contracts import FINAL_PROMPT_VERSION, JSON_SCHEMA_VERSION
from ..protocol.identities import model_input_id
from ..protocol.prompts import render_seq_dir_messages
from .experiment_config import (
    load_experiment,
    load_private_paths,
    require_api_key,
    require_matching_request_options,
    run_directory,
)
from .model_client import create_client, require_exact_model
from .run_change_labels import execute_specs


def final_response_spec(
    case: dict[str, Any],
    setting: str,
    safe_messages: list[dict[str, Any]],
    expected_model_input_id: str,
    model: str,
    *,
    consumed_change_labels: list[str] | None,
    dependency_ids: list[str],
) -> dict[str, Any]:
    """Create a model-backed final-answer response specification."""
    spec = {
        "result_id": case["result_ids"][setting],
        "final_case_id": case["final_case_id"],
        "setting": setting,
        "model_input_id": expected_model_input_id,
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
        "dependency_ids": dependency_ids,
        "dependency_statuses": [],
        "consumed_change_labels": consumed_change_labels,
        "safe_model_input": safe_messages,
        "response_type": "final",
        "timepoint_count_for_parser": case["timepoint_count"],
        "task_for_parser": case["task"],
    }
    spec["identity_fields"] = (
        "result_id",
        "final_case_id",
        "setting",
        "model_input_id",
        "benchmark_version",
        "trajectory_id",
        "dataset",
        "annotation_source",
        "patient_id",
        "timepoint_count",
        "target_finding",
        "task",
        "reference_answer",
        "model",
        "prompt_version",
        "json_schema_version",
        "dependency_ids",
        "consumed_change_labels",
    )

    def finalize(record: dict[str, Any]) -> None:
        """Retain the final answer and exact per-case correctness."""
        output = record["parsed_output"]
        predicted = output["answer"] if record["status"] == "valid" else None
        record["predicted_answer"] = predicted
        record["correct"] = predicted == record["reference_answer"]

    spec["finalize"] = finalize
    return spec


def _model_input_lookup(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the safe input registry while rejecting duplicate lookup keys."""
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "prepared" / "model_inputs.jsonl"):
        key = (row["record_type"], row["record_id"])
        if key in result:
            raise ValueError(f"Duplicate model-input registry entry: {key}")
        result[key] = row
    return result


def build_seq_dir_specs(
    run_dir: Path,
    final_cases: list[dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Recompute every SEQ-DIR input and verify its prepared fingerprint."""
    lookup = _model_input_lookup(run_dir)
    specs: list[dict[str, Any]] = []
    for case in final_cases:
        result_identifier = case["result_ids"]["SEQ-DIR"]
        messages = render_seq_dir_messages(
            case["target_finding"]["display_name"],
            [image["image_id"] for image in case["images"]],
            case["task"],
        )
        expected = model_input_id(
            safe_model_input=messages,
            model=model,
            user_prompt_version=FINAL_PROMPT_VERSION,
            json_schema_version=JSON_SCHEMA_VERSION,
            benchmark_version=case["benchmark_version"],
        )
        saved = lookup.get(("SEQ-DIR", result_identifier))
        if saved is None or saved.get("model_input_id") != expected:
            raise ValueError(
                f"Prepared SEQ-DIR model input conflicts: {result_identifier}"
            )
        specs.append(
            final_response_spec(
                case,
                "SEQ-DIR",
                messages,
                expected,
                model,
                consumed_change_labels=None,
                dependency_ids=[],
            )
        )
    return specs


def run_seq_dir(
    run_dir: Path,
    experiment: Mapping[str, Any],
    image_root: Path,
    *,
    workers: int = 1,
    progress: bool = False,
    client: Any | None = None,
) -> dict[str, int]:
    """Run or resume every SEQ-DIR final-answer call."""
    require_matching_request_options(run_dir, dict(experiment))
    final_cases = read_json(run_dir / "prepared" / "final_cases.json")
    if not isinstance(final_cases, list):
        raise ValueError("prepared/final_cases.json must contain an array")
    selected_client = client or create_client(
        str(experiment["base_url"]), require_api_key(),
        timeout=experiment.get("request_timeout_seconds", 300),
    )
    require_exact_model(selected_client, str(experiment["model"]))
    return execute_specs(
        build_seq_dir_specs(
            run_dir, final_cases, str(experiment["model"]),
        ),
        run_dir / "responses" / "seq_dir.jsonl",
        "result_id",
        client=selected_client,
        model=str(experiment["model"]),
        reasoning_effort=experiment.get("reasoning_effort"),
        max_completion_tokens=experiment.get("max_completion_tokens"),
        image_root=image_root,
        workers=workers,
        progress=progress,
    )


def parse_args() -> argparse.Namespace:
    """Parse separately resumable SEQ-DIR stage options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run SEQ-DIR from the prepared experiment."""
    args = parse_args()
    experiment = load_experiment(args.experiment)
    paths = load_private_paths(args.paths)
    summary = run_seq_dir(
        run_directory(experiment),
        experiment,
        paths["chexpert_plus_image_root"],
        workers=args.workers,
        progress=args.progress,
    )
    print(summary)


if __name__ == "__main__":
    main()

"""Run the independently prepared PAIR and SEQ change-label predictions."""

import argparse
import math
import re
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..artifact_io import assert_no_image_payload, append_jsonl, read_json, read_jsonl
from ..protocol.contracts import FAILURE_REASONS, interval_keys, require_exact_keys
from ..protocol.prompts import SHORT_EXPLANATION_PREFIX
from .api_errors import EndpointRequestError
from .experiment_config import (
    load_experiment,
    load_private_paths,
    require_api_key,
    require_matching_request_options,
    run_directory,
    validate_worker_count,
)
from .json_recovery import parse_response
from .model_client import create_client, require_exact_model, submit_messages


_INTERNAL_SPEC_FIELDS = {
    "safe_model_input",
    "response_type",
    "timepoint_count_for_parser",
    "task_for_parser",
    "identity_fields",
    "finalize",
}
_MODEL_TERMINAL_STATUSES = {"valid", "invalid_response", "response_error"}
_TRANSPORT_FAILURE_REASONS = {"refusal", "content_filter"}


def _load_saved(
    path: Path,
    specs: list[dict[str, Any]],
    identifier_field: str,
) -> dict[str, dict[str, Any]]:
    """Validate every saved row against its approved response identity."""
    expected = {spec[identifier_field]: spec for spec in specs}
    saved: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return saved
    for row in read_jsonl(path):
        identifier = row.get(identifier_field)
        if not isinstance(identifier, str) or identifier not in expected:
            raise ValueError(f"Unexpected {identifier_field} in {path}: {identifier}")
        if identifier in saved:
            raise ValueError(f"Duplicate {identifier_field} in {path}: {identifier}")
        spec = expected[identifier]
        for field in spec["identity_fields"]:
            if row.get(field) != spec.get(field):
                raise ValueError(
                    f"Saved response identity conflict for {identifier}: {field}"
                )
        if "precomputed_record" in spec:
            if row != spec["precomputed_record"]:
                raise ValueError(f"Saved propagated response conflicts: {identifier}")
        else:
            _validate_saved_model_record(row, spec, identifier)
        saved[identifier] = row
    return saved


def _record_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the exact persisted record fields supplied by one call spec."""
    return {
        key: value
        for key, value in spec.items()
        if key not in _INTERNAL_SPEC_FIELDS
    }


def _validate_saved_model_record(
    row: dict[str, Any],
    spec: dict[str, Any],
    identifier: str,
) -> None:
    """Fully validate one existing model-backed row before resume reuse."""
    status = row.get("status")
    if status not in _MODEL_TERMINAL_STATUSES:
        raise ValueError(
            f"Saved model-backed response has invalid status: {identifier}"
        )

    raw_text = row.get("raw_text")
    if status in {"valid", "invalid_response"}:
        failure_reason = row.get("failure_reason")
        if failure_reason in _TRANSPORT_FAILURE_REASONS:
            if failure_reason == "refusal" and (
                not isinstance(raw_text, str) or not raw_text.strip()
            ):
                raise ValueError(f"Saved refusal lacks text: {identifier}")
            if failure_reason == "content_filter" and not (
                raw_text is None or isinstance(raw_text, str)
            ):
                raise ValueError(
                    f"Saved content-filter response has invalid text: {identifier}"
                )
            response = _response_fields(
                {
                    "kind": failure_reason,
                    "raw_text": raw_text,
                    "error": None,
                },
                spec["response_type"],
                timepoint_count=spec.get("timepoint_count_for_parser"),
                task=spec.get("task_for_parser"),
            )
        else:
            if not isinstance(raw_text, str):
                raise ValueError(f"Saved textual response lacks raw text: {identifier}")
            response = _response_fields(
                {"kind": "text", "raw_text": raw_text, "error": None},
                spec["response_type"],
                timepoint_count=spec.get("timepoint_count_for_parser"),
                task=spec.get("task_for_parser"),
            )
    else:
        error = row.get("response_error")
        if not isinstance(error, dict):
            raise ValueError(f"Saved response_error is malformed: {identifier}")
        require_exact_keys(error, {"type", "message"}, "response_error")
        if any(
            not isinstance(error[field], str) or not error[field]
            for field in ("type", "message")
        ):
            raise ValueError(f"Saved response_error is malformed: {identifier}")
        response = {
            "status": "response_error",
            "raw_text": None,
            "parsed_output": None,
            "parsing_mode": None,
            "failure_reason": None,
            "response_error": error,
        }

    if response["status"] != status or row.get("failure_reason") not in (
        None,
        *FAILURE_REASONS,
    ):
        raise ValueError(f"Saved model-backed response is malformed: {identifier}")
    expected = _record_from_spec(spec)
    expected.update(response)
    if "usage" in row:
        usage = row["usage"]
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError(f"Saved token usage is malformed: {identifier}")
            require_exact_keys(
                usage, {"input_tokens", "output_tokens", "total_tokens"}, "usage"
            )
            if any(
                value is not None and (type(value) is not int or value < 0)
                for value in usage.values()
            ):
                raise ValueError(f"Saved token usage is malformed: {identifier}")
        expected["usage"] = usage
    if "request_seconds" in row:
        seconds = row["request_seconds"]
        if seconds is not None and (
            type(seconds) not in (int, float)
            or seconds < 0 or not math.isfinite(seconds)
        ):
            raise ValueError(f"Saved request_seconds is malformed: {identifier}")
        expected["request_seconds"] = seconds
    if "attempt_count" in row or "attempt_seconds" in row:
        count = row.get("attempt_count")
        durations = row.get("attempt_seconds")
        if (
            type(count) is not int or count < 1
            or not isinstance(durations, list) or len(durations) != count
            or any(
                type(value) not in (int, float)
                or value < 0 or not math.isfinite(value)
                for value in durations
            )
        ):
            raise ValueError(f"Saved attempt timing is malformed: {identifier}")
        expected["attempt_count"] = count
        expected["attempt_seconds"] = durations
    spec["finalize"](expected)
    if row != expected:
        raise ValueError(f"Saved model-backed response is malformed: {identifier}")
    assert_no_image_payload(row)


def _response_fields(
    transport: dict[str, Any],
    response_type: str,
    *,
    timepoint_count: int | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    """Convert one terminal transport outcome into parser/status fields."""
    kind = transport["kind"]
    raw_text = transport["raw_text"]
    if kind == "response_error":
        return {
            "status": "response_error",
            "raw_text": None,
            "parsed_output": None,
            "parsing_mode": None,
            "failure_reason": None,
            "response_error": transport["error"],
        }
    if kind in {"refusal", "content_filter"}:
        return {
            "status": "invalid_response",
            "raw_text": raw_text,
            "parsed_output": None,
            "parsing_mode": None,
            "failure_reason": kind,
            "response_error": None,
        }
    if kind != "text" or not isinstance(raw_text, str):
        raise ValueError(f"Unsupported terminal transport outcome: {kind}")
    parsed = parse_response(
        raw_text,
        response_type,
        timepoint_count=timepoint_count,
        task=task,
    )
    return {
        **parsed,
        "raw_text": raw_text,
        "response_error": None,
    }


def _submit_spec(
    spec: dict[str, Any],
    client: Any,
    model: str,
    image_root: Path | None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Submit and finalize one prepared model-backed record."""
    request_options = (
        {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}
    )
    if max_completion_tokens is not None:
        request_options["max_completion_tokens"] = max_completion_tokens
    try:
        transport = submit_messages(
            client,
            model,
            spec["safe_model_input"],
            image_root,
            **request_options,
        )
    except EndpointRequestError as error:
        identifier = spec.get("model_input_id")
        if isinstance(identifier, str) and re.fullmatch(r"[0-9a-f]{64}", identifier):
            error.add_note(f"Benchmark model_input_id={identifier}")
        raise
    response = _response_fields(
        transport,
        spec["response_type"],
        timepoint_count=spec.get("timepoint_count_for_parser"),
        task=spec.get("task_for_parser"),
    )
    record = _record_from_spec(spec)
    record.update(response)
    record["usage"] = transport.get("usage")
    record["request_seconds"] = transport.get("request_seconds")
    for field in ("attempt_count", "attempt_seconds"):
        if field in transport:
            record[field] = transport[field]
    spec["finalize"](record)
    return record


def execute_specs(
    specs: list[dict[str, Any]],
    output_path: Path,
    identifier_field: str,
    *,
    client: Any,
    model: str,
    image_root: Path | None,
    workers: int,
    progress: bool,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> dict[str, int]:
    """Run missing specs concurrently with one append-and-flush writer."""
    validate_worker_count(workers)
    saved = _load_saved(output_path, specs, identifier_field)
    pending = [spec for spec in specs if spec[identifier_field] not in saved]
    propagated = [spec for spec in pending if "precomputed_record" in spec]
    callable_specs = [spec for spec in pending if "precomputed_record" not in spec]
    completed = 0

    def persist(record: dict[str, Any]) -> None:
        """Serialize one completed response from the controlling thread."""
        nonlocal completed
        append_jsonl(output_path, record)
        saved[record[identifier_field]] = record
        completed += 1
        if progress:
            print(
                f"[{completed}/{len(pending)}] {output_path.stem}",
                file=sys.stdout,
                flush=True,
            )

    for spec in propagated:
        persist(spec["precomputed_record"])
    if workers == 1:
        for spec in callable_specs:
            persist(_submit_spec(
                spec, client, model, image_root, reasoning_effort,
                max_completion_tokens=max_completion_tokens,
            ))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            handled = set()
            writer_healthy = True
            first_error: Exception | None = None
            try:
                for spec in callable_specs:
                    future = executor.submit(
                        _submit_spec, spec, client, model, image_root, reasoning_effort,
                        max_completion_tokens=max_completion_tokens,
                    )
                    futures[future] = spec
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except CancelledError:
                        continue
                    except Exception as error:
                        if first_error is None:
                            first_error = error
                            for pending_future in futures:
                                if pending_future is not future:
                                    pending_future.cancel()
                        continue
                    try:
                        persist(record)
                        handled.add(future)
                    except BaseException:
                        # An interrupted append may already have written bytes.
                        # Leave its tail for recovery rather than append again.
                        writer_healthy = False
                        raise
            except BaseException as error:
                # A controller interruption bypasses future.result() handling.
                # Cancel the queue before the context manager waits for workers.
                executor.shutdown(wait=False, cancel_futures=True)
                try:
                    if writer_healthy:
                        # shutdown() removes cancelled tasks from the work queue.
                        # Exclude them because as_completed() may otherwise wait
                        # for a worker notification that can no longer arrive.
                        active = [
                            future for future in futures
                            if future not in handled and not future.cancelled()
                        ]
                        for future in as_completed(active):
                            try:
                                record = future.result()
                            except BaseException:
                                continue
                            persist(record)
                except BaseException:
                    error.add_note("Failed to save remaining completed responses during cleanup")
                raise
            if first_error is not None:
                raise first_error
    return {
        "expected": len(specs),
        "reused": len(specs) - len(pending),
        "submitted": len(callable_specs),
        "propagated": len(propagated),
    }


def _pair_spec(value: dict[str, Any]) -> dict[str, Any]:
    """Convert one prepared PAIR input to an executable response spec."""
    spec = dict(value)
    spec["response_type"] = "pair"
    spec["timepoint_count_for_parser"] = None
    spec["task_for_parser"] = None
    spec["identity_fields"] = (
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

    def finalize(record: dict[str, Any]) -> None:
        """Retain PAIR prediction and exact correctness."""
        output = record["parsed_output"]
        predicted = output["change_label"] if record["status"] == "valid" else None
        record["predicted_change_label"] = predicted
        record["correct"] = predicted == record["reference_change_label"]

    spec["finalize"] = finalize
    return spec


def _seq_spec(value: dict[str, Any]) -> dict[str, Any]:
    """Convert one prepared SEQ input to an executable response spec."""
    spec = dict(value)
    spec["response_type"] = "seq"
    spec["timepoint_count_for_parser"] = value["timepoint_count"]
    spec["task_for_parser"] = None
    spec["identity_fields"] = (
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

    def finalize(record: dict[str, Any]) -> None:
        """Retain chronological SEQ labels and per-label correctness."""
        if record["status"] == "valid":
            values = record["parsed_output"]["change_labels"]
            predicted = [
                values[key] for key in interval_keys(record["timepoint_count"])
            ]
            label_correctness = [
                prediction == reference
                for prediction, reference in zip(
                    predicted,
                    record["reference_change_labels"],
                    strict=True,
                )
            ]
        else:
            predicted = None
            label_correctness = None
        record["predicted_change_labels"] = predicted
        record["label_correctness"] = label_correctness
        record["correct"] = bool(label_correctness and all(label_correctness))

    spec["finalize"] = finalize
    return spec


def run_change_labels(
    run_dir: Path,
    experiment: Mapping[str, Any],
    image_root: Path,
    *,
    workers: int = 1,
    progress: bool = False,
    client: Any | None = None,
) -> dict[str, dict[str, int]]:
    """Run or resume every prepared PAIR and SEQ prediction."""
    require_matching_request_options(run_dir, dict(experiment))
    validate_worker_count(workers)
    pair_inputs = read_json(run_dir / "prepared" / "pair_inputs.json")
    seq_inputs = read_json(run_dir / "prepared" / "seq_inputs.json")
    if not isinstance(pair_inputs, list) or not isinstance(seq_inputs, list):
        raise ValueError("Prepared change-label registries must be arrays")
    for value in pair_inputs + seq_inputs:
        content = value["safe_model_input"][0]["content"]
        instruction = content[-1]["text"]
        if SHORT_EXPLANATION_PREFIX not in instruction:
            raise ValueError("Prepared change-label prompt conflicts with permanent short explanations")
    selected_client = client or create_client(
        str(experiment["base_url"]), require_api_key(),
        timeout=experiment.get("request_timeout_seconds", 300),
    )
    require_exact_model(selected_client, str(experiment["model"]))
    pair_summary = execute_specs(
        [_pair_spec(value) for value in pair_inputs],
        run_dir / "responses" / "pair.jsonl",
        "prediction_id",
        client=selected_client,
        model=str(experiment["model"]),
        reasoning_effort=experiment.get("reasoning_effort"),
        max_completion_tokens=experiment.get("max_completion_tokens"),
        image_root=image_root,
        workers=workers,
        progress=progress,
    )
    seq_summary = execute_specs(
        [_seq_spec(value) for value in seq_inputs],
        run_dir / "responses" / "seq.jsonl",
        "prediction_id",
        client=selected_client,
        model=str(experiment["model"]),
        reasoning_effort=experiment.get("reasoning_effort"),
        max_completion_tokens=experiment.get("max_completion_tokens"),
        image_root=image_root,
        workers=workers,
        progress=progress,
    )
    return {"pair": pair_summary, "seq": seq_summary}


def parse_args() -> argparse.Namespace:
    """Parse separately resumable change-label stage options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run PAIR and SEQ stages from their prepared run directory."""
    args = parse_args()
    experiment = load_experiment(args.experiment)
    paths = load_private_paths(args.paths)
    summary = run_change_labels(
        run_directory(experiment),
        experiment,
        paths["chexpert_plus_image_root"],
        workers=args.workers,
        progress=args.progress,
    )
    print(summary)


if __name__ == "__main__":
    main()

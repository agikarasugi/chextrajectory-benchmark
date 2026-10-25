"""Metadata-only completed runs with explicitly selected prediction outcomes."""

import json
from pathlib import Path
from typing import Any

from ...validation.legacy_prompts import (
    build_answer_only_registries, render_answer_only,
)
from ...protocol.prompt_compatibility import (
    LEGACY_FIXTURE_SHA256,
)
from ...artifact_io import read_json, repository_root, sha256_file, write_json, write_jsonl
from ...inference.derive_fixed_code import build_fixed_code_records
from ...inference.experiment_config import RUNS_RELATIVE_ROOT
from ...inference.prepare_run import _prompt_preview, build_registries
from ...inference.run_lc import _propagated_lc_record
from ...protocol.contracts import (
    DATASET,
    FINAL_PROMPT_VERSION,
    JSON_SCHEMA_VERSION,
    LC_INFERENCE_POLICY,
    MODEL_BACKED_SETTINGS,
    MOST_SEVERE_EXCLUSION,
    NO_IDENTIFIABLE,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
    TASKS,
    interval_keys,
)
from ..validation.test_validate_run import (
    _late_input,
    _valid_final,
    _valid_pair,
    _valid_seq,
)


def item(
    name: str,
    labels: list[str],
    answers: list[str | None],
    *,
    source: str = "silver",
    patient: str | None = None,
    finding: str = "edema",
    image_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build an item from literal references, without deriving expected answers."""
    count = len(labels) + 1
    image_ids = image_ids or [f"{name}/visit-{i}.jpg" for i in range(1, count + 1)]
    return {
        "benchmark_version": "fixture-compatible-v1",
        "trajectory_id": f"{DATASET}:{name}",
        "source_sequence_id": name,
        "dataset": DATASET,
        "annotation_source": source,
        "patient_id": patient or name,
        "timepoint_count": count,
        "target_finding": {
            "canonical_name": finding,
            "display_name": finding.replace("_", " "),
        },
        "images": [
            {
                "timepoint": f"T{i}",
                "image_id": identifier,
                "relative_path": identifier.removesuffix(".jpg") + ".png",
            }
            for i, identifier in enumerate(image_ids, start=1)
        ],
        "reference_change_labels": labels,
        "tasks": {
            task: {
                "eligible": answer is not None,
                "reference_answer": answer,
                "exclusion_reason": (
                    None if answer is not None else MOST_SEVERE_EXCLUSION
                ),
            }
            for task, answer in zip(TASKS, answers, strict=True)
        },
    }


def _fail(row: dict[str, Any], status: str) -> None:
    """Set one realistic terminal failure without introducing a new attempt."""
    row.update({
        "status": status,
        "correct": False,
        "raw_text": '{"unexpected":"field"}' if status == "invalid_response" else None,
        "parsed_output": None,
        "parsing_mode": None,
        "failure_reason": "schema_mismatch" if status == "invalid_response" else None,
        "response_error": (
            {"type": "TimeoutError", "message": "fixture retries exhausted"}
            if status == "response_error" else None
        ),
    })
    for field in (
        "predicted_change_label", "predicted_change_labels", "label_correctness",
        "predicted_answer",
    ):
        if field in row:
            row[field] = None


def write_run(
    root: Path,
    items: list[dict[str, Any]],
    *,
    run_name: str = "fixture-run",
    pair: dict[tuple[str, int], str] | None = None,
    seq: dict[str, list[str] | str] | None = None,
    final: dict[tuple[str, str, str], str] | None = None,
    benchmark_manifest: Path | None = None,
    legacy_answer_only: bool = False,
    lc_inference_policy: str | None = None,
) -> Path:
    """Write complete saved run records, with no image files or API calls.

    Overrides use short source sequence IDs. PAIR interval indices are zero
    based. Final keys are (sequence, task, setting). Unspecified LC answers
    follow fixed code, and unspecified direct answers follow the reference.
    """
    run_dir = root / RUNS_RELATIVE_ROOT / run_name
    benchmark_path = root / f"{run_name}-benchmark.json"
    versions = {
        "benchmark_version": "fixture-compatible-v1",
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "seq_prompt_version": SEQ_PROMPT_VERSION,
        "final_prompt_version": FINAL_PROMPT_VERSION,
        "json_schema_version": JSON_SCHEMA_VERSION,
    }
    registry_manifest = versions
    if benchmark_manifest is None:
        write_json(benchmark_path, {**versions, "dataset": DATASET, "fixture_items": items})
    else:
        benchmark_path = benchmark_manifest
        registry_manifest = read_json(benchmark_path)
        versions = {key: registry_manifest[key] for key in versions}
    experiment = {
        "run_name": run_name,
        "model": "fixture-model",
        "base_url": "http://127.0.0.1:1/v1",
        "benchmark_path": str(benchmark_path),
    }
    builder = build_answer_only_registries if legacy_answer_only else build_registries
    registries = builder(
        experiment, registry_manifest, items,
    )
    fixture_checksums = (
        repository_root() / "experiments/temporal_reasoning_single_finding"
        / "tests/fixtures/golden_prompts/sha256.json"
    )
    write_json(run_dir / "run_manifest.json", {
        **({"lc_inference_policy": lc_inference_policy}
           if lc_inference_policy is not None else {}),
        **versions,
        "experiment": experiment,
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "prompt_fixture_checksums": {
            "golden_prompts/sha256.json": (
                LEGACY_FIXTURE_SHA256 if legacy_answer_only else sha256_file(fixture_checksums)
            )
        },
    })
    pair_rows = {s["prediction_id"]: _valid_pair(s) for s in registries["pair_inputs"]}
    cases_by_trajectory = {c["trajectory_id"]: c for c in registries["final_cases"]}
    for (name, index), outcome in (pair or {}).items():
        case = cases_by_trajectory[f"{DATASET}:{name}"]
        row = pair_rows[case["pair_dependency_ids"][index]]
        if outcome in {"invalid_response", "response_error"}:
            _fail(row, outcome)
        else:
            row["predicted_change_label"] = outcome
            row["parsed_output"] = {"change_label": outcome}
            row["raw_text"] = json.dumps(row["parsed_output"])
            row["correct"] = outcome == row["reference_change_label"]
    seq_rows = {}
    for spec in registries["seq_inputs"]:
        row = _valid_seq(spec)
        name = spec["source_sequence_id"].split(":", 1)[1]
        outcome = (seq or {}).get(name, spec["reference_change_labels"])
        if isinstance(outcome, str):
            _fail(row, outcome)
        else:
            row["predicted_change_labels"] = outcome
            row["label_correctness"] = [
                a == b for a, b in zip(
                    outcome, spec["reference_change_labels"], strict=True
                )
            ]
            row["correct"] = all(row["label_correctness"])
            row["parsed_output"] = {"change_labels": dict(zip(
                interval_keys(spec["timepoint_count"]), outcome, strict=True
            ))}
            row["raw_text"] = json.dumps(row["parsed_output"])
        seq_rows[spec["prediction_id"]] = row
    pair_fc, seq_fc = build_fixed_code_records(
        registries["final_cases"], pair_rows, seq_rows
    )
    fc_rows = {r["result_id"]: r for r in pair_fc + seq_fc}
    model_inputs = list(registries["model_inputs"])
    initial = {(r["record_type"], r["record_id"]): r for r in model_inputs}
    final_rows = {setting: [] for setting in MODEL_BACKED_SETTINGS}
    for case in registries["final_cases"]:
        for setting in MODEL_BACKED_SETTINGS:
            labels, dependencies = None, []
            outcome = case["reference_answer"]
            if setting in {"PAIR-LC", "SEQ-LC"}:
                fc = fc_rows[case["result_ids"][setting.replace("LC", "FC")]]
                mismatch = (
                    lc_inference_policy == "reference-supported-answer-v1"
                    and fc["status"] == "valid"
                    and not fc["correct"]
                )
                callable_statuses = {"valid"}
                if lc_inference_policy == LC_INFERENCE_POLICY:
                    callable_statuses.add(NO_IDENTIFIABLE)
                if fc["status"] not in callable_statuses or mismatch:
                    final_rows[setting].append(_propagated_lc_record(
                        case, setting, fc, experiment["model"],
                        status="upstream_answer_mismatch" if mismatch else None,
                    ))
                    continue
                labels = fc["consumed_change_labels"]
                dependencies = fc["dependency_ids"]
                outcome = fc["predicted_answer"] or case["reference_answer"]
                input_record = _late_input(
                    case, setting, labels, experiment["model"],
                    legacy_answer_only=legacy_answer_only,
                )
                model_inputs.append(input_record)
            else:
                input_record = initial[(setting, case["result_ids"][setting])]
                if setting == "REF-LC":
                    labels = case["reference_change_labels"]
            row = _valid_final(case, setting, input_record, labels, dependencies)
            name = case["trajectory_id"].split(":", 1)[1]
            outcome = (final or {}).get((name, case["task"], setting), outcome)
            if outcome in {"invalid_response", "response_error"}:
                _fail(row, outcome)
            else:
                row["predicted_answer"] = outcome
                row["parsed_output"] = {"answer": outcome}
                row["raw_text"] = json.dumps(row["parsed_output"])
                row["correct"] = outcome == case["reference_answer"]
            final_rows[setting].append(row)

    prepared = run_dir / "prepared"
    for key in ("pair_inputs", "seq_inputs", "final_cases"):
        write_json(prepared / f"{key}.json", registries[key])
    preview = _prompt_preview()
    if legacy_answer_only:
        preview = {
            name: render_answer_only(lambda: messages)
            for name, messages in preview.items()
        }
    write_json(prepared / "prompt_preview.json", preview)
    write_jsonl(prepared / "model_inputs.jsonl", model_inputs)
    write_json(prepared / "manifest.json", {
        "call_plan": registries["call_plan"],
        "counts": {
            **{
                key: len(registries[key])
                for key in ("pair_inputs", "seq_inputs", "final_cases")
            },
            "initial_model_inputs": len(registries["model_inputs"]),
            "model_inputs": len(model_inputs),
        },
        "files": {p.name: sha256_file(p) for p in sorted(prepared.iterdir())},
    })
    write_jsonl(run_dir / "responses/pair.jsonl", pair_rows.values())
    write_jsonl(run_dir / "responses/seq.jsonl", seq_rows.values())
    write_json(run_dir / "derived/pair_fc.json", pair_fc)
    write_json(run_dir / "derived/seq_fc.json", seq_fc)
    for setting, rows in final_rows.items():
        filename = setting.lower().replace("-", "_") + ".jsonl"
        write_jsonl(run_dir / "responses" / filename, rows)
    return run_dir

"""Build deterministic model-specific registries and run preparation artifacts."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..artifact_io import (
    assert_no_image_payload,
    read_json,
    repository_root,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..protocol.prompt_compatibility import normalize_prompt_fixture
from ..protocol.contracts import (
    FINAL_PROMPT_VERSION,
    JSON_SCHEMA_VERSION,
    LC_INFERENCE_POLICY,
    PAIR_PROMPT_VERSION,
    SEQ_PROMPT_VERSION,
    SETTINGS,
    TASKS,
)
from ..protocol.identities import (
    final_case_id,
    model_input_id,
    pair_prediction_id,
    result_id,
    seq_prediction_id,
)
from ..protocol.task_selection import (
    selected_tasks, unreduced_call_ceiling, validate_task_selection,
)
from ..protocol.prompts import (
    render_lc_messages,
    render_pair_messages,
    render_seq_dir_messages,
    render_seq_messages,
)

from .experiment_config import experiment_resume_identity, normalize_experiment


def run_dir_for(experiment: Mapping[str, Any]) -> Path:
    """Resolve the single run directory selected by a safe run name."""
    return (
        repository_root()
        / "artifacts"
        / "temporal_reasoning_single_finding"
        / "runs"
        / str(experiment["run_name"])
    )


def _version(manifest: Mapping[str, Any], key: str) -> str:
    """Read a required frozen version identifier."""
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Benchmark manifest lacks {key}")
    return value


def _model_input_record(
    *,
    record_type: str,
    record_id: str,
    messages: list[dict[str, Any]],
    model: str,
    prompt_version: str,
    schema_version: str,
    benchmark_version: str,
) -> dict[str, Any]:
    """Create one privacy-safe model-input registry record."""
    identifier = model_input_id(
        safe_model_input=messages,
        model=model,
        user_prompt_version=prompt_version,
        json_schema_version=schema_version,
        benchmark_version=benchmark_version,
    )
    record = {
        "record_type": record_type,
        "record_id": record_id,
        "model_input_id": identifier,
        "model": model,
        "user_prompt_version": prompt_version,
        "json_schema_version": schema_version,
        "benchmark_version": benchmark_version,
        "safe_model_input": messages,
    }
    assert_no_image_payload(record)
    return record


def _render_current(renderer: Callable, *args: Any) -> list[dict[str, Any]]:
    """Render the permanent prompt without any output-mode selection."""
    return renderer(*args)


def build_registries(
    experiment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build registries using the permanent short-explanation prompts."""
    return _build_registries(experiment, manifest, items, _render_current)


def _build_registries(
    experiment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    items: list[dict[str, Any]],
    render: Callable,
) -> dict[str, Any]:
    """Build complete deduplicated PAIR, SEQ, case, and initial input registries."""
    model = str(experiment["model"])
    validate_task_selection(manifest, items)
    benchmark_version = _version(manifest, "benchmark_version")
    pair_version = _version(manifest, "pair_prompt_version")
    seq_version = _version(manifest, "seq_prompt_version")
    final_version = _version(manifest, "final_prompt_version")
    schema_version = _version(manifest, "json_schema_version")
    if (
        pair_version != PAIR_PROMPT_VERSION
        or seq_version != SEQ_PROMPT_VERSION
        or final_version != FINAL_PROMPT_VERSION
        or schema_version != JSON_SCHEMA_VERSION
    ):
        raise ValueError(
            "Benchmark manifest uses unsupported prompt or schema versions"
        )

    pair_by_id: dict[str, dict[str, Any]] = {}
    seq_by_id: dict[str, dict[str, Any]] = {}
    case_records: list[dict[str, Any]] = []
    model_inputs: list[dict[str, Any]] = []
    source_metadata: dict[str, tuple[Any, ...]] = {}

    for item in sorted(items, key=lambda value: value["trajectory_id"]):
        if item["benchmark_version"] != benchmark_version:
            raise ValueError("Benchmark item version conflicts with its manifest")
        trajectory_id = item["trajectory_id"]
        metadata = (
            item["dataset"],
            item["annotation_source"],
            item["patient_id"],
            item["target_finding"]["canonical_name"],
            tuple(image["image_id"] for image in item["images"]),
        )
        previous_metadata = source_metadata.setdefault(trajectory_id, metadata)
        if previous_metadata != metadata:
            raise ValueError(
                "Namespaced source sequence has conflicting annotation-source metadata"
            )
        display_finding = item["target_finding"]["display_name"]
        canonical_finding = item["target_finding"]["canonical_name"]
        images = item["images"]
        pair_dependencies: list[str] = []
        for index, reference_label in enumerate(item["reference_change_labels"]):
            earlier = images[index]
            later = images[index + 1]
            prediction_id = pair_prediction_id(
                dataset=item["dataset"],
                annotation_source=item["annotation_source"],
                earlier_image_id=earlier["image_id"],
                later_image_id=later["image_id"],
                target_finding=canonical_finding,
                model=model,
                pair_prompt_version=pair_version,
                json_schema_version=schema_version,
                benchmark_version=benchmark_version,
            )
            pair_dependencies.append(prediction_id)
            messages = render(
                render_pair_messages,
                display_finding,
                earlier["image_id"],
                later["image_id"],
            )
            input_record = _model_input_record(
                record_type="PAIR",
                record_id=prediction_id,
                messages=messages,
                model=model,
                prompt_version=pair_version,
                schema_version=schema_version,
                benchmark_version=benchmark_version,
            )
            candidate = {
                "prediction_id": prediction_id,
                "prediction_type": "PAIR",
                "model_input_id": input_record["model_input_id"],
                "benchmark_version": benchmark_version,
                "dataset": item["dataset"],
                "annotation_source": item["annotation_source"],
                "earlier_image_id": earlier["image_id"],
                "later_image_id": later["image_id"],
                "earlier_relative_path": earlier["relative_path"],
                "later_relative_path": later["relative_path"],
                "target_finding": item["target_finding"],
                "model": model,
                "prompt_version": pair_version,
                "json_schema_version": schema_version,
                "reference_change_label": reference_label,
                "safe_model_input": messages,
            }
            existing = pair_by_id.get(prediction_id)
            if existing is not None and existing != candidate:
                raise ValueError("PAIR model input has conflicting source references")
            if existing is None:
                pair_by_id[prediction_id] = candidate
                model_inputs.append(input_record)

        seq_id = seq_prediction_id(
            source_sequence_id=trajectory_id,
            model=model,
            seq_prompt_version=seq_version,
            json_schema_version=schema_version,
            benchmark_version=benchmark_version,
        )
        seq_messages = render(
            render_seq_messages,
            display_finding,
            [image["image_id"] for image in images],
        )
        seq_input_record = _model_input_record(
            record_type="SEQ",
            record_id=seq_id,
            messages=seq_messages,
            model=model,
            prompt_version=seq_version,
            schema_version=schema_version,
            benchmark_version=benchmark_version,
        )
        seq_candidate = {
            "prediction_id": seq_id,
            "prediction_type": "SEQ",
            "model_input_id": seq_input_record["model_input_id"],
            "benchmark_version": benchmark_version,
            "source_sequence_id": trajectory_id,
            "dataset": item["dataset"],
            "annotation_source": item["annotation_source"],
            "patient_id": item["patient_id"],
            "timepoint_count": item["timepoint_count"],
            "target_finding": item["target_finding"],
            "images": images,
            "model": model,
            "prompt_version": seq_version,
            "json_schema_version": schema_version,
            "reference_change_labels": item["reference_change_labels"],
            "safe_model_input": seq_messages,
        }
        if "assigned_task" in item:
            seq_candidate["assigned_task"] = item["assigned_task"]
        existing_seq = seq_by_id.get(seq_id)
        if existing_seq is not None and existing_seq != seq_candidate:
            raise ValueError(
                "SEQ model input has conflicting metadata or reference labels"
            )
        if existing_seq is None:
            seq_by_id[seq_id] = seq_candidate
            model_inputs.append(seq_input_record)

        for task in selected_tasks(item):
            task_record = item["tasks"][task]
            if not task_record["eligible"]:
                continue
            case_identifier = final_case_id(
                source_sequence_id=trajectory_id,
                task=task,
            )
            result_ids = {
                setting: result_id(case_id=case_identifier, setting=setting)
                for setting in SETTINGS
            }
            case = {
                "final_case_id": case_identifier,
                "benchmark_version": benchmark_version,
                "trajectory_id": trajectory_id,
                "dataset": item["dataset"],
                "annotation_source": item["annotation_source"],
                "patient_id": item["patient_id"],
                "timepoint_count": item["timepoint_count"],
                "target_finding": item["target_finding"],
                "images": images,
                "task": task,
                "reference_answer": task_record["reference_answer"],
                "reference_change_labels": item["reference_change_labels"],
                "pair_dependency_ids": pair_dependencies,
                "seq_dependency_id": seq_id,
                "result_ids": result_ids,
            }
            case_records.append(case)
            seq_dir_messages = render(
                render_seq_dir_messages,
                display_finding,
                [image["image_id"] for image in images],
                task,
            )
            model_inputs.append(
                _model_input_record(
                    record_type="SEQ-DIR",
                    record_id=result_ids["SEQ-DIR"],
                    messages=seq_dir_messages,
                    model=model,
                    prompt_version=final_version,
                    schema_version=schema_version,
                    benchmark_version=benchmark_version,
                )
            )
            ref_lc_messages = render(
                render_lc_messages,
                display_finding,
                item["reference_change_labels"],
                task,
            )
            model_inputs.append(
                _model_input_record(
                    record_type="REF-LC",
                    record_id=result_ids["REF-LC"],
                    messages=ref_lc_messages,
                    model=model,
                    prompt_version=final_version,
                    schema_version=schema_version,
                    benchmark_version=benchmark_version,
                )
            )

    pair_inputs = sorted(pair_by_id.values(), key=lambda value: value["prediction_id"])
    seq_inputs = sorted(seq_by_id.values(), key=lambda value: value["prediction_id"])
    final_cases = sorted(case_records, key=lambda value: value["final_case_id"])
    model_inputs.sort(key=lambda value: (value["record_type"], value["record_id"]))
    unreduced_maximum = unreduced_call_ceiling(items)
    call_plan = {
        "pair_predictions": len(pair_inputs),
        "seq_predictions": len(seq_inputs),
        "seq_dir": len(final_cases),
        "ref_lc": len(final_cases),
        "pair_lc_potential": len(final_cases),
        "seq_lc_potential": len(final_cases),
        "potential_logical_calls_before_retries": (
            len(pair_inputs) + len(seq_inputs) + 4 * len(final_cases)
        ),
        "unreduced_maximum_calls_before_retries": unreduced_maximum,
    }
    call_plan["maximum_transport_attempts"] = (
        call_plan["potential_logical_calls_before_retries"] * 3
    )
    return {
        "pair_inputs": pair_inputs,
        "seq_inputs": seq_inputs,
        "final_cases": final_cases,
        "model_inputs": model_inputs,
        "call_plan": call_plan,
    }


def _prompt_preview() -> dict[str, Any]:
    """Build all 19 privacy-safe prompt variants for inspection."""
    preview: dict[str, Any] = {
        "PAIR": render_pair_messages(
            "pleural effusion",
            "EARLIER_IMAGE_ID",
            "LATER_IMAGE_ID",
        )
    }
    for count in (3, 4):
        images = [f"T{index}_IMAGE_ID" for index in range(1, count + 1)]
        preview[f"SEQ-{count}TP"] = render_seq_messages(
            "pleural effusion", images,
        )
        for task in TASKS:
            preview[f"SEQ-DIR-{count}TP-{task}"] = render_seq_dir_messages(
                "pleural effusion", images, task,
            )
            preview[f"LC-{count}TP-{task}"] = render_lc_messages(
                "pleural effusion",
                ["deterioration"] * (count - 1),
                task,
            )
    assert_no_image_payload(preview)
    if len(preview) != 19:
        raise AssertionError("Prompt preview must contain all 19 approved variants")
    return preview


def prepare_run(
    experiment: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create or exactly validate one resumable run preparation."""
    run_dir = run_dir_for(experiment)
    registries = build_registries(
        experiment, manifest, items,
    )
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "golden_prompts"
        / "sha256.json"
    )
    run_manifest = {
        "lc_inference_policy": LC_INFERENCE_POLICY,
        "experiment": normalize_experiment(dict(experiment)),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "benchmark_version": manifest["benchmark_version"],
        "pair_prompt_version": manifest["pair_prompt_version"],
        "seq_prompt_version": manifest["seq_prompt_version"],
        "final_prompt_version": manifest["final_prompt_version"],
        "json_schema_version": manifest["json_schema_version"],
        "prompt_fixture_checksums": {
            "golden_prompts/sha256.json": sha256_file(fixture_path)
        },
    }
    existing_manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        if not existing_manifest_path.is_file():
            raise ValueError("Existing run directory lacks run_manifest.json")
        saved_manifest = read_json(existing_manifest_path)
        # Older runs recorded a second, machine-specific path. It is not
        # benchmark identity and must not prevent relocation or resume.
        if isinstance(saved_manifest, dict):
            saved_manifest.pop("benchmark_manifest_path", None)
            saved_manifest["experiment"] = experiment_resume_identity(
                saved_manifest["experiment"]
            )
        current_manifest = dict(
            run_manifest, experiment=experiment_resume_identity(dict(experiment))
        )
        if normalize_prompt_fixture(saved_manifest) != current_manifest:
            raise ValueError("Existing run directory does not match this experiment")
    else:
        run_dir.mkdir(parents=True)
        write_json(existing_manifest_path, run_manifest)

    prepared = run_dir / "prepared"
    prepared.mkdir(exist_ok=True)
    files = {
        "pair_inputs.json": registries["pair_inputs"],
        "seq_inputs.json": registries["seq_inputs"],
        "final_cases.json": registries["final_cases"],
        "prompt_preview.json": _prompt_preview(),
    }
    for filename, value in files.items():
        path = prepared / filename
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Existing prepared artifact conflicts: {filename}")
        if not path.exists():
            write_json(path, value)
    model_inputs_path = prepared / "model_inputs.jsonl"
    if model_inputs_path.exists():
        from ..artifact_io import read_jsonl

        saved_inputs = read_jsonl(model_inputs_path)
        saved_by_key = {
            (row.get("record_type"), row.get("record_id")): row
            for row in saved_inputs
        }
        if len(saved_by_key) != len(saved_inputs):
            raise ValueError("Existing model-input registry contains duplicate entries")
        for expected in registries["model_inputs"]:
            key = (expected["record_type"], expected["record_id"])
            if saved_by_key.get(key) != expected:
                raise ValueError("Existing initial model-input registry conflicts")
    else:
        write_jsonl(model_inputs_path, registries["model_inputs"])
    prepared_manifest = {
        "call_plan": registries["call_plan"],
        "counts": {
            "pair_inputs": len(registries["pair_inputs"]),
            "seq_inputs": len(registries["seq_inputs"]),
            "final_cases": len(registries["final_cases"]),
            "initial_model_inputs": len(registries["model_inputs"]),
        },
        "files": {
            filename: sha256_file(prepared / filename)
            for filename in (
                "pair_inputs.json",
                "seq_inputs.json",
                "final_cases.json",
                "model_inputs.jsonl",
                "prompt_preview.json",
            )
        },
    }
    prepared_manifest_path = prepared / "manifest.json"
    if prepared_manifest_path.exists():
        existing_prepared = read_json(prepared_manifest_path)
        for key in ("call_plan",):
            if existing_prepared.get(key) != prepared_manifest[key]:
                raise ValueError("Existing prepared manifest conflicts")
        for filename in (
            "pair_inputs.json",
            "seq_inputs.json",
            "final_cases.json",
            "prompt_preview.json",
        ):
            if (
                existing_prepared.get("files", {}).get(filename)
                != prepared_manifest["files"][filename]
            ):
                raise ValueError("Existing prepared manifest conflicts")
    else:
        write_json(prepared_manifest_path, prepared_manifest)
    for directory in ("responses", "derived", "validation"):
        (run_dir / directory).mkdir(exist_ok=True)
    return {"run_dir": run_dir, **registries}

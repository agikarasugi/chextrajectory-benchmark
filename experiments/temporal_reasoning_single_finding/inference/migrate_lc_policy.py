"""Copy a legacy run into the historical gated policy without model calls."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..artifact_io import read_json, sha256_file, write_json, write_jsonl
from ..protocol.contracts import REFERENCE_SUPPORTED_LC_POLICY
from ..protocol.prompt_compatibility import normalize_prompt_fixture
from ..validation.preflight import (
    _check_run_target,
    _expected_run_manifest,
    load_and_validate_benchmark,
)
from ..validation.validate_run import (
    _load_prepared,
    _validate_run_manifest,
    load_validated_run,
)
from .derive_fixed_code import build_fixed_code_records
from .experiment_config import normalize_experiment, resolve_benchmark_path
from .prepare_run import _prompt_preview, build_registries
from .run_change_labels import _load_saved, _pair_spec, _seq_spec
from .run_lc import build_predicted_lc_specs, build_ref_lc_specs
from .run_seq_dir import build_seq_dir_specs


_PREPARED_FILES = (
    "pair_inputs.json", "seq_inputs.json", "final_cases.json",
    "prompt_preview.json", "model_inputs.jsonl", "manifest.json",
)
_RESPONSE_NAMES = ("pair", "seq", "seq_dir", "ref_lc", "pair_lc", "seq_lc")
_SOURCE_FILES = (
    "run_manifest.json",
    *(f"prepared/{name}" for name in _PREPARED_FILES),
    *(f"responses/{name}.jsonl" for name in _RESPONSE_NAMES),
    "derived/pair_fc.json", "derived/seq_fc.json",
)


def _snapshot(run_dir: Path) -> dict[str, str | None]:
    """Fingerprint only the allowlisted run JSON, including absent responses."""
    result: dict[str, str | None] = {}
    for relative in _SOURCE_FILES:
        path = run_dir / relative
        if path.is_symlink():
            raise ValueError(f"Migration source must not be a symlink: {relative}")
        result[relative] = sha256_file(path) if path.is_file() else None
    return result


def _input_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict]:
    """Index exact model-input records without silently accepting duplicates."""
    lookup = {(row["record_type"], row["record_id"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate model-input registry entry")
    return lookup


def _migrated_rows(
    specs: list[dict[str, Any]], saved: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Retain supported outcomes and materialize skips for all unsupported cases."""
    rows = []
    counts = {
        "saved_before": len(saved), "retained_model_responses": 0,
        "retained_skips": 0, "replaced_model_responses": 0,
        "new_skips": 0, "pending_model_calls": 0,
    }
    for spec in specs:
        previous = saved.get(spec["result_id"])
        if "precomputed_record" in spec:
            row = spec["precomputed_record"]
            rows.append(row)
            if previous is None:
                counts["new_skips"] += 1
            elif previous == row:
                counts["retained_skips"] += 1
            else:
                counts["replaced_model_responses"] += 1
        elif previous is not None:
            rows.append(previous)
            counts["retained_model_responses"] += 1
        else:
            counts["pending_model_calls"] += 1
    counts["saved_after"] = len(rows)
    return rows, counts


def migrate_lc_policy(source_run_dir: Path, new_run_name: str) -> dict[str, Any]:
    """Validate a stopped legacy run and atomically publish a separate gated copy.

    PAIR and SEQ predictions and fixed-code results must be complete. Final
    model stages may be partial. Only known JSON artifacts are copied. The
    source, image files, endpoint, and existing destination runs are untouched.
    """
    source = Path(source_run_dir).resolve()
    saved_manifest = _validate_run_manifest(source / "run_manifest.json")
    if "lc_inference_policy" in saved_manifest:
        raise ValueError("Migration requires a legacy run without an LC policy")
    experiment = normalize_experiment(saved_manifest["experiment"])
    migrated_experiment = normalize_experiment(dict(experiment, run_name=new_run_name))
    destination = source.parent / new_run_name
    if (
        new_run_name == experiment["run_name"]
        or destination.exists() or destination.is_symlink()
    ):
        raise ValueError("Use a new run name with no existing destination")

    before = _snapshot(source)
    prepared = _load_prepared(source)
    benchmark = load_and_validate_benchmark(resolve_benchmark_path(experiment))
    expected_manifest = _expected_run_manifest(experiment, benchmark)
    expected_manifest.pop("lc_inference_policy")
    normalized = normalize_prompt_fixture(dict(saved_manifest, experiment=experiment))
    normalized.pop("benchmark_manifest_path", None)
    if normalized != expected_manifest:
        raise ValueError("Legacy run manifest conflicts with the current benchmark")
    registries = build_registries(experiment, benchmark.manifest, benchmark.items)
    for key in ("pair_inputs", "seq_inputs", "final_cases", "call_plan"):
        if prepared[key] != registries[key]:
            raise ValueError(f"Legacy prepared {key} conflicts with current prompts")
    if read_json(source / "prepared/prompt_preview.json") != _prompt_preview():
        raise ValueError("Legacy run must already use short-explanation prompts")

    saved: dict[str, dict[str, dict[str, Any]]] = {}
    specs_by_stage: dict[str, list[dict[str, Any]]] = {}
    for name, builder in (("pair", _pair_spec), ("seq", _seq_spec)):
        specs = [builder(row) for row in prepared[f"{name}_inputs"]]
        specs_by_stage[name] = specs
        saved[name] = _load_saved(
            source / f"responses/{name}.jsonl", specs, "prediction_id"
        )
        if len(saved[name]) != len(specs):
            raise ValueError(f"Migration requires complete {name.upper()} predictions")
    cases = prepared["final_cases"]
    pair_fc, seq_fc = build_fixed_code_records(cases, saved["pair"], saved["seq"])
    for name, expected in (("pair", pair_fc), ("seq", seq_fc)):
        if read_json(source / f"derived/{name}_fc.json") != expected:
            raise ValueError(f"Saved {name.upper()}-FC conflicts with predictions")

    model = experiment["model"]
    for name, builder in (
        ("seq_dir", build_seq_dir_specs), ("ref_lc", build_ref_lc_specs),
    ):
        specs = builder(source, cases, model)
        specs_by_stage[name] = specs
        saved[name] = _load_saved(
            source / f"responses/{name}.jsonl", specs, "result_id"
        )

    legacy_inputs = list(registries["model_inputs"])
    current_inputs = list(registries["model_inputs"])
    lc_rows = {}
    stages = {}
    for label_source in ("PAIR", "SEQ"):
        name = f"{label_source.lower()}_lc"
        legacy_specs, inputs = build_predicted_lc_specs(
            source, cases, model, label_source, lc_inference_policy=None,
        )
        legacy_inputs.extend(inputs)
        saved[name] = _load_saved(
            source / f"responses/{name}.jsonl", legacy_specs, "result_id"
        )
        specs, inputs = build_predicted_lc_specs(
            source, cases, model, label_source,
            lc_inference_policy=REFERENCE_SUPPORTED_LC_POLICY,
        )
        specs_by_stage[name] = specs
        current_inputs.extend(inputs)
        lc_rows[name], stages[name] = _migrated_rows(specs, saved[name])

    expected_inputs = _input_lookup(legacy_inputs)
    actual_inputs = _input_lookup(prepared["model_inputs"])
    for key, row in actual_inputs.items():
        if expected_inputs.get(key) != row:
            raise ValueError(f"Legacy model-input registry conflicts: {key}")
    for row in registries["model_inputs"]:
        if actual_inputs.get((row["record_type"], row["record_id"])) != row:
            raise ValueError("Legacy model-input registry lacks an initial input")
    for rows in saved.values():
        for row in rows.values():
            if row.get("model_input_id") is not None:
                setting = row.get("setting", row.get("prediction_type"))
                identifier = row.get("result_id", row.get("prediction_id"))
                registered = actual_inputs.get((setting, identifier))
                if (
                    registered is None
                    or registered["model_input_id"] != row["model_input_id"]
                ):
                    raise ValueError("Saved response lacks its exact registered input")

    current_inputs.sort(key=lambda row: (row["record_type"], row["record_id"]))
    report = {
        "schema_version": 1,
        "source_run_dir": str(source),
        "source_run_name": experiment["run_name"],
        "destination_run_name": new_run_name,
        "lc_inference_policy": REFERENCE_SUPPORTED_LC_POLICY,
        "source_files": before,
        "stages": stages,
        "model_calls_submitted_by_migration": 0,
        "source_preserved": True,
    }
    pending = sum(stage["pending_model_calls"] for stage in stages.values())
    for name in ("pair", "seq", "seq_dir", "ref_lc"):
        missing = len(specs_by_stage[name]) - len(saved[name])
        stages[name] = {
            "retained_responses": len(saved[name]), "pending_model_calls": missing,
        }
        pending += missing
    report["pending_model_calls"] = pending
    report["status"] = "ready_to_resume" if pending else "complete"

    with tempfile.TemporaryDirectory(
        prefix=".lc-policy-migration-", dir=source.parent,
    ) as temp:
        staging = Path(temp) / new_run_name
        staging.mkdir()
        for relative in _SOURCE_FILES:
            if relative in {
                "run_manifest.json", "prepared/manifest.json",
                "prepared/model_inputs.jsonl", "responses/pair_lc.jsonl",
                "responses/seq_lc.jsonl",
            }:
                continue
            if before[relative] is not None:
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
        write_json(
            staging / "run_manifest.json",
            _expected_run_manifest(
                migrated_experiment, benchmark,
                lc_inference_policy=REFERENCE_SUPPORTED_LC_POLICY,
            ),
        )
        write_json(staging / "experiment.json", migrated_experiment)
        write_jsonl(staging / "prepared/model_inputs.jsonl", current_inputs)
        for name, rows in lc_rows.items():
            write_jsonl(staging / f"responses/{name}.jsonl", rows)
        prepared_manifest = read_json(source / "prepared/manifest.json")
        prepared_manifest["counts"]["model_inputs"] = len(current_inputs)
        prepared_manifest["files"]["model_inputs.jsonl"] = sha256_file(
            staging / "prepared/model_inputs.jsonl"
        )
        write_json(staging / "prepared/manifest.json", prepared_manifest)
        _check_run_target(
            staging, migrated_experiment, benchmark,
            lc_inference_policy=REFERENCE_SUPPORTED_LC_POLICY,
        )
        _load_prepared(staging)
        for name, specs in specs_by_stage.items():
            identifier = "prediction_id" if name in {"pair", "seq"} else "result_id"
            _load_saved(staging / f"responses/{name}.jsonl", specs, identifier)
        if not pending:
            load_validated_run(staging)
        if _snapshot(source) != before:
            raise ValueError(
                "Source run changed during migration. Stop writers and retry."
            )
        write_json(staging / "migration.json", report)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Migration destination appeared during validation")
        staging.rename(destination)
    return {
        **report, "run_dir": str(destination),
        "experiment_path": str(destination / "experiment.json"),
    }


def main() -> None:
    """Copy one stopped legacy run and print metadata-only migration counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--new-run-name", required=True)
    args = parser.parse_args()
    result = migrate_lc_policy(args.run_dir, args.new_run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

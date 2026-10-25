"""Shared run identity, metadata checks, and interrupted-response recovery."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..artifact_io import read_json, safe_relative_path, sha256_file, write_json
from ..inference.experiment_config import EXECUTION_OPTION_FIELDS, REQUEST_OPTION_FIELDS
from .config import VLLM_LOG_ENV


ATTEMPT_METADATA_FIELDS = {"source_sha256", "profile", "versions", "workers"}
EXECUTION_VLLM_OPTIONS = {
    "gpu-memory-utilization", "max-model-len", "max-num-seqs",
    "max-num-batched-tokens", "tensor-parallel-size", "pipeline-parallel-size",
}


def _without_execution_options(args: list[str]) -> list[str]:
    """Remove only agreed runtime options, preserving all other command tokens."""
    remaining = []
    index = 0
    while index < len(args):
        token = args[index]
        option = token.split("=", 1)[0].removeprefix("--").replace("_", "-")
        if token.startswith("--") and option in EXECUTION_VLLM_OPTIONS:
            if "=" not in token:
                if index + 1 >= len(args) or args[index + 1].startswith("--"):
                    raise ValueError(f"Missing value for vLLM option: {token}")
                index += 1
        else:
            remaining.append(token)
        index += 1
    return remaining


def _request_identity(identity: dict) -> dict:
    """Normalize request options and exclude compatible execution metadata."""
    if identity.get("enable_short_explanation", True) is not True:
        raise ValueError(
            "Modal provenance changed: saved prompt mode is not short explanations; "
            "select a new run_name for inference\n"
            "  enable_short_explanation: "
            f"value={json.dumps(identity['enable_short_explanation'])}, required=true"
        )
    normalized = {
        key: value for key, value in identity.items()
        if key not in ATTEMPT_METADATA_FIELDS and key != "enable_short_explanation"
    }
    experiment = normalized.get("experiment")
    if isinstance(experiment, dict):
        normalized["experiment"] = {
            key: value for key, value in experiment.items()
            if key not in EXECUTION_OPTION_FIELDS
            and (key not in REQUEST_OPTION_FIELDS or value is not None)
        }
    serving = normalized.get("serving")
    if isinstance(serving, dict):
        serving = dict(serving)
        if isinstance(serving.get("vllm_args"), list):
            serving["vllm_args"] = _without_execution_options(serving["vllm_args"])
        if isinstance(serving.get("vllm_env"), dict):
            remaining = {
                key: value for key, value in serving["vllm_env"].items()
                if key not in VLLM_LOG_ENV
            }
            if remaining:
                serving["vllm_env"] = remaining
            else:
                serving.pop("vllm_env")
        normalized["serving"] = serving
    if isinstance(normalized.get("server_command"), list):
        normalized["server_command"] = _without_execution_options(
            normalized["server_command"]
        )
    return normalized


def _provenance_changes(previous: dict, current: dict, prefix: str = "") -> list[str]:
    """Describe changed fields deterministically, distinguishing missing from null."""
    changes = []
    for key in sorted(previous.keys() | current.keys()):
        field = f"{prefix}.{key}" if prefix else key
        if key in previous and key in current:
            old, new = previous[key], current[key]
            if old == new:
                continue
            if isinstance(old, dict) and isinstance(new, dict):
                changes.extend(_provenance_changes(old, new, field))
                continue
        saved = (
            json.dumps(previous[key], sort_keys=True) if key in previous else "<missing>"
        )
        requested = (
            json.dumps(current[key], sort_keys=True) if key in current else "<missing>"
        )
        changes.append(f"  {field}: saved={saved}, current={requested}")
    return changes


def require_provenance(path: Path, identity: dict) -> None:
    """Preserve run identity while allowing different code and infrastructure."""
    if path.exists():
        saved = read_json(path)
        # Profile and installed versions may change when switching GPU images.
        # Logging controls may also change; unknown future fields remain strict.
        previous = _request_identity(saved)
        current = _request_identity(identity)
        if previous != current:
            changes = "\n".join(_provenance_changes(previous, current))
            raise ValueError(
                "Modal provenance changed; select a new run_name\n"
                f"Saved provenance: {path}\n{changes}"
            )
    else:
        write_json(path, identity)


def _validate_import(run_dir: Path, experiment: dict) -> dict:
    """Check an imported run's benchmark and prepared text without model calls."""
    from ..inference.experiment_config import resolve_benchmark_path
    from ..inference.prepare_run import _prompt_preview, build_registries
    from ..validation.preflight import _check_run_target, load_and_validate_benchmark
    from ..validation.validate_run import _load_prepared, _validate_run_manifest

    _validate_run_manifest(run_dir / "run_manifest.json")
    benchmark = load_and_validate_benchmark(resolve_benchmark_path(experiment))
    _check_run_target(run_dir, experiment, benchmark)
    prepared = _load_prepared(run_dir)
    expected = build_registries(experiment, benchmark.manifest, benchmark.items)
    for key in ("pair_inputs", "seq_inputs", "final_cases", "call_plan"):
        if prepared[key] != expected[key]:
            raise ValueError(f"Imported prepared artifact conflicts: {key}")
    if read_json(run_dir / "prepared/prompt_preview.json") != _prompt_preview():
        raise ValueError("Imported prompt preview conflicts with current prompts")
    inputs = {
        (row["record_type"], row["record_id"]): row
        for row in prepared["model_inputs"]
    }
    if len(inputs) != len(prepared["model_inputs"]) or any(
        inputs.get((row["record_type"], row["record_id"])) != row
        for row in expected["model_inputs"]
    ):
        raise ValueError("Imported initial model inputs conflict with current prompts")
    return {
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "prepared_manifest_sha256": sha256_file(run_dir / "prepared/manifest.json"),
    }


def require_run_provenance(run_dir: Path, metadata: Path, identity: dict) -> None:
    """Adopt validated external runs and retain known migration provenance."""
    path = metadata / "provenance.json"
    if path.exists() or not run_dir.exists():
        require_provenance(path, identity)
        return
    imported = _validate_import(run_dir, identity["experiment"])
    imported["origin"] = "external"
    migration_path = run_dir / "migration.json"
    if migration_path.exists():
        from ..inference.experiment_config import normalize_experiment

        migration = read_json(migration_path)
        source_name = migration["source_run_name"]
        normalize_experiment(dict(identity["experiment"], run_name=source_name))
        if (
            migration.get("destination_run_name") != identity["experiment"]["run_name"]
            or source_name == identity["experiment"]["run_name"]
        ):
            raise ValueError("Migration run names conflict with imported run")
        imported.update(
            source_run_name=source_name,
            migration_sha256=sha256_file(migration_path),
        )
        source_path = metadata.parent / source_name / "provenance.json"
        if source_path.exists():
            comparison = dict(
                identity, experiment=dict(identity["experiment"], run_name=source_name)
            )
            require_provenance(source_path, comparison)
            imported.update(
                origin="modal_migration",
                source_provenance_path=str(source_path.relative_to(metadata.parent)),
                source_provenance_sha256=sha256_file(source_path),
            )
    # Publish provenance last so an interrupted import repeats validation.
    write_json(metadata / "import.json", imported)
    require_provenance(path, identity)


def record_attempt(metadata: Path, identity: dict) -> Path:
    """Record each accepted attempt without replacing prior execution history."""
    attempt = metadata / "attempts" / uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(
        attempt / "provenance.json",
        {"started_at": datetime.now(UTC).isoformat(), "provenance": identity},
    )
    write_json(attempt / "status.json", {"status": "running"})
    return attempt


def check_file_metadata(root: Path, files: dict) -> None:
    """Check inventory paths and sizes without reading payload contents."""
    if not isinstance(files, dict) or not files:
        raise ValueError("Staged file inventory is empty or invalid")
    for relative, expected in files.items():
        path = safe_relative_path(root.resolve(), relative)
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError("Staged file is missing or differs from its recorded size")


def recover_response_tails(run_dir: Path, metadata: Path) -> None:
    """Back up and repair only an interrupted final JSONL append on resume."""
    import shutil

    for path in sorted((run_dir / "responses").glob("*.jsonl")):
        data = path.read_bytes()
        if not data or data.endswith(b"\n"):
            continue
        boundary = data.rfind(b"\n") + 1
        # Fail on corruption in completed lines; only the unterminated tail is
        # eligible for recovery after a killed append writer.
        for line in data[:boundary].splitlines():
            json.loads(line)
        backup = metadata / "recovery" / (path.name + "." + sha256_file(path))
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copyfile(path, backup)
        try:
            json.loads(data[boundary:])
        except (ValueError, UnicodeError):
            with path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

"""Strict model-run and private-path configuration loading."""

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..artifact_io import read_json, repository_root
from ..protocol.contracts import require_exact_keys


EXPERIMENT_FIELDS = {"run_name", "model", "base_url", "benchmark_path"}
REQUEST_OPTION_FIELDS = {"reasoning_effort", "max_completion_tokens", "request_timeout_seconds"}
EXECUTION_OPTION_FIELDS = {"request_timeout_seconds"}
PRIVATE_PATH_FIELDS = {"chexpert_plus_image_root"}
RUN_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
RUNS_RELATIVE_ROOT = Path("artifacts/temporal_reasoning_single_finding/runs")


def _require_unpadded_string(value: Any, field: str) -> str:
    """Return one non-empty string with no surrounding whitespace."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without padding")
    return value


def _validate_base_url(value: str) -> None:
    """Require an HTTP(S) API root without query or fragment components."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")


def validate_experiment(config: dict[str, Any]) -> None:
    """Validate the exact frozen experiment JSON contract."""
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    required = {
        key: value for key, value in config.items() if key not in REQUEST_OPTION_FIELDS
    }
    require_exact_keys(required, EXPERIMENT_FIELDS, "experiment")
    if config.get("reasoning_effort") is not None:
        _require_unpadded_string(config["reasoning_effort"], "reasoning_effort")

    limit = config.get("max_completion_tokens")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("max_completion_tokens must be a positive integer")

    timeout = config.get("request_timeout_seconds")
    if timeout is not None and (type(timeout) is not int or timeout <= 0):
        raise ValueError("request_timeout_seconds must be a positive integer")

    run_name = _require_unpadded_string(config["run_name"], "run_name")
    if RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError(
            "run_name must match lowercase [a-z0-9][a-z0-9._-]*"
        )
    _require_unpadded_string(config["model"], "model")
    base_url = _require_unpadded_string(config["base_url"], "base_url")
    _validate_base_url(base_url)
    _require_unpadded_string(config["benchmark_path"], "benchmark_path")


def load_experiment(path: Path) -> dict[str, Any]:
    """Load and validate one experiment JSON file."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    return normalize_experiment(value)


def normalize_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize null request options to absent fields."""
    validate_experiment(config)
    normalized = dict(config)
    for field in REQUEST_OPTION_FIELDS:
        if normalized.get(field) is None:
            normalized.pop(field, None)
    return normalized


def experiment_resume_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Compare validated experiment settings without mutable execution options."""
    return {
        key: value for key, value in normalize_experiment(config).items()
        if key not in EXECUTION_OPTION_FIELDS
    }


def require_matching_request_options(run_dir: Path, config: dict[str, Any]) -> None:
    """Prevent standalone stages from mixing request settings in one run."""
    current = normalize_experiment(dict(config))
    manifest = read_json(run_dir / "run_manifest.json")
    saved = normalize_experiment(manifest["experiment"])
    for field in sorted(REQUEST_OPTION_FIELDS - EXECUTION_OPTION_FIELDS):
        if saved.get(field) != current.get(field):
            raise ValueError(
                f"{field} differs from the saved run; use a new run_name"
            )


def resolve_benchmark_path(
    config: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    """Resolve the configured manifest from the repository root when relative."""
    validate_experiment(config)
    configured = Path(config["benchmark_path"])
    if configured.is_absolute():
        return configured.resolve()
    project_root = (root if root is not None else repository_root()).resolve()
    return (project_root / configured).resolve()


def run_directory(
    config: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    """Return the frozen run directory selected by the safe run name."""
    validate_experiment(config)
    project_root = (root if root is not None else repository_root()).resolve()
    return project_root / RUNS_RELATIVE_ROOT / config["run_name"]


def load_private_paths(path: Path) -> dict[str, Path]:
    """Load the exact private image-root file without enumerating images."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Private paths configuration must be a JSON object")
    require_exact_keys(value, PRIVATE_PATH_FIELDS, "private paths")
    raw_root = _require_unpadded_string(
        value["chexpert_plus_image_root"],
        "chexpert_plus_image_root",
    )
    image_root = Path(raw_root)
    if not image_root.is_absolute():
        raise ValueError("chexpert_plus_image_root must be absolute")
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(
            f"CheXpert Plus image root is not a directory: {image_root}"
        )
    return {"chexpert_plus_image_root": image_root}


def require_api_key() -> str:
    """Read the API key from the environment without storing it in artifacts."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required")
    return api_key


def validate_worker_count(workers: int) -> None:
    """Require a positive integer worker count."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("--workers must be a positive integer")

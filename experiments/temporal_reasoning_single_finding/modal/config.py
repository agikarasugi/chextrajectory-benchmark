"""Validate portable job configuration without importing the Modal SDK."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..artifact_io import read_json, repository_root
from ..inference.experiment_config import normalize_experiment

BASE_URL = "http://127.0.0.1:8000/v1"
BENCHMARK_ROOT = "artifacts/temporal_reasoning_single_finding/benchmarks"
RUN_ROOT = "artifacts/temporal_reasoning_single_finding/runs"
PACKAGE = "experiments.temporal_reasoning_single_finding"
VLLM_LOG_ENV = {"VLLM_LOGGING_LEVEL", "VLLM_LOG_STATS_INTERVAL"}
# Config files and aliases could bypass connection, payload logging, or identity
# controls. Unknown model-specific long options otherwise pass through unchanged.
RESERVED = {
    "model",
    "served-model-name",
    "host",
    "port",
    "config",
    "api-key",
    "hf-token",
    "uds",
    "root-path",
    "ssl-keyfile",
    "ssl-certfile",
    "enable-log-requests",
    "no-enable-log-requests",
    "disable-log-requests",
    "enable-log-outputs",
    "no-enable-log-outputs",
    "log-config-file",
    "headless",
    "data-parallel-size",
    "data-parallel-size-local",
    "data-parallel-rank",
    "data-parallel-start-rank",
    "grpc",
}


def text_value(value: Any, name: str) -> str:
    """Require a nonempty printable string without padding."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def positive_int(value: Any, name: str, maximum: int = 86400) -> int:
    """Validate bounded positive integral resource settings."""
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def under_path(value: Any, parent: str) -> str:
    """Require a normalized POSIX path strictly below a specified parent."""
    value = text_value(value, "path")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value or path == PurePosixPath(parent):
        raise ValueError("Path must be normalized and identify a child")
    if not path.is_relative_to(parent):
        raise ValueError(f"Path must be below {parent}")
    return value


def validate_serving(value: Any) -> dict[str, Any]:
    """Preserve the user's vLLM tokens while protecting launcher-owned options."""
    fields = {
        "model_path",
        "served_model_name",
        "vllm_args",
    }
    if (
        not isinstance(value, dict)
        or not fields <= set(value)
        or set(value) - fields - {"vllm_env"}
    ):
        raise ValueError(
            "Serving JSON requires model_path, served_model_name, vllm_args"
        )
    logging_env = value.get("vllm_env", {})
    if not isinstance(logging_env, dict) or set(logging_env) - VLLM_LOG_ENV:
        raise ValueError("vllm_env accepts only vLLM logging level and stats interval")
    for name, setting in logging_env.items():
        text_value(setting, name)
        if name == "VLLM_LOGGING_LEVEL":
            if setting not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                raise ValueError("Invalid VLLM_LOGGING_LEVEL")
        else:
            interval = float(setting)
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("VLLM_LOG_STATS_INTERVAL must be finite and positive")
    under_path(value["model_path"], "/models")
    text_value(value["served_model_name"], "served_model_name")
    args = value["vllm_args"]
    if not isinstance(args, list):
        raise ValueError("vllm_args must be a list of strings")
    options = set()
    for index, arg in enumerate(args):
        text_value(arg, "vllm_args entry")
        if arg == "--" or (arg.startswith("-") and not arg.startswith("--")):
            if not re.fullmatch(r"-\d+(\.\d+)?", arg):
                raise ValueError(
                    "Use full long vLLM options, not aliases or separators"
                )
        if arg.startswith("--"):
            option = arg[2:].split("=", 1)[0].replace("_", "-")
            if option in RESERVED or any(name.startswith(option) for name in RESERVED):
                raise ValueError("vLLM argument conflicts with launcher-owned options")
            if option in options:
                raise ValueError("Duplicate vLLM option")
            options.add(option)
        elif index == 0 or args[index - 1].startswith("--") is False:
            raise ValueError("Each option value must be a single argument")
        elif "=" in args[index - 1]:
            raise ValueError("Unexpected positional vLLM argument")
    return json.loads(json.dumps(value))


def validate_profile(value: Any) -> dict[str, Any]:
    """Validate the separately selected environment and Modal resources."""
    fields = {
        "image",
        "gpu",
        "timeout",
        "startup_timeout",
        "commit_interval",
        "volume_prefix",
    }
    if (
        not isinstance(value, dict)
        or not fields <= set(value)
        or set(value) - fields - {"region", "pip_packages", "memory_mib"}
    ):
        raise ValueError("Invalid Modal profile fields")
    image = text_value(value["image"], "image")
    if not ("@sha256:" in image or ":" in image.rsplit("/", 1)[-1]) or image.endswith(
        ":latest"
    ):
        raise ValueError("Select a version-tagged or digest-pinned container image")
    if not re.fullmatch(r"[A-Za-z0-9-]+(?::[1-8])?", text_value(value["gpu"], "gpu")):
        raise ValueError("gpu must name a GPU with optional :count")
    for name in ("timeout", "startup_timeout", "commit_interval"):
        positive_int(value[name], name)
    if "memory_mib" in value:
        positive_int(value["memory_mib"], "memory_mib")
    if value["startup_timeout"] >= value["timeout"]:
        raise ValueError("startup_timeout must be shorter than job timeout")
    if not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,39}", text_value(value["volume_prefix"], "volume_prefix")
    ):
        raise ValueError("volume_prefix must be a short lowercase name")
    if "region" in value:
        text_value(value["region"], "region")
    packages = value.get("pip_packages", [])
    if not isinstance(packages, list) or any(
        not isinstance(p, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9_.+-]+", p
        )
        for p in packages
    ):
        raise ValueError("pip_packages must contain exact package==version pins")
    return json.loads(json.dumps(value))


def validate_job(
    serving: Any, experiment: Any, profile: Any
) -> tuple[dict, dict, dict]:
    """Check cross-configuration identity before any cloud submission."""
    serving = validate_serving(serving)
    experiment = normalize_experiment(experiment)
    profile = validate_profile(profile)
    if experiment["model"] != serving["served_model_name"]:
        raise ValueError("Experiment model must match served_model_name")
    if experiment["base_url"] != BASE_URL:
        raise ValueError(f"Modal experiment base_url must be {BASE_URL}")
    under_path(experiment["benchmark_path"], BENCHMARK_ROOT)
    args = serving["vllm_args"]
    parallel = {"tensor-parallel-size": 1, "pipeline-parallel-size": 1}
    for index, arg in enumerate(args):
        option = arg.split("=", 1)[0].removeprefix("--").replace("_", "-")
        if option in parallel:
            try:
                parallel[option] = int(
                    arg.split("=", 1)[1] if "=" in arg else args[index + 1]
                )
            except (IndexError, ValueError):
                raise ValueError("Parallel size options need integers") from None
    count = int(profile["gpu"].split(":", 1)[1]) if ":" in profile["gpu"] else 1
    required = parallel["tensor-parallel-size"] * parallel["pipeline-parallel-size"]
    if any(value < 1 for value in parallel.values()) or required > count:
        raise ValueError("Model parallelism exceeds allocated GPUs or is invalid")
    return serving, experiment, profile


def server_command(serving: dict) -> list[str]:
    """Build an argv list without shell expansion or model-dependent branches."""
    serving = validate_serving(serving)
    return [
        "vllm",
        "serve",
        serving["model_path"],
        "--served-model-name",
        serving["served_model_name"],
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        *serving["vllm_args"],
    ]


def source_files(root: Path | None = None) -> list[Path]:
    """Select only the active runtime, adapter, and required prompt fixture."""
    root = root or repository_root()
    package = root / "experiments/temporal_reasoning_single_finding"
    paths = [
        root / "experiments/__init__.py",
        package / "__init__.py",
        package / "artifact_io.py",
    ]
    for directory in ("protocol", "inference", "validation", "evaluation", "modal"):
        paths.extend(sorted((package / directory).glob("*.py")))
    paths.append(package / "tests/fixtures/golden_prompts/sha256.json")
    return paths


def package_source(image: Any, root: Path | None = None) -> Any:
    """Copy the runtime allowlist in one layer, pruning unrelated directories."""
    root = (root or repository_root()).resolve()
    selected = source_files(root)
    allowed = set()
    for path in selected:
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Packaged source must be a regular, non-symlink file: {path}")
        relative = path.relative_to(root)
        allowed.add(relative)
        allowed.update(relative.parents)

    from modal import FilePatternMatcher

    class SourceMatcher(FilePatternMatcher):
        """Allow exact source paths and prune every unrelated directory."""

        def __call__(self, path: Path) -> bool:
            """Exclude paths absent from the source and ancestor allowlist."""
            return path not in allowed

        def can_prune_directories(self) -> bool:
            """Allow pruning because every selected file's ancestors are included."""
            return True

    return image.add_local_dir(str(root), "/app", copy=True, ignore=SourceMatcher())


def source_digest(root: Path | None = None) -> str:
    """Fingerprint precisely the source and fixture files packaged remotely."""
    root = root or repository_root()
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()



PROFILE_ENV = "CHEXTEMPORAL_MODAL_RESOLVED_PROFILE"


def profile_argument() -> Path:
    """Read the required CLI profile before Modal constructs resource definitions."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--profile", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.profile


def runtime_profile(is_local: bool) -> dict:
    """Read explicit local configuration or validated values baked into the image."""
    if is_local:
        return validate_profile(read_json(profile_argument()))
    value = os.environ.get(PROFILE_ENV)
    if not value:
        raise ValueError("Container is missing the resolved Modal resource profile")
    return validate_profile(json.loads(value))

def load_job(serving: str, experiment: str, profile: dict) -> tuple[dict, dict, dict]:
    """Read both local JSON files with the existing strict JSON loader."""
    return validate_job(read_json(Path(serving)), read_json(Path(experiment)), profile)


def validate_download_arguments(revision: str | None, model_path: str) -> None:
    """Reject malformed download inputs before constructing a Modal App."""
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full commit SHA")
    under_path(model_path, "/models")

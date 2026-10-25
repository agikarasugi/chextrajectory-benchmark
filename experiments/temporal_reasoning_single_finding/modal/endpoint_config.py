"""Validate endpoint jobs locally without importing Modal or contacting an API."""

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..artifact_io import read_json
from ..inference.experiment_config import (
    EXPERIMENT_FIELDS, normalize_experiment, validate_worker_count,
)
from .config import BENCHMARK_ROOT, profile_argument, positive_int, text_value, under_path

PROFILE_PATH_ENV = "CHEXTEMPORAL_MODAL_ENDPOINT_PROFILE"
PROFILE_ENV = "CHEXTEMPORAL_MODAL_RESOLVED_ENDPOINT_PROFILE"


def identifier(value: Any, name: str) -> str:
    """Reject control characters and unfilled example placeholders."""
    value = text_value(value, name)
    if any(char in value for char in "<>{}") or "$" in value:
        raise ValueError(f"{name} must not contain template placeholders")
    return value


def validate_profile(value: Any) -> dict:
    """Accept only non-secret CPU resources and optional deployment attribution."""
    required = {
        "volume_prefix", "secret_name", "cpu", "memory", "timeout",
        "startup_timeout",
    }
    if (
        not isinstance(value, dict) or not required <= value.keys()
        or value.keys() - required - {"region", "deployment_identity"}
    ):
        raise ValueError("Invalid endpoint profile fields")
    for name in ("volume_prefix", "secret_name"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", identifier(value[name], name)):
            raise ValueError(f"{name} must be a short lowercase name")
    for name, maximum in (("cpu", 64), ("memory", 262144),
                          ("timeout", 86400), ("startup_timeout", 86400)):
        positive_int(value[name], name, maximum)
    if value["startup_timeout"] >= value["timeout"]:
        raise ValueError("startup_timeout must be shorter than job timeout")
    if "region" in value:
        if not re.fullmatch(r"[a-z0-9-]+", identifier(value["region"], "region")):
            raise ValueError("region must name a Modal compute region")
    if "deployment_identity" in value:
        identity = value["deployment_identity"]
        if (
            not isinstance(identity, dict) or not identity
            or identity.keys() - {"deployment_id", "checkpoint_revision"}
        ):
            raise ValueError("Invalid deployment_identity fields")
        for name, item in identity.items():
            identifier(item, name)
    return json.loads(json.dumps(value))


def validate_job(experiment: Any, profile: Any, workers: int = 1) -> tuple[dict, dict]:
    """Validate endpoint identity, paths, and resources before App startup."""
    experiment = normalize_experiment(experiment)
    for name in EXPERIMENT_FIELDS | {"reasoning_effort"}:
        if name in experiment:
            identifier(experiment[name], name)
    url = experiment["base_url"]
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or not parsed.hostname
        or parsed.username is not None or parsed.password is not None
        or "?" in url or "#" in url or "\\" in url
        or any(char.isspace() for char in url)
        or not parsed.path.rstrip("/").endswith("/v1")
    ):
        raise ValueError("Endpoint base_url must be an HTTPS API root ending in /v1")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Endpoint base_url has an invalid port") from None
    if port is not None and port == 0:
        raise ValueError("Endpoint base_url has an invalid port")
    under_path(experiment["benchmark_path"], BENCHMARK_ROOT)
    validate_worker_count(workers)
    return experiment, validate_profile(profile)


def runtime_profile(is_local: bool) -> dict:
    """Resolve identical resources from local JSON or the container environment."""
    if is_local:
        return validate_profile(read_json(profile_argument()))
    name = PROFILE_ENV
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} to the endpoint resource profile")
    return validate_profile(json.loads(value))


def load_job(experiment: str, workers: int = 1) -> tuple[dict, dict]:
    """Load the local experiment and selected resource profile without cloud calls."""
    return validate_job(read_json(Path(experiment)), runtime_profile(True), workers)

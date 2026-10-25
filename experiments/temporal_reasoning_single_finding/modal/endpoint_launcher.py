"""Run inference and scoring directly against an authenticated Modal Endpoint."""

import importlib.metadata
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from ..artifact_io import (
    assert_no_image_payload, read_json, repository_root, sha256_file, write_json,
)
from ..evaluation.evaluate import evaluate_run
from ..inference.api_errors import EndpointRequestError
from ..inference.experiment_config import require_api_key
from ..inference.model_client import create_client, discover_model_ids
from ..inference.run_pipeline import run_pipeline
from .config import RUN_ROOT, source_digest
from .endpoint_config import validate_job
from .runtime import (
    check_file_metadata, record_attempt, recover_response_tails, require_run_provenance,
)


def wait_ready(base_url: str, model: str, timeout: int) -> str:
    """Poll authenticated metadata with bounded requests and body-free diagnostics."""
    key = require_api_key()
    deadline = time.monotonic() + timeout
    with create_client(base_url, key) as client:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                models = discover_model_ids(
                    client.with_options(timeout=min(10.0, remaining), max_retries=0), model,
                )
            except EndpointRequestError as error:
                status = error.status_code
                if not error.is_connection_error and (
                    status is None or (status not in {408, 429} and status < 500)
                ):
                    raise
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Endpoint returned invalid model metadata") from None
            else:
                if time.monotonic() >= deadline:
                    break
                if model not in models:
                    raise ValueError("Endpoint does not advertise the exact configured model")
                return model
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("Endpoint readiness deadline exceeded")


def package_versions() -> dict[str, str]:
    """Record CPU runtime versions without examining credentials or model files."""
    return {
        "python": sys.version.split()[0],
        **{name: importlib.metadata.version(name) for name in ("openai", "modal")},
    }


def interrupt_job(signum: int, frame: object) -> None:
    """Allow graceful termination to record partial progress and restore the handler."""
    raise InterruptedError("Modal endpoint job received termination")


def save_status(metadata: Path, attempt: Path, status: str) -> None:
    """Write small lifecycle records without serializing exceptions or credentials."""
    for directory in (attempt, metadata):
        write_json(directory / "status.json", {"status": status})


def execute_job(
    experiment: dict,
    profile: dict,
    code_sha256: str,
    commit: Callable[[], None],
    *,
    workers: int = 1,
    retry_failed: bool = False,
) -> dict:
    """Validate staged inputs, run the existing pipeline, and commit each boundary."""
    experiment, profile = validate_job(experiment, profile, workers)
    require_api_key()
    root = repository_root()
    if source_digest(root) != code_sha256:
        raise ValueError("Packaged source fingerprint differs from submitted source")
    benchmark_path = root / experiment["benchmark_path"]
    benchmark_sha256 = sha256_file(benchmark_path)
    staged = read_json(benchmark_path.parent / "modal_images.json")
    if staged["benchmark_sha256"] != benchmark_sha256:
        raise ValueError("Staged image inventory belongs to another benchmark")
    check_file_metadata(Path("/data/images"), staged["images"])
    runs = root / RUN_ROOT
    run_name = experiment["run_name"]
    metadata = runs / ".modal" / run_name
    identity = {
        "execution_mode": "endpoint",
        "experiment": experiment,
        "benchmark_sha256": benchmark_sha256,
        "workers": workers,
        "profile": profile,
        "source_sha256": code_sha256,
        "versions": package_versions(),
        "user_supplied_deployment_identity": profile.get("deployment_identity", {}),
    }
    require_run_provenance(runs / run_name, metadata, identity)
    attempt = record_attempt(metadata, identity)
    previous_handler = signal.signal(signal.SIGTERM, interrupt_job)
    try:
        save_status(metadata, attempt, "running")
        recover_response_tails(runs / run_name, metadata)
        commit()
        print(f"Recorded endpoint execution attempt: {attempt.name}", flush=True)
        discovered = wait_ready(
            experiment["base_url"], experiment["model"], profile["startup_timeout"]
        )
        write_json(attempt / "endpoint.json", {
            "base_url": experiment["base_url"],
            "configured_model": experiment["model"],
            "discovered_model": discovered,
            "weights_verified": False,
        })
        commit()
        with tempfile.TemporaryDirectory(prefix="modal-endpoint-config-") as directory:
            config_dir = Path(directory)
            experiment_path = config_dir / "experiment.json"
            paths_path = config_dir / "paths.json"
            write_json(experiment_path, experiment)
            write_json(paths_path, {"chexpert_plus_image_root": "/data/images"})
            print("Running endpoint benchmark pipeline", flush=True)
            summary = run_pipeline(
                experiment_path, paths_path, workers=workers, progress=True,
                retry_failed=retry_failed,
            )
            assert_no_image_payload(summary)
            write_json(attempt / "inference.json", summary)
            commit()
            print("Scoring saved responses", flush=True)
            evaluate_run(run_name)
        save_status(metadata, attempt, "complete")
        commit()
    except BaseException as error:
        try:
            save_status(metadata, attempt, "failed")
            commit()
        except BaseException:
            error.add_note("Failed to persist failure status and partial results")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return {
        "status": "complete", "run_name": run_name,
        "run_dir": str(runs / run_name),
    }

"""Supervise a loopback vLLM process and the unchanged benchmark commands."""

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ..artifact_io import (
    read_json,
    repository_root,
    sha256_file,
    write_json,
)
from .config import (
    BASE_URL,
    PACKAGE,
    RUN_ROOT,
    server_command,
    source_digest,
    validate_job,
)
from .runtime import (
    check_file_metadata,
    record_attempt,
    recover_response_tails,
    require_provenance,
    require_run_provenance,
)


def stop_process(process: subprocess.Popen | None) -> None:
    """Terminate an isolated process group, including surviving child workers."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def endpoint_ready(model: str) -> bool:
    """Read bounded model metadata only, without emitting response bodies."""
    try:
        with urlopen(BASE_URL + "/models", timeout=2) as response:
            payload = response.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            return False
        body = json.loads(payload)
        return any(item.get("id") == model for item in body.get("data", []))
    except (URLError, TimeoutError, ValueError, AttributeError, TypeError):
        return False


def wait_ready(server: subprocess.Popen, model: str, timeout: int) -> None:
    """Wait for exact model discovery while checking server liveness."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Model server exited before readiness")
        if endpoint_ready(model):
            return
        time.sleep(1)
    raise TimeoutError("Model server readiness deadline exceeded")


def run_stage(
    command: list[str],
    server: subprocess.Popen,
    commit: Callable,
    interval: int,
    root: Path,
    env: dict,
) -> None:
    """Forward stage stdout and stderr while supervising liveness and persistence."""
    process = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        start_new_session=True,
        stdout=None,
        stderr=None,
    )
    checkpoint = time.monotonic() + interval
    try:
        while process.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Model server exited during the experiment")
            if time.monotonic() >= checkpoint:
                commit()
                print("Persisted experiment checkpoint", flush=True)
                checkpoint = time.monotonic() + interval
            time.sleep(0.5)
        if process.returncode != 0:
            raise RuntimeError(
                "Experiment subprocess failed; partial records are preserved"
            )
        commit()
    finally:
        stop_process(process)


def package_versions() -> dict[str, str]:
    """Record installed runtime versions without inspecting environment secrets."""
    versions = {"python": sys.version.split()[0]}
    for name in ("vllm", "openai", "torch", "transformers", "modal"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def interrupt_job(signum: int, frame: object) -> None:
    """Turn graceful termination into cleanup of child processes and outputs."""
    raise InterruptedError("Modal job received termination")


def execute_job(
    serving: dict,
    experiment: dict,
    profile: dict,
    code_sha256: str,
    commit: Callable,
    *,
    workers: int = 1,
    retry_failed: bool = False,
) -> dict:
    """Run a single model-independent job and persist even on handled failure."""
    serving, experiment, profile = validate_job(serving, experiment, profile)
    from ..inference.experiment_config import validate_worker_count

    validate_worker_count(workers)
    root = repository_root()
    if source_digest(root) != code_sha256:
        raise ValueError("Packaged source fingerprint differs from submitted source")
    model = Path(serving["model_path"])
    if (
        not model.resolve().is_relative_to(Path("/models").resolve())
        or not model.is_dir()
    ):
        raise ValueError("Model checkpoint is missing or outside the model Volume")
    marker = model / "modal_checkpoint.json"
    if not marker.is_file():
        raise ValueError("Checkpoint is not ready: run the CPU downloader first")
    checkpoint = read_json(marker)
    check_file_metadata(model, checkpoint["files"])
    benchmark_path = root / experiment["benchmark_path"]
    staged = read_json(benchmark_path.parent / "modal_images.json")
    if staged["benchmark_sha256"] != sha256_file(benchmark_path):
        raise ValueError("Staged image inventory belongs to another benchmark")
    check_file_metadata(Path("/data/images"), staged["images"])
    runs = root / RUN_ROOT
    metadata = runs / ".modal" / experiment["run_name"]
    identity = {
        "serving": serving,
        "experiment": experiment,
        "profile": profile,
        "source_sha256": code_sha256,
        "checkpoint": checkpoint,
        "versions": package_versions(),
        "workers": workers,
        "server_command": server_command(serving),
    }
    require_run_provenance(runs / experiment["run_name"], metadata, identity)
    attempt = record_attempt(metadata, identity)
    commit()
    print(f"Recorded execution attempt: {attempt.name}", flush=True)
    recover_response_tails(runs / experiment["run_name"], metadata)
    commit()
    server = None
    status = "failed"
    # Forward diagnostics from both streams; vLLM logging controls limit noise.
    env = dict(
        os.environ,
        OPENAI_API_KEY="modal-loopback",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        VLLM_LOGGING_LEVEL="WARNING",
        PYTHONPATH=str(root),
        PYTHONUNBUFFERED="1",
    )
    env.update(serving.get("vllm_env", {}))
    previous_handler = signal.signal(signal.SIGTERM, interrupt_job)
    try:
        # Any listener could be another checkpoint, even with a different alias.
        import socket

        with socket.socket() as connection:
            occupied = connection.connect_ex(("127.0.0.1", 8000)) == 0
        if occupied:
            raise RuntimeError("Port 8000 already serves a model; refusing to reuse it")
        server = subprocess.Popen(
            server_command(serving),
            cwd=root,
            env=env,
            start_new_session=True,
            stdout=None,
            stderr=None,
        )
        print("Waiting for model server", flush=True)
        wait_ready(server, serving["served_model_name"], profile["startup_timeout"])
        with tempfile.TemporaryDirectory(prefix="modal-config-") as directory:
            config_dir = Path(directory)
            write_json(config_dir / "experiment.json", experiment)
            write_json(
                config_dir / "paths.json", {"chexpert_plus_image_root": "/data/images"}
            )
            command = [
                sys.executable,
                "-m",
                PACKAGE + ".inference.run_pipeline",
                "--progress",
                "--experiment",
                str(config_dir / "experiment.json"),
                "--paths",
                str(config_dir / "paths.json"),
                "--workers",
                str(workers),
            ]
            if retry_failed:
                command.append("--retry-failed")
            print("Running benchmark pipeline", flush=True)
            run_stage(command, server, commit, profile["commit_interval"], root, env)
            print("Scoring saved responses", flush=True)
            run_stage(
                [
                    sys.executable,
                    "-m",
                    PACKAGE + ".evaluation.evaluate",
                    "--run-name",
                    experiment["run_name"],
                ],
                server,
                commit,
                profile["commit_interval"],
                root,
                env,
            )
        status = "complete"
        return {"status": status, "run_name": experiment["run_name"]}
    finally:
        try:
            stop_process(server)
        finally:
            try:
                write_json(attempt / "status.json", {"status": status})
                write_json(metadata / "status.json", {"status": status})
                commit()
            finally:
                signal.signal(signal.SIGTERM, previous_handler)

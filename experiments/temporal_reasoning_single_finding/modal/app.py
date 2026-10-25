"""Modal entrypoints; select infrastructure with --profile."""

import json

import modal

from experiments.temporal_reasoning_single_finding.modal.config import (
    PROFILE_ENV,
    runtime_profile,
    BENCHMARK_ROOT,
    RUN_ROOT,
    load_job,
    source_digest,
    package_source,
)

app = modal.App("temporal-reasoning-single-finding")
PROFILE = runtime_profile(modal.is_local())
PYTHON_ALIAS_COMMAND = (
    'if ! command -v python >/dev/null 2>&1; then '
    'python_bin=$(command -v python3) || exit 1; '
    'ln -s "$python_bin" /usr/local/bin/python; fi'
)


def volumes(prefix: str) -> dict:
    """Resolve named persistent stores lazily in the active Modal environment."""
    return {
        name: modal.Volume.from_name(f"{prefix}-{name}", create_if_missing=True)
        for name in ("images", "benchmarks", "models", "results")
    }


stores = volumes(PROFILE["volume_prefix"])
MOUNTS = {
    "/data/images": stores["images"].with_mount_options(read_only=True),
    "/app/" + BENCHMARK_ROOT: stores["benchmarks"].with_mount_options(
        read_only=True
    ),
    "/models": stores["models"].with_mount_options(read_only=True),
    "/app/" + RUN_ROOT: stores["results"],
}
# A model-compatible, versioned image is selected by the user. No GPU build
# or model download is performed on the local machine.
IMAGE = (
    modal.Image.from_registry(PROFILE["image"])
    .entrypoint([])
    .env({"PYTHONPATH": "/app", PROFILE_ENV: json.dumps(PROFILE)})
    .workdir("/app")
)
# Modal's installer calls `python`; some registry images expose only python3.
# Alias the existing interpreter to preserve its installed vLLM environment.
IMAGE = IMAGE.run_commands(PYTHON_ALIAS_COMMAND)
IMAGE = IMAGE.uv_pip_install("openai==2.48.0")
if PROFILE.get("pip_packages"):
    IMAGE = IMAGE.uv_pip_install(*PROFILE["pip_packages"])
if modal.is_local():
    IMAGE = package_source(IMAGE)


@app.function(
    image=IMAGE, volumes=MOUNTS, gpu=PROFILE["gpu"], timeout=PROFILE["timeout"],
    memory=PROFILE.get("memory_mib"),
    region=PROFILE.get("region"), retries=0, max_containers=1,
)
def run_benchmark(
    serving: dict,
    experiment: dict,
    profile: dict,
    code_sha256: str,
    workers: int,
    retry_failed: bool = False,
) -> dict:
    """Lock one run, refresh mounts, and execute the ordinary benchmark pipeline."""
    from experiments.temporal_reasoning_single_finding.modal.launcher import execute_job
    from experiments.temporal_reasoning_single_finding.modal.locks import acquire_call_lock

    stores = volumes(profile["volume_prefix"])
    locks = modal.Dict.from_name(
        profile["volume_prefix"] + "-locks", create_if_missing=True
    )
    key = "run:" + experiment["run_name"]
    acquire_call_lock(locks, key, modal.current_function_call_id(), "Run is locked")
    try:
        for volume in stores.values():
            volume.reload()
        return execute_job(
            serving,
            experiment,
            profile,
            code_sha256,
            stores["results"].commit,
            workers=workers,
            retry_failed=retry_failed,
        )
    finally:
        locks.pop(key)


def main(
    serving: str,
    experiment: str,
    workers: int = 1,
    retry_failed: bool = False,
) -> None:
    """Read local JSON inputs and submit values, never local image payloads."""
    from experiments.temporal_reasoning_single_finding.inference.experiment_config import (
        validate_worker_count,
    )

    selected, experiment_config, profile = load_job(serving, experiment, PROFILE)
    validate_worker_count(workers)
    call = run_benchmark.spawn(
        selected,
        experiment_config,
        profile,
        source_digest(),
        workers,
        retry_failed=retry_failed,
    )
    print(f"Submitted benchmark: {call.object_id}", flush=True)
    print(call.get())

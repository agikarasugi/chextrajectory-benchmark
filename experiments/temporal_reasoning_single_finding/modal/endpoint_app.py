"""Independent CPU evaluation App for both Shared and Dedicated Modal Endpoints."""

import json

import modal

from .config import BENCHMARK_ROOT, RUN_ROOT, package_source
from .endpoint_config import PROFILE_ENV, runtime_profile, validate_job

PROFILE = runtime_profile(modal.is_local())
app = modal.App("temporal-single-endpoint-evaluation")
stores = {
    name: modal.Volume.from_name(
        f"{PROFILE['volume_prefix']}-{name}", create_if_missing=name == "results",
    )
    for name in ("images", "benchmarks", "results")
}
MOUNTS = {
    "/data/images": stores["images"].with_mount_options(read_only=True),
    "/app/" + BENCHMARK_ROOT: stores["benchmarks"].with_mount_options(read_only=True),
    "/app/" + RUN_ROOT: stores["results"],
}
SECRET = modal.Secret.from_name(PROFILE["secret_name"], required_keys=["OPENAI_API_KEY"])
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("openai==2.48.0", "modal==1.5.5")
    .env({"PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1", PROFILE_ENV: json.dumps(PROFILE)})
    .workdir("/app")
)
if modal.is_local():
    IMAGE = package_source(IMAGE)


@app.function(
    image=IMAGE, volumes=MOUNTS, secrets=[SECRET],
    cpu=PROFILE["cpu"], memory=PROFILE["memory"], timeout=PROFILE["timeout"],
    region=PROFILE.get("region"), retries=0, max_containers=1,
    # Source is already present through the explicit allowlisted image copy.
    include_source=False,
)
def run_benchmark(
    experiment: dict, profile: dict, code_sha256: str, workers: int = 1,
    retry_failed: bool = False,
) -> dict:
    """Lock the selected run and refresh the three mounts before direct execution."""
    from .endpoint_launcher import execute_job
    from .locks import acquire_call_lock

    experiment, profile = validate_job(experiment, profile, workers)
    if profile != PROFILE:
        raise ValueError("Submitted endpoint profile differs from mounted resources")
    locks = modal.Dict.from_name(profile["volume_prefix"] + "-locks", create_if_missing=True)
    key = "run:" + experiment["run_name"]
    acquire_call_lock(locks, key, modal.current_function_call_id(), "Run is locked")
    try:
        for volume in stores.values():
            volume.reload()
        result = execute_job(
            experiment, profile, code_sha256, stores["results"].commit, workers=workers,
            retry_failed=retry_failed,
        )
    except BaseException as error:
        try:
            locks.pop(key)
        except BaseException:
            error.add_note("Failed to release the run lock; inspect its owning invocation")
        raise
    else:
        locks.pop(key)
        return result

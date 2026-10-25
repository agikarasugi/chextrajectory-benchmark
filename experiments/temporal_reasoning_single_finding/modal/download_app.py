"""Independent CPU-only Modal App for pinned checkpoint downloads."""

import json

import modal

from experiments.temporal_reasoning_single_finding.modal.config import (
    PROFILE_ENV,
    runtime_profile,
    package_source,
    validate_download_arguments,
)

app = modal.App("temporal-single-checkpoint-download")
PROFILE = runtime_profile(modal.is_local())

store = modal.Volume.from_name(
    PROFILE["volume_prefix"] + "-models", create_if_missing=True
)
DOWNLOAD_MOUNTS = {"/models": store}
DOWNLOAD_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==1.23.0")
    .env({"PYTHONPATH": "/app", PROFILE_ENV: json.dumps(PROFILE)})
    .workdir("/app")
)
if modal.is_local():
    DOWNLOAD_IMAGE = package_source(DOWNLOAD_IMAGE)


@app.function(
    image=DOWNLOAD_IMAGE,
    volumes=DOWNLOAD_MOUNTS,
    timeout=86400,
    retries=0,
    max_containers=1,
)
def fetch_model(repo_id: str, revision: str | None, model_path: str, prefix: str) -> dict:
    """Download a checkpoint on CPU, serializing writers for its destination."""
    from experiments.temporal_reasoning_single_finding.modal.download_model import (
        download_checkpoint,
    )
    from experiments.temporal_reasoning_single_finding.modal.locks import acquire_call_lock

    store = modal.Volume.from_name(prefix + "-models", create_if_missing=True)
    locks = modal.Dict.from_name(prefix + "-locks", create_if_missing=True)
    key = "model:" + model_path
    acquire_call_lock(
        locks, key, modal.current_function_call_id(), "Checkpoint download is already locked"
    )
    try:
        store.reload()
        return download_checkpoint(repo_id, revision, model_path, store.commit)
    finally:
        locks.pop(key)


def download(repo_id: str, revision: str | None, model_path: str, secret: str = "") -> None:
    """Download a pinned model on Modal, optionally injecting an HF_TOKEN Secret."""
    validate_download_arguments(revision, model_path)
    options = {}
    if secret:
        options["secrets"] = [modal.Secret.from_name(secret)]
    if PROFILE.get("region"):
        options["region"] = PROFILE["region"]
    call = fetch_model.with_options(**options).spawn(
        repo_id,
        revision,
        model_path,
        PROFILE["volume_prefix"],
    )
    print(f"Submitted checkpoint download: {call.object_id}", flush=True)
    print(call.get())

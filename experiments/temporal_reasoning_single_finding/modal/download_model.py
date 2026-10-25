"""Download an immutable Hub checkpoint on a CPU worker into a mounted Volume."""

import re
from collections.abc import Callable
from pathlib import Path

from ..artifact_io import read_json, write_json
from .config import under_path


def download_checkpoint(
    repo_id: str,
    revision: str | None,
    model_path: str,
    commit: Callable,
    *,
    model_root: Path = Path("/models"),
) -> dict:
    """Publish readiness only after a pinned download and file-size inventory."""
    from huggingface_hub import HfApi, snapshot_download

    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character commit SHA")
    target = Path(under_path(model_path, str(model_root)))
    if not target.resolve().is_relative_to(model_root.resolve()):
        raise ValueError("Checkpoint destination escapes model Volume")
    target.mkdir(parents=True, exist_ok=True)
    pending = target / "modal_download.json"
    if pending.exists():
        saved_identity = read_json(pending)
        saved_revision = saved_identity.get("revision")
        if not isinstance(saved_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", saved_revision
        ):
            raise ValueError("Checkpoint download record lacks a full commit SHA")
        if saved_identity.get("repo_id") != repo_id:
            raise ValueError("Checkpoint directory belongs to a different repository")
        if revision is not None and revision != saved_revision:
            raise ValueError("Checkpoint directory belongs to a different revision")
        revision = saved_revision
    elif any(target.iterdir()):
        raise ValueError("Checkpoint destination contains unmanaged files")
    else:
        if revision is None:
            revision = HfApi().model_info(repo_id, revision="main").sha
            if not isinstance(revision, str) or not re.fullmatch(
                r"[0-9a-f]{40}", revision
            ):
                raise ValueError("Hub did not resolve main to a full commit SHA")
        identity = {"repo_id": repo_id, "revision": revision}
        write_json(pending, identity)
    identity = {"repo_id": repo_id, "revision": revision}
    commit()
    print(f"Checkpoint revision: {repo_id}@{revision}", flush=True)
    ready = target / "modal_checkpoint.json"
    if ready.exists():
        from .launcher import check_file_metadata

        saved = read_json(ready)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Checkpoint readiness identity changed")
        check_file_metadata(target, saved["files"])
        return {
            "repo_id": repo_id,
            "revision": revision,
            "files": len(saved["files"]),
            "bytes": sum(item["bytes"] for item in saved["files"].values()),
        }
    try:
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=target)
        files = {}
        for path in sorted(target.rglob("*")):
            relative = path.relative_to(target)
            if (
                path.is_file()
                and ".cache" not in relative.parts
                and not path.name.startswith("modal_")
            ):
                files[relative.as_posix()] = {
                    "bytes": path.stat().st_size,
                }
        if not files or "config.json" not in files:
            raise ValueError("Download lacks a model configuration")
        inventory = {**identity, "files": files}
        write_json(target / "modal_checkpoint.json", inventory)
        return {
            "repo_id": repo_id,
            "revision": revision,
            "files": len(files),
            "bytes": sum(item["bytes"] for item in files.values()),
        }
    finally:
        commit()

"""Inventory selected benchmark PNGs and optionally upload them once to Modal."""

import argparse
import hashlib
import json
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..artifact_io import safe_relative_path, sha256_file, write_json
from ..inference.experiment_config import load_private_paths
from ..inference.model_client import _runtime_png
from ..validation.preflight import load_and_validate_benchmark
from .config import BENCHMARK_ROOT, under_path
from .stage_recovery import recover_images


def report(message: str) -> None:
    """Emit immediately visible metadata-only staging progress."""
    print(message, flush=True)


@contextmanager
def phase(label: str, interval: float = 10) -> Iterator[None]:
    """Report elapsed time while a blocking phase has no finer progress API."""
    report(label)
    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        """Keep long operations visibly active without inspecting payloads."""
        while not stopped.wait(interval):
            elapsed = time.monotonic() - started
            report(f"{label}: still working ({elapsed:.0f}s elapsed)")

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def upload_images(
    volume: object, root: Path, names: list[str], images: dict,
    checkpoint: Callable[[list[str]], None],
) -> None:
    """Publish bounded batches and count bytes only after successful completion."""
    total = sum(images[name]["bytes"] for name in names)
    completed_bytes = 0
    report(f"Images to upload: {len(names)} files, {total} bytes")
    for offset in range(0, len(names), 16):
        selected = names[offset : offset + 16]
        end = offset + len(selected)
        with phase(f"Uploading image batch {offset + 1}-{end}/{len(names)}"):
            with volume.batch_upload() as batch:
                for name in selected:
                    batch.put_file(root / name, "/" + name)
        checkpoint(selected)
        completed_bytes += sum(images[name]["bytes"] for name in selected)
        report(
            f"Images uploaded: {end}/{len(names)} files, "
            f"{completed_bytes}/{total} file bytes confirmed"
        )


def inventory(benchmark_path: Path, image_root: Path) -> tuple[dict, dict]:
    """Hash only referenced files without displaying or retaining image payloads."""
    image_root = image_root.resolve()
    report("Validating local benchmark")
    benchmark = load_and_validate_benchmark(benchmark_path)
    report("Hashing referenced local images")
    images = {}
    for item in benchmark.items:
        for image in item["images"]:
            path = _runtime_png(image_root, image["image_id"])
            relative = path.relative_to(image_root).as_posix()
            if relative not in images:
                images[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                if len(images) % 25 == 0:
                    report(f"Local images hashed: {len(images)}")
    report(f"Local images hashed: {len(images)} (complete)")
    return benchmark.manifest, dict(sorted(images.items()))


def remote_index(volume: object, path: str) -> dict:
    """Read a text-only staging index, treating absence as an empty inventory."""
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return {}


def remote_file_digest(volume: object, path: str) -> str:
    """Stream an existing file into a digest without exposing its payload."""
    digest = hashlib.sha256()
    for chunk in volume.read_file(path):
        digest.update(chunk)
    return digest.hexdigest()


def upload(
    benchmark_path: Path,
    relative_manifest: str,
    image_root: Path,
    prefix: str,
    manifest: dict,
    images: dict,
) -> None:
    """Upload missing immutable inputs and publish a per-benchmark image inventory."""
    import modal

    report("Connecting to Modal Volumes and acquiring staging lock")
    image_store = modal.Volume.from_name(prefix + "-images", create_if_missing=True)
    benchmark_store = modal.Volume.from_name(
        prefix + "-benchmarks", create_if_missing=True
    )
    lock = modal.Dict.from_name(prefix + "-locks", create_if_missing=True)
    if not lock.put("staging", "local-input-upload", skip_if_exists=True):
        raise RuntimeError(
            f"Staging lock exists in {prefix}-locks. If the previous upload was "
            "cancelled, confirm no uploader is running before removing the "
            "staging key and retrying. Do not clear an active uploader lock."
        )
    try:
        report("Reading remote image index")
        existing = remote_index(image_store, "/.modal-images.json")
        report(f"Indexed images reusable: {len(images.keys() & existing.keys())}")
        for name, entry in images.items():
            if name in existing and existing[name] != entry:
                raise ValueError("An image path already has different staged content")
        missing_images = []
        unchecked = sorted(images.keys() - existing.keys())
        report(f"Checking {len(unchecked)} unindexed remote image paths")
        recovered = recover_images(image_store, unchecked)
        for name in unchecked:
            entry = recovered[name]
            if entry is None:
                missing_images.append(name)
            elif entry != images[name]:
                raise ValueError("Unindexed image already exists with different content")
            else:
                existing[name] = entry
        report(f"Images reused: {len(images) - len(missing_images)}/{len(images)}")
        with tempfile.TemporaryDirectory(prefix="modal-stage-") as directory:
            temporary = Path(directory)

            def checkpoint(names: list[str]) -> None:
                """Save only recovered or successfully uploaded image entries."""
                existing.update({name: images[name] for name in names})
                write_json(temporary / "index.json", existing)
                with image_store.batch_upload(force=True) as batch:
                    batch.put_file(temporary / "index.json", "/.modal-images.json")
                report(f"Image index saved: {len(existing)} entries")

            # Persist recovered entries before starting any new image upload.
            checkpoint([])
            upload_images(image_store, image_root, missing_images, images, checkpoint)
            relative_dir = Path(relative_manifest).relative_to(BENCHMARK_ROOT).parent
            records = [benchmark_path]
            records.extend(
                safe_relative_path(benchmark_path.parent.resolve(), row["path"])
                for row in manifest["files"].values()
            )
            # Existing frozen files must match. Only missing files are uploaded.
            report("Checking frozen benchmark files")
            missing = []
            for path in records:
                target = (
                    "/"
                    + (
                        relative_dir / path.relative_to(benchmark_path.parent)
                    ).as_posix()
                )
                try:
                    remote_sha = remote_file_digest(benchmark_store, target)
                except FileNotFoundError:
                    missing.append((path, target))
                else:
                    if remote_sha != sha256_file(path):
                        raise ValueError(
                            "Frozen benchmark already exists with different content"
                        )
            report(
                f"Benchmark files: {len(records) - len(missing)} reused, "
                f"{len(missing)} to upload"
            )
            with benchmark_store.batch_upload() as batch:
                for path, target in missing:
                    batch.put_file(path, target)
            report("Publishing benchmark readiness inventory")
            write_json(
                temporary / "inventory.json",
                {
                    "benchmark_sha256": sha256_file(benchmark_path),
                    "images": images,
                },
            )
            with benchmark_store.batch_upload(force=True) as batch:
                batch.put_file(
                    temporary / "inventory.json",
                    "/" + (relative_dir / "modal_images.json").as_posix(),
                )
    finally:
        lock.pop("staging")


def main() -> None:
    """Default to metadata inventory; upload only with the explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--volume-prefix", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", args.volume_prefix):
        raise ValueError("Invalid volume prefix")
    under_path(args.benchmark, BENCHMARK_ROOT)
    path = Path(args.benchmark).resolve()
    image_root = load_private_paths(args.paths)["chexpert_plus_image_root"]
    with phase("Preparing local inventory"):
        manifest, images = inventory(path, image_root)
    print(
        json.dumps(
            {
                "unique_images": len(images),
                "image_bytes": sum(entry["bytes"] for entry in images.values()),
                "upload_requested": args.upload,
            }
        ),
        flush=True,
    )
    if args.upload:
        try:
            with phase("Remote staging"):
                upload(
                    path, args.benchmark, image_root,
                    args.volume_prefix, manifest, images,
                )
        except KeyboardInterrupt:
            report(
                "Staging cancelled. Completed batches may be reused on retry; "
                "readiness is not confirmed."
            )
            raise SystemExit(130) from None
        report("Staging complete")


if __name__ == "__main__":
    main()

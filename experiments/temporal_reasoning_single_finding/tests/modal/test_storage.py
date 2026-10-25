"""Exercise synthetic staging and pinned checkpoint persistence without cloud I/O."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import read_json, write_json
from experiments.temporal_reasoning_single_finding.modal import (
    download_model, stage, stage_recovery,
)


class StorageTests(unittest.TestCase):
    """Check deduplication, download readiness, and immutable revisions."""

    def test_upload_progress_counts_only_completed_batches(self) -> None:
        """A failed batch must not contribute confirmed files or byte counts."""
        from contextlib import contextmanager

        calls = []

        @contextmanager
        def batch_upload():
            """Complete the first batch and fail while publishing the second."""
            batch = Mock()
            calls.append(batch)
            yield batch
            if len(calls) == 2:
                raise RuntimeError("synthetic upload failure")

        names = [f"synthetic-{i}" for i in range(17)]
        images = {name: {"bytes": 10} for name in names}
        volume = SimpleNamespace(batch_upload=batch_upload)
        checkpoint = Mock()
        with patch.object(stage, "report") as output:
            with self.assertRaisesRegex(RuntimeError, "synthetic upload failure"):
                stage.upload_images(volume, Path("/synthetic"), names, images, checkpoint)
        confirmed = [
            call.args[0] for call in output.call_args_list
            if call.args[0].startswith("Images uploaded:")
        ]
        self.assertEqual(confirmed, [
            "Images uploaded: 16/17 files, 160/170 file bytes confirmed"
        ])
        self.assertEqual(calls[0].put_file.call_count, 16)
        self.assertEqual(calls[1].put_file.call_count, 1)
        checkpoint.assert_called_once_with(names[:16])

    def test_phase_heartbeat_stops_after_failure(self) -> None:
        """Blocking phases emit a heartbeat and join their thread on failure."""
        import threading

        tick = threading.Event()

        def capture(message: str) -> None:
            """Signal a heartbeat without reading any input payload."""
            if "still working" in message:
                tick.set()

        with patch.object(stage, "report", side_effect=capture) as output:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with stage.phase("Synthetic phase", interval=0.01):
                    self.assertTrue(tick.wait(2))
                    raise RuntimeError("synthetic")
            count = output.call_count
            self.assertGreaterEqual(count, 2)
        self.assertFalse(any(
            thread.name.endswith("(heartbeat)") for thread in threading.enumerate()
        ))

    def test_index_failure_stops_before_next_batch(self) -> None:
        """Do not continue uploads when a completed batch cannot be indexed."""
        from contextlib import nullcontext

        batch = Mock()
        volume = Mock()
        volume.batch_upload.return_value = nullcontext(batch)
        names = [f"synthetic-{i}" for i in range(17)]
        images = {name: {"bytes": 10} for name in names}
        checkpoint = Mock(side_effect=RuntimeError("index failure"))
        with self.assertRaisesRegex(RuntimeError, "index failure"):
            stage.upload_images(volume, Path("/synthetic"), names, images, checkpoint)
        self.assertEqual(volume.batch_upload.call_count, 1)
        self.assertEqual(batch.put_file.call_count, 16)
        checkpoint.assert_called_once_with(names[:16])

    def test_recovery_uses_cpu_mount_and_metadata_batches(self) -> None:
        """Cloud recovery needs no profile and sends bounded path-only requests."""
        import sys
        from contextlib import nullcontext

        sdk = Mock()
        app = sdk.App.return_value
        app.run.return_value = nullcontext()
        function = app.function.return_value.return_value
        function.remote.side_effect = lambda names: dict.fromkeys(names)
        volume = Mock()
        names = [f"synthetic-{i}" for i in range(257)]
        with patch.dict(sys.modules, {"modal": sdk}):
            self.assertEqual(stage_recovery.recover_images(volume, []), {})
            sdk.App.assert_not_called()
            self.assertEqual(
                stage_recovery.recover_images(volume, names), dict.fromkeys(names)
            )
            self.assertEqual(function.remote.call_args_list[0].args, (names[:256],))
            self.assertEqual(function.remote.call_args_list[1].args, (names[256:],))
            options = app.function.call_args.kwargs
            self.assertNotIn("gpu", options)
            self.assertTrue(options["serialized"])
            volume.with_mount_options.assert_called_once_with(read_only=True)
            function.remote.side_effect = lambda names: {}
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                stage_recovery.recover_images(volume, ["synthetic"])

    def test_cloud_hashing_returns_metadata_and_rejects_escape(self) -> None:
        """Recovery returns only metadata for synthetic files inside its mount."""
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic").write_bytes(b"synthetic content")
            result = stage_recovery.hash_paths(["synthetic", "absent"], directory)
            self.assertEqual(result, {
                "synthetic": {
                    "bytes": 17,
                    "sha256": hashlib.sha256(b"synthetic content").hexdigest(),
                },
                "absent": None,
            })
            for name in ["../escape", "/absolute", "."]:
                with self.assertRaises(ValueError):
                    stage_recovery.hash_paths([name], directory)
            (root / "escape").symlink_to(root.parent)
            with self.assertRaises(ValueError):
                stage_recovery.hash_paths(["escape/file"], directory)

    def test_inventory_deduplicates_reused_images(self) -> None:
        """Only unique referenced synthetic files contribute upload bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.png").write_bytes(b"synthetic fixture, not an image")
            image = {"image_id": "synthetic.jpg"}
            benchmark = SimpleNamespace(manifest={}, items=[{"images": [image, image]}])
            with patch.object(
                stage, "load_and_validate_benchmark", return_value=benchmark
            ):
                _, files = stage.inventory(root / "manifest.json", root)
            self.assertEqual(list(files), ["synthetic.png"])
            self.assertEqual(files["synthetic.png"]["bytes"], 31)

    def test_inventory_rejects_missing_files(self) -> None:
        """A missing selected image blocks staging before uploads."""
        with tempfile.TemporaryDirectory() as directory:
            benchmark = SimpleNamespace(
                manifest={}, items=[{"images": [{"image_id": "missing.jpg"}]}]
            )
            with (
                patch.object(
                    stage, "load_and_validate_benchmark", return_value=benchmark
                ),
                self.assertRaises(FileNotFoundError),
            ):
                stage.inventory(Path(directory) / "manifest.json", Path(directory))

    def test_download_marker_and_repeat_without_redownload(self) -> None:
        """Reuse a completed checkpoint without reading or hashing its payloads."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "checkpoint"

            def fetch(**kwargs: object) -> None:
                """Write synthetic checkpoint contents in the mocked downloader."""
                (target / "config.json").write_text("{}")
                (target / "weights.safetensors").write_bytes(b"synthetic weights")

            commit = Mock()
            original_open = Path.open

            def metadata_only(path: Path, *args, **kwargs):
                """Allow downloader writes and metadata reads, but no weight reads."""
                mode = args[0] if args else kwargs.get("mode", "r")
                if path.name == "weights.safetensors" and "r" in mode:
                    raise AssertionError("Checkpoint payload was reread")
                return original_open(path, *args, **kwargs)

            with patch(
                "huggingface_hub.snapshot_download", side_effect=fetch
            ) as download, patch.object(Path, "open", metadata_only):
                first = download_model.download_checkpoint(
                    "owner/model", "a" * 40, str(target), commit, model_root=root
                )
                second = download_model.download_checkpoint(
                    "owner/model", "a" * 40, str(target), commit, model_root=root
                )
            inventory = read_json(target / "modal_checkpoint.json")
            self.assertTrue(all(set(entry) == {"bytes"} for entry in inventory["files"].values()))
            # Existing markers remain usable without migration or a new download.
            for entry in inventory["files"].values():
                entry["sha256"] = "legacy-hash"
            write_json(target / "modal_checkpoint.json", inventory)
            with patch.object(Path, "open", metadata_only):
                legacy = download_model.download_checkpoint(
                    "owner/model", "a" * 40, str(target), commit, model_root=root
                )
            self.assertEqual(first, legacy)
            self.assertEqual(first, second)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(
                len(read_json(target / "modal_checkpoint.json")["files"]), 2
            )
            self.assertGreaterEqual(commit.call_count, 3)
            with self.assertRaises(ValueError):
                download_model.download_checkpoint(
                    "owner/model", "b" * 40, str(target), commit, model_root=root
                )

    def test_interrupted_download_has_no_readiness_marker(self) -> None:
        """Persist pending state but never advertise an incomplete model."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "checkpoint"
            commit = Mock()
            with (
                patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=RuntimeError("synthetic"),
                ),
                self.assertRaises(RuntimeError),
            ):
                download_model.download_checkpoint(
                    "owner/model", "a" * 40, str(target), commit, model_root=root
                )
            self.assertFalse((target / "modal_checkpoint.json").exists())
            self.assertTrue((target / "modal_download.json").exists())
            self.assertEqual(commit.call_count, 2)

    def test_unpinned_revision_is_rejected(self) -> None:
        """A moving branch must not define experiment checkpoint identity."""
        with self.assertRaises(ValueError):
            download_model.download_checkpoint(
                "owner/model", "main", "/models/model", Mock()
            )

    def test_remote_index_only_treats_missing_as_empty(self) -> None:
        """Do not silently overwrite storage when an index is corrupt."""
        volume = Mock()
        volume.read_file.side_effect = FileNotFoundError
        self.assertEqual(stage.remote_index(volume, "/index.json"), {})
        volume.read_file.side_effect = None
        volume.read_file.return_value = [b"invalid JSON"]
        with self.assertRaises(ValueError):
            stage.remote_index(volume, "/index.json")

    def test_upload_reuses_files_and_recovers_missing_index(self) -> None:
        """Repeated uploads preserve inputs, including orphaned indexed metadata."""
        import hashlib
        import sys

        class MemoryVolume:
            """Keep synthetic upload payloads in an isolated in-memory store."""

            def __init__(self) -> None:
                """Initialize empty file storage and upload counts."""
                self.files = {}
                self.writes = []

            def read_file(self, name: str) -> list[bytes]:
                """Expose only the test's synthetic bytes to the staging helper."""
                if name.endswith(".png"):
                    raise AssertionError("Image payload downloaded locally")
                if name not in self.files:
                    raise FileNotFoundError(name)
                return [self.files[name]]

            def batch_upload(self, force: bool = False) -> object:
                """Create a context implementing Modal's upload protocol."""
                store = self

                class Batch:
                    """Minimal SDK upload context for local verification."""

                    def __enter__(self) -> object:
                        """Return the upload facade."""
                        return self

                    def __exit__(self, *args: object) -> None:
                        """Finish without suppressing errors."""

                    def put_file(self, local: Path, remote: str) -> None:
                        """Apply the requested immutable or explicit index write."""
                        if remote in store.files and not force:
                            raise FileExistsError(remote)
                        store.files[remote] = Path(local).read_bytes()
                        store.writes.append(remote)

                return Batch()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_root = root / "images"
            images_root.mkdir()
            (images_root / "synthetic.png").write_bytes(b"synthetic")
            benchmark = root / "manifest.json"
            benchmark.write_text("{}")
            images = {
                "synthetic.png": {
                    "bytes": 9,
                    "sha256": hashlib.sha256(b"synthetic").hexdigest(),
                }
            }
            image_store, benchmark_store = MemoryVolume(), MemoryVolume()
            sdk = Mock()
            sdk.Volume.from_name.side_effect = lambda name, **kw: (
                image_store if name.endswith("-images") else benchmark_store
            )
            sdk.Dict.from_name.return_value.put.return_value = True
            relative = stage.BENCHMARK_ROOT + "/smoke/manifest.json"
            def recover(volume: object, names: list[str]) -> dict:
                """Emulate cloud hashing without local image read_file calls."""
                return {
                    name: {
                        "bytes": len(volume.files["/" + name]),
                        "sha256": hashlib.sha256(volume.files["/" + name]).hexdigest(),
                    } if "/" + name in volume.files else None
                    for name in names
                }

            with (
                patch.dict(sys.modules, {"modal": sdk}),
                patch.object(stage, "recover_images", side_effect=recover),
            ):
                stage.upload(
                    benchmark, relative, images_root, "test", {"files": {}}, images
                )
                del image_store.files["/.modal-images.json"]
                stage.upload(
                    benchmark, relative, images_root, "test", {"files": {}}, images
                )
                self.assertEqual(image_store.writes.count("/synthetic.png"), 1)
                self.assertEqual(
                    benchmark_store.writes.count("/smoke/manifest.json"), 1
                )
                with self.assertRaises(ValueError):
                    stage.upload(
                        benchmark,
                        relative,
                        images_root,
                        "test",
                        {"files": {}},
                        {"synthetic.png": {"bytes": 1, "sha256": "changed"}},
                    )

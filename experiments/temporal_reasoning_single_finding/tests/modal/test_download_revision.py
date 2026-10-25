"""Verify latest selection, immutable recovery, and download identity offline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    read_json,
    write_json,
)
from experiments.temporal_reasoning_single_finding.modal.download_model import (
    download_checkpoint,
)


class DownloadRevisionTests(unittest.TestCase):
    """Exercise checkpoint selection without network or cloud resources."""

    def setUp(self) -> None:
        """Create an isolated destination and mock the Hub entry points."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "checkpoint"
        self.sha = "a" * 40
        self.commit = Mock()
        self.api = self.enterContext(patch("huggingface_hub.HfApi"))
        self.api.return_value.model_info.return_value.sha = self.sha
        self.fetch = self.enterContext(
            patch("huggingface_hub.snapshot_download", side_effect=self.write_model)
        )

    def write_model(self, **kwargs: object) -> None:
        """Assert identity is persisted before writing synthetic model files."""
        self.assertEqual(kwargs["revision"], self.sha)
        self.assertEqual(
            read_json(self.target / "modal_download.json"),
            {"repo_id": "owner/model", "revision": self.sha},
        )
        self.commit.assert_called()
        (self.target / "config.json").write_text("{}")

    def run_download(self, revision: str | None = None) -> dict:
        """Invoke the downloader against the isolated model root."""
        return download_checkpoint(
            "owner/model", revision, str(self.target), self.commit,
            model_root=self.root,
        )

    def test_latest_is_resolved_once_and_completed_checkpoint_reused(self) -> None:
        """A later main commit cannot silently replace the recorded checkpoint."""
        first = self.run_download()
        self.api.return_value.model_info.return_value.sha = "b" * 40
        second = self.run_download()
        self.assertEqual(first, second)
        self.assertEqual(first["revision"], self.sha)
        self.api.return_value.model_info.assert_called_once_with(
            "owner/model", revision="main"
        )
        self.fetch.assert_called_once_with(
            repo_id="owner/model", revision=self.sha, local_dir=self.target
        )
        self.assertEqual(
            read_json(self.target / "modal_checkpoint.json")["revision"], self.sha
        )

    def test_interrupted_download_resumes_saved_commit(self) -> None:
        """Recovery uses the original SHA even after upstream main advances."""
        self.fetch.side_effect = RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_download()
        self.assertFalse((self.target / "modal_checkpoint.json").exists())
        self.api.return_value.model_info.return_value.sha = "b" * 40
        self.fetch.side_effect = self.write_model
        self.assertEqual(self.run_download()["revision"], self.sha)
        self.api.return_value.model_info.assert_called_once()

    def test_explicit_revision_skips_resolution(self) -> None:
        """Pinned requests retain their original behavior."""
        self.assertEqual(self.run_download(self.sha)["revision"], self.sha)
        self.api.assert_not_called()

    def test_identity_conflicts_fail_without_hub_calls(self) -> None:
        """Repository and explicit SHA conflicts preserve the existing record."""
        self.target.mkdir()
        identity = {"repo_id": "owner/model", "revision": self.sha}
        pending = self.target / "modal_download.json"
        write_json(pending, identity)
        for repo_id, revision in [("other/model", None), ("owner/model", "b" * 40)]:
            with self.subTest(repo_id=repo_id, revision=revision):
                with self.assertRaisesRegex(ValueError, "belongs to a different"):
                    download_checkpoint(
                        repo_id, revision, str(self.target), self.commit,
                        model_root=self.root,
                    )
                self.assertEqual(read_json(pending), identity)
        self.api.assert_not_called()
        self.fetch.assert_not_called()
        self.commit.assert_not_called()

    def test_unmanaged_destination_fails_before_resolution(self) -> None:
        """Latest selection cannot take ownership of existing unmanaged files."""
        self.target.mkdir()
        (self.target / "config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            self.run_download()
        self.api.assert_not_called()
        self.fetch.assert_not_called()

    def test_failed_or_invalid_resolution_does_not_publish_identity(self) -> None:
        """Resolution errors leave no download or readiness record."""
        info = self.api.return_value.model_info
        for result in [None, "main", "abc123", RuntimeError("unavailable")]:
            with self.subTest(result=result):
                info.side_effect = result if isinstance(result, Exception) else None
                info.return_value.sha = result
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_download()
                self.assertEqual(list(self.target.iterdir()), [])
        self.fetch.assert_not_called()
        self.commit.assert_not_called()

    def test_invalid_saved_revision_is_rejected(self) -> None:
        """A malformed saved identity cannot turn recovery into a moving download."""
        self.target.mkdir()
        write_json(self.target / "modal_download.json", {
            "repo_id": "owner/model", "revision": "main",
        })
        with self.assertRaisesRegex(ValueError, "full commit SHA"):
            self.run_download()
        self.api.assert_not_called()
        self.fetch.assert_not_called()

"""Check external and migrated run adoption without cloud or model calls."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from ...artifact_io import read_json, sha256_file, write_json
from ...modal.runtime import require_run_provenance
from ...protocol.contracts import LC_INFERENCE_POLICY
from ...validation.preflight import ValidatedBenchmark
from ..evaluation.fixtures import item, write_run


class ImportedRunTests(unittest.TestCase):
    """Validate prepared input identity before recording a first Modal attempt."""

    def setUp(self) -> None:
        """Build a synthetic saved run and replace only benchmark disk loading."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        items = [item("sample", ["unchanged", "unchanged"],
                      ["constant", "none", "T1-T3", "none"])]
        self.run = write_run(self.root, items, lc_inference_policy=LC_INFERENCE_POLICY)
        experiment = read_json(self.run / "run_manifest.json")["experiment"]
        benchmark_path = Path(experiment["benchmark_path"])
        benchmark = ValidatedBenchmark(
            benchmark_path, read_json(benchmark_path), {3: items}, items,
        )
        patcher = patch(
            "experiments.temporal_reasoning_single_finding.validation.preflight."
            "load_and_validate_benchmark", return_value=benchmark,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.metadata = self.run.parent / ".modal" / self.run.name
        self.identity = {
            "experiment": experiment,
            "serving": {"model_path": "/models/pinned", "vllm_args": []},
            "checkpoint": {"revision": "pinned"},
        }

    def test_external_partial_run_is_adopted_without_rewriting_artifacts(self) -> None:
        """Missing LC responses are allowed and existing responses remain intact."""
        (self.run / "responses/pair_lc.jsonl").unlink()
        before = {p: sha256_file(p) for p in self.run.rglob("*") if p.is_file()}
        require_run_provenance(self.run, self.metadata, self.identity)
        imported = read_json(self.metadata / "import.json")
        self.assertEqual(imported["origin"], "external")
        self.assertEqual(imported["run_manifest_sha256"], before[self.run / "run_manifest.json"])
        self.assertEqual(read_json(self.metadata / "provenance.json"), self.identity)
        self.assertEqual(before, {p: sha256_file(p) for p in before})
        snapshot = {p: p.read_bytes() for p in self.metadata.iterdir()}
        changed = deepcopy(self.identity)
        changed["serving"]["vllm_args"] = ["--max-model-len", "9639"]
        require_run_provenance(self.run, self.metadata, changed)
        self.assertEqual(snapshot, {p: p.read_bytes() for p in snapshot})

    def test_import_rejects_changed_request_and_corrupt_prepared_inputs(self) -> None:
        """Reject incompatible settings and damaged inputs before writing provenance."""
        for change in ({"model": "different"}, {"max_completion_tokens": 512}):
            changed = deepcopy(self.identity)
            changed["experiment"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(
                ValueError, "does not match this experiment",
            ):
                require_run_provenance(self.run, self.metadata, changed)
            self.assertFalse(self.metadata.exists())
        path = self.run / "prepared/pair_inputs.json"
        path.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            require_run_provenance(self.run, self.metadata, self.identity)
        self.assertFalse(self.metadata.exists())

    def test_migrated_run_preserves_and_checks_source_provenance(self) -> None:
        """Link a renamed run to its existing history without changing the source."""
        source_name = "original-run"
        source_path = self.metadata.parent / source_name / "provenance.json"
        original = deepcopy(self.identity)
        original["experiment"]["run_name"] = source_name
        write_json(source_path, original)
        before = source_path.read_bytes()
        write_json(self.run / "migration.json", {
            "source_run_name": source_name, "destination_run_name": self.run.name,
        })
        changed = deepcopy(self.identity)
        changed["checkpoint"]["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint.revision"):
            require_run_provenance(self.run, self.metadata, changed)
        self.assertFalse(self.metadata.exists())
        require_run_provenance(self.run, self.metadata, self.identity)
        imported = read_json(self.metadata / "import.json")
        self.assertEqual(imported["origin"], "modal_migration")
        self.assertEqual(imported["source_run_name"], source_name)
        self.assertEqual(imported["source_provenance_sha256"], sha256_file(source_path))
        self.assertEqual(source_path.read_bytes(), before)

    def test_migrated_external_run_and_invalid_source_name(self) -> None:
        """An external migration is accepted but cannot escape the metadata tree."""
        path = self.run / "migration.json"
        write_json(path, {
            "source_run_name": "../outside", "destination_run_name": self.run.name,
        })
        with self.assertRaises(ValueError):
            require_run_provenance(self.run, self.metadata, self.identity)
        self.assertFalse(self.metadata.exists())
        write_json(path, {
            "source_run_name": "pc-original", "destination_run_name": self.run.name,
        })
        require_run_provenance(self.run, self.metadata, self.identity)
        imported = read_json(self.metadata / "import.json")
        self.assertEqual(imported["origin"], "external")
        self.assertEqual(imported["source_run_name"], "pc-original")

"""Verify smoke sampling, parent preservation, call budgets, and frozen publication."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from experiments.temporal_reasoning_single_finding.artifact_io import (
    sha256_file, write_json,
)
from experiments.temporal_reasoning_single_finding.construction import (
    constructor_gold_pilot as pilot,
)
from experiments.temporal_reasoning_single_finding.construction.constructor_smoke_v2 import (
    construct_smoke_v2, select_smoke_items,
)
from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS
from experiments.temporal_reasoning_single_finding.protocol.task_selection import (
    SMOKE_V2_BENCHMARK_VERSION, selection_summary, validate_task_selection,
)
from experiments.temporal_reasoning_single_finding.tests.construction.test_source_data import (
    _audit_row, _record, _write_audit,
)
from experiments.temporal_reasoning_single_finding.validation.preflight import (
    load_and_validate_benchmark,
)


class SmokeV2Tests(unittest.TestCase):
    """Build synthetic parent metadata without images or model endpoints."""

    def setUp(self) -> None:
        """Create a valid assigned parent with spare candidates in every cell."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source_paths = {length: self.root / f"source-{length}.json" for length in (3, 4)}
        records = []
        for length in (3, 4):
            rows = [
                _record(
                    patient_number=length * 100 + index,
                    labels=["New"] + ["Stable"] * (length - 2), tier="gold",
                )
                for index in range(16)
            ]
            records.extend(rows)
            write_json(source_paths[length], rows)
        audit = self.root / "audit.parquet"
        _write_audit(audit, [_audit_row(record) for record in records])
        quotas = {length: {task: 4 for task in TASKS} for length in (3, 4)}
        with patch.object(pilot, "TASK_QUOTAS", quotas):
            result = pilot.construct_gold_pilot(
                three_tp_source=source_paths[3], four_tp_source=source_paths[4],
                audit_sidecar=audit, output_root=self.root / "parent",
            )
        self.parent_path = Path(result["gold_manifest_path"])
        self.parent = load_and_validate_benchmark(self.parent_path)
        # Derived construction must work without the original sources or sidecar.
        for path in [*source_paths.values(), audit]:
            path.unlink()

    def _construct(self, directory: str = "output") -> dict:
        """Construct from the frozen fixture parent into an isolated directory."""
        return construct_smoke_v2(
            parent_manifest=self.parent_path, output_root=self.root / directory,
        )

    def test_balanced_sample_preserves_parent_and_exact_call_budget(self) -> None:
        """Keep assignments and references while producing 16 calls per answer type."""
        before = {path.name: sha256_file(path) for path in self.parent_path.parent.iterdir()}
        result = self._construct()
        benchmark = load_and_validate_benchmark(Path(result["smoke_manifest_path"]))
        self.assertEqual(len(benchmark.items), 16)
        self.assertEqual(len({item["patient_id"] for item in benchmark.items}), 16)
        parent_by_id = {item["trajectory_id"]: item for item in self.parent.items}
        for item in benchmark.items:
            expected = dict(
                parent_by_id[item["trajectory_id"]],
                benchmark_version=SMOKE_V2_BENCHMARK_VERSION,
            )
            self.assertEqual(item, expected)
        for length in (3, 4):
            self.assertEqual(
                Counter(
                    item["assigned_task"] for item in benchmark.items
                    if item["timepoint_count"] == length
                ),
                Counter({task: 2 for task in TASKS}),
            )
        plan = result["call_plan"]
        self.assertEqual(plan["pair_predictions"], 40)
        for key in (
            "seq_predictions", "seq_dir", "ref_lc",
            "pair_lc_potential", "seq_lc_potential",
        ):
            self.assertEqual(plan[key], 16)
        self.assertEqual(plan["potential_logical_calls_before_retries"], 120)
        self.assertEqual(before, {
            path.name: sha256_file(path)
            for path in self.parent_path.parent.iterdir()
        })

    def test_determinism_and_identical_reuse(self) -> None:
        """Input order and destination do not affect bytes or rewrite frozen files."""
        self.assertEqual(
            select_smoke_items(self.parent.items),
            select_smoke_items(list(reversed(self.parent.items))),
        )
        first = Path(self._construct()["smoke_manifest_path"]).parent
        snapshot = {
            path.name: (sha256_file(path), path.stat().st_mtime_ns)
            for path in first.iterdir()
        }
        self._construct()
        self.assertEqual(snapshot, {
            path.name: (sha256_file(path), path.stat().st_mtime_ns)
            for path in first.iterdir()
        })
        second = Path(self._construct("second")["smoke_manifest_path"]).parent
        self.assertEqual(
            {name: digest for name, (digest, _) in snapshot.items()},
            {path.name: sha256_file(path) for path in second.iterdir()},
        )

    def test_conflicting_output_is_preserved(self) -> None:
        """A changed frozen file causes failure without replacement."""
        path = Path(self._construct()["smoke_manifest_path"]).parent / "3tp.json"
        path.write_text("[]\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._construct()
        self.assertEqual(path.read_text(), "[]\n")

    def test_insufficient_candidates_fail_and_patient_reuse_is_allowed(self) -> None:
        """Fail short quotas but allow overlap when distinct patients are exhausted."""
        short = [item for item in self.parent.items if item["assigned_task"] != TASKS[0]]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_smoke_items(short)
        shared = copy.deepcopy(self.parent.items)
        for item in shared:
            item["patient_id"] = "one-patient"
        self.assertEqual(sum(map(len, select_smoke_items(shared).values())), 16)

    def test_wrong_quota_and_missing_assignment_contract_fail(self) -> None:
        """Reject internally consistent summaries that violate smoke v2 quotas."""
        manifest = {"benchmark_version": SMOKE_V2_BENCHMARK_VERSION}
        items = [item for rows in select_smoke_items(self.parent.items).values() for item in rows]
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_task_selection(manifest, items)
        items[0]["assigned_task"] = next(
            task for task in TASKS if task != items[0]["assigned_task"]
        )
        manifest["task_selection"] = selection_summary(items)
        with self.assertRaisesRegex(ValueError, "two Gold"):
            validate_task_selection(manifest, items)

    def test_preflight_rejects_altered_derivation(self) -> None:
        """Reject a changed selection rule even with unchanged valid item files."""
        path = Path(self._construct()["smoke_manifest_path"])
        benchmark = load_and_validate_benchmark(path)
        benchmark.manifest["construction"]["seed"] = 99
        write_json(path, benchmark.manifest)
        with self.assertRaisesRegex(ValueError, "construction.seed"):
            load_and_validate_benchmark(path)


if __name__ == "__main__":
    unittest.main()

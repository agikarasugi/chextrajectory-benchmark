"""Tests for frozen benchmark Reference Check behavior."""

import unittest

from experiments.temporal_reasoning_single_finding.protocol.contracts import TASKS
from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
    derive_answer,
)
from experiments.temporal_reasoning_single_finding.validation.reference_check import (
    run_reference_check,
    validate_reference_check_summary,
)


def _item(labels: tuple[str, ...]) -> dict[str, object]:
    """Build one complete task fixture from a direction vector."""
    tasks: dict[str, dict[str, object]] = {}
    for task in TASKS:
        answer = derive_answer(task, labels)
        if answer == "no_identifiable_most_severe_visit_or_plateau":
            tasks[task] = {
                "eligible": False,
                "reference_answer": None,
                "exclusion_reason": "no_unique_most_severe_visit_or_plateau",
            }
        else:
            tasks[task] = {
                "eligible": True,
                "reference_answer": answer,
                "exclusion_reason": None,
            }
    return {
        "trajectory_id": "chexpert_plus:fixture",
        "timepoint_count": len(labels) + 1,
        "reference_change_labels": list(labels),
        "tasks": tasks,
    }


class ReferenceCheckTests(unittest.TestCase):
    """Exercise eligible, excluded, and mismatched references."""

    def test_counts_every_task_and_excluded_most_severe(self) -> None:
        item = _item(("improvement", "deterioration"))

        summary = run_reference_check([item])

        self.assertEqual(
            summary,
            {
                "status": "passed",
                "checked_task_entries": 4,
                "eligible_final_cases": 3,
                "ineligible_most_severe_task_entries": 1,
                "mismatches": 0,
            },
        )

    def test_rejects_reference_mismatch(self) -> None:
        item = _item(("deterioration", "improvement"))
        item["tasks"]["trajectory_shape"]["reference_answer"] = "constant"

        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            run_reference_check([item])

    def test_rejects_ineligible_always_defined_task(self) -> None:
        item = _item(("unchanged", "unchanged"))
        item["tasks"]["turning_regions"] = {
            "eligible": False,
            "reference_answer": None,
            "exclusion_reason": "no_unique_most_severe_visit_or_plateau",
        }

        with self.assertRaisesRegex(ValueError, "cannot be ineligible"):
            run_reference_check([item])

    def test_validates_manifest_summary_exactly(self) -> None:
        summary = run_reference_check([_item(("unchanged", "unchanged"))])
        self.assertEqual(validate_reference_check_summary(summary), summary)

        invalid = dict(summary, mismatches=1)
        with self.assertRaisesRegex(ValueError, "mismatches"):
            validate_reference_check_summary(invalid)


if __name__ == "__main__":
    unittest.main()

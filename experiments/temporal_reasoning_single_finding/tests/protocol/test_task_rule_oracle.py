"""Compare production task rules with an independent frozen answer oracle."""

import itertools
import json
from pathlib import Path
import unittest

from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
    derive_answer,
)


FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "task_rule_oracle.json"
)
CHANGE_LABELS = ("deterioration", "unchanged", "improvement")
TASKS = (
    "trajectory_shape",
    "turning_regions",
    "most_severe_visit_or_plateau",
    "first_improvement_between_visits",
)


class TaskRuleOracleTests(unittest.TestCase):
    """Check all task-vector answers against literal, non-production values."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the static oracle without using production test helpers."""
        cls.payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_oracle_has_exact_exhaustive_coverage(self) -> None:
        """Require one complete literal answer set for all 3TP and 4TP vectors."""
        self.assertEqual(
            set(self.payload),
            {"schema_version", "cases"},
        )
        self.assertEqual(
            self.payload["schema_version"],
            "task-rule-oracle-v1",
        )
        cases = self.payload["cases"]
        self.assertIsInstance(cases, list)
        self.assertEqual(len(cases), 36)

        expected_vectors = {
            vector
            for interval_count in (2, 3)
            for vector in itertools.product(
                CHANGE_LABELS,
                repeat=interval_count,
            )
        }
        actual_vectors: set[tuple[str, ...]] = set()
        answer_count = 0
        for case in cases:
            self.assertEqual(set(case), {"change_labels", "answers"})
            labels = tuple(case["change_labels"])
            self.assertIn(len(labels), (2, 3))
            self.assertTrue(all(label in CHANGE_LABELS for label in labels))
            self.assertNotIn(labels, actual_vectors)
            actual_vectors.add(labels)

            answers = case["answers"]
            self.assertEqual(set(answers), set(TASKS))
            self.assertTrue(
                all(isinstance(value, str) for value in answers.values())
            )
            answer_count += len(answers)

        self.assertEqual(actual_vectors, expected_vectors)
        self.assertEqual(answer_count, 144)

    def test_production_rules_match_every_oracle_answer(self) -> None:
        """Match each production answer to its independently stored exact value."""
        for case in self.payload["cases"]:
            labels = tuple(case["change_labels"])
            for task in TASKS:
                with self.subTest(change_labels=labels, task=task):
                    self.assertEqual(
                        derive_answer(task, labels),
                        case["answers"][task],
                    )


if __name__ == "__main__":
    unittest.main()

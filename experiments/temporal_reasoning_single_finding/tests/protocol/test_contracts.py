"""Tests for frozen names and vocabularies."""

import unittest

from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    SETTINGS,
    TASKS,
    answer_choices,
    interval_keys,
    validate_change_labels,
)


class ContractTests(unittest.TestCase):
    """Verify exact public protocol tokens."""

    def test_frozen_names(self) -> None:
        """Only approved tasks and settings are exposed."""
        self.assertEqual(
            TASKS,
            (
                "trajectory_shape",
                "turning_regions",
                "most_severe_visit_or_plateau",
                "first_improvement_between_visits",
            ),
        )
        self.assertEqual(
            SETTINGS,
            ("SEQ-DIR", "PAIR-FC", "REF-LC", "SEQ-FC", "PAIR-LC", "SEQ-LC"),
        )

    def test_length_specific_vocabularies(self) -> None:
        """Sequence keys and answers follow the frozen lengths."""
        self.assertEqual(interval_keys(3), ("T1_to_T2", "T2_to_T3"))
        self.assertEqual(len(answer_choices("turning_regions", 4)), 9)
        self.assertNotIn("fluctuating", answer_choices("trajectory_shape", 3))
        self.assertIn("fluctuating", answer_choices("trajectory_shape", 4))
        with self.assertRaises(ValueError):
            validate_change_labels(["Deterioration"], 2)


if __name__ == "__main__":
    unittest.main()

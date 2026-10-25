"""Exhaustive tests for deterministic final-task rules."""

import itertools
import unittest

from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    CHANGE_LABELS,
    NO_IDENTIFIABLE,
    TASKS,
    answer_choices,
)
from experiments.temporal_reasoning_single_finding.protocol.task_rules import (
    derive_answer,
    first_improvement_between_visits,
    most_severe_visit_or_plateau,
    trajectory_shape,
    turning_regions,
)


class TaskRuleTests(unittest.TestCase):
    """Cover every possible 3TP and 4TP label vector."""

    def test_every_vector_maps_to_frozen_vocabulary(self) -> None:
        """All 9 and 27 vectors produce only approved task values."""
        for timepoint_count in (3, 4):
            vectors = list(itertools.product(CHANGE_LABELS, repeat=timepoint_count - 1))
            self.assertEqual(len(vectors), 9 if timepoint_count == 3 else 27)
            for labels in vectors:
                for task in TASKS:
                    answer = derive_answer(task, labels)
                    if (
                        task == "most_severe_visit_or_plateau"
                        and answer == NO_IDENTIFIABLE
                    ):
                        continue
                    self.assertIn(answer, answer_choices(task, timepoint_count))

    def test_most_severe_uses_qualitative_order(self) -> None:
        """Only 8 of 9 and 20 of 27 vectors identify one maximum."""
        for length, expected_undefined in ((2, 1), (3, 7)):
            answers = [
                most_severe_visit_or_plateau(labels)
                for labels in itertools.product(CHANGE_LABELS, repeat=length)
            ]
            self.assertEqual(answers.count(NO_IDENTIFIABLE), expected_undefined)
        self.assertEqual(
            most_severe_visit_or_plateau(
                ("deterioration", "unchanged", "improvement")
            ),
            "T2-T3",
        )

    def test_representative_rules(self) -> None:
        """Direction switches, plateaus, and first improvement are exact."""
        labels = ("deterioration", "unchanged", "improvement")
        self.assertEqual(trajectory_shape(labels), "single_peak")
        self.assertEqual(turning_regions(labels), "peak:T2-T3")
        self.assertEqual(first_improvement_between_visits(labels), "T3-T4")
        self.assertEqual(
            turning_regions(("deterioration", "improvement", "deterioration")),
            "peak:T2,trough:T3",
        )


if __name__ == "__main__":
    unittest.main()

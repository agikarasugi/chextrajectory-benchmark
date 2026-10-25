"""Independent examples for balanced accuracy and all-case FC/LC scoring."""

import unittest

from ...evaluation.metrics import _task_metrics
from ...protocol.contracts import REFERENCE_SUPPORTED_LC_POLICY, NO_IDENTIFIABLE, SETTINGS
from ...validation.validate_run import ValidatedRun


def score(
    references: list[str],
    fc: list[tuple[str, str | None]],
    lc: list[tuple[str, str | None]],
    *,
    task: str = "turning_regions",
    count: int = 3,
    gated: bool = False,
) -> dict:
    """Score literal outcomes independently of inference and task derivation."""
    cases = [
        {"reference_answer": answer, "result_ids": dict.fromkeys(SETTINGS, str(i))}
        for i, answer in enumerate(references)
    ]
    run = ValidatedRun(
        manifest={"lc_inference_policy": REFERENCE_SUPPORTED_LC_POLICY} if gated else {},
        prepared={}, pair_predictions={}, seq_predictions={}, coverage={},
        results={
            setting: {
                str(i): {"status": status, "predicted_answer": answer}
                for i, (status, answer) in enumerate(
                    fc if setting.endswith("-FC") else lc
                )
            }
            for setting in SETTINGS
        },
    )
    return _task_metrics(cases, run, task, count, len(cases))


class TaskMetricTests(unittest.TestCase):
    """Check denominators, class balancing, and unsuccessful answer handling."""

    def test_majority_prediction_and_failed_answers(self) -> None:
        """Always predicting none gives 75% accuracy but 50% balanced accuracy."""
        references = ["none", "none", "none", "peak:T2"]
        outcomes = [("valid", "none")] * 4
        result = score(references, outcomes, outcomes)
        for metrics in result["settings"].values():
            self.assertEqual(metrics["final_answer_accuracy"], {
                "correct": 3, "total": 4, "accuracy": 0.75,
            })
            self.assertEqual(metrics["balanced_accuracy"], {
                "value": 0.5,
                "per_class": {
                    "none": {"support": 3, "correct": 3, "recall": 1.0},
                    "peak:T2": {"support": 1, "correct": 0, "recall": 0.0},
                },
            })
        outcomes[0] = ("invalid_response", None)
        result = score(references, outcomes, outcomes)
        for metrics in result["settings"].values():
            self.assertEqual(metrics["final_answer_accuracy"]["accuracy"], 0.5)
            self.assertAlmostEqual(metrics["balanced_accuracy"]["value"], 1 / 3)
            self.assertEqual(metrics["balanced_accuracy"]["per_class"]["none"], {
                "support": 3, "correct": 2, "recall": 2 / 3,
            })

    def test_complete_turning_answer_is_one_class(self) -> None:
        """A partial match to a multiple-turn answer earns no class recall."""
        references = ["none", "none", "peak:T2,trough:T3"]
        outcomes = [("valid", "none"), ("valid", "none"), ("valid", "peak:T2")]
        result = score(references, outcomes, outcomes, count=4)
        for metrics in result["settings"].values():
            self.assertEqual(metrics["balanced_accuracy"]["value"], 0.5)
            self.assertEqual(set(metrics["balanced_accuracy"]["per_class"]), {
                "none", "peak:T2,trough:T3",
            })

    def test_conditional_accuracy_and_matching_wrong_answers(self) -> None:
        """LC can recover wrong FC answers, while matching wrong answers agree."""
        fc = [("valid", answer) for answer in ("T2", "T2", "T1", "T1")]
        lc = [("valid", "T2"), ("invalid_response", None),
              ("valid", "T2"), ("valid", "T1")]
        for gated in (False, True):
            with self.subTest(gated=gated):
                # Scoring uses all cases regardless of the saved inference policy.
                result = score(["T2"] * 4, fc, lc,
                               task="most_severe_visit_or_plateau", gated=gated)
                for method in ("PAIR", "SEQ"):
                    metrics = result["settings"][f"{method}-LC"]
                    for group in ("correct", "wrong"):
                        self.assertEqual(metrics[f"accuracy_when_fc_{group}"], {
                            "correct": 1, "total": 2, "accuracy": 0.5,
                        })
                    self.assertEqual(
                        result["rule_based_answer_agreement"]
                        [f"{method}-LC vs {method}-FC"],
                        {"agreements": 2, "total": 4, "excluded": 0,
                         "agreement_rate": 0.5},
                    )

    def test_unsuccessful_outcomes_are_wrong_and_never_agree(self) -> None:
        """Undefined, invalid, failed, and skipped answers stay in denominators."""
        unsuccessful = [
            ("invalid_response", None), ("response_error", None),
            ("upstream_invalid", None), ("upstream_response_error", None),
            (NO_IDENTIFIABLE, None), ("upstream_answer_mismatch", None),
            ("valid", None), ("valid", NO_IDENTIFIABLE),
        ]
        for outcome in unsuccessful:
            for fc, lc in ((outcome, outcome), (outcome, ("valid", "T2")),
                           (("valid", "T2"), outcome)):
                with self.subTest(fc=fc, lc=lc):
                    result = score(["T2"], [fc], [lc],
                                   task="most_severe_visit_or_plateau")
                    group = "correct" if fc == ("valid", "T2") else "wrong"
                    lc_correct = int(lc == ("valid", "T2"))
                    for method in ("PAIR", "SEQ"):
                        metrics = result["settings"][f"{method}-LC"]
                        self.assertEqual(metrics[f"accuracy_when_fc_{group}"], {
                            "correct": lc_correct, "total": 1,
                            "accuracy": float(lc_correct),
                        })
                        self.assertEqual(metrics["balanced_accuracy"]["value"],
                                         float(lc_correct))
                        self.assertEqual(
                            result["rule_based_answer_agreement"]
                            [f"{method}-LC vs {method}-FC"],
                            {"agreements": 0, "total": 1, "excluded": 0,
                             "agreement_rate": 0.0},
                        )

    def test_empty_groups_are_null(self) -> None:
        """Empty tasks and absent FC groups have zero counts and null accuracy."""
        result = score([], [], [])
        for metrics in result["settings"].values():
            self.assertEqual(metrics["balanced_accuracy"], {
                "value": None, "per_class": {},
            })
        for method in ("PAIR", "SEQ"):
            for group in ("correct", "wrong"):
                self.assertEqual(
                    result["settings"][f"{method}-LC"][f"accuracy_when_fc_{group}"],
                    {"correct": 0, "total": 0, "accuracy": None},
                )
            self.assertEqual(
                result["rule_based_answer_agreement"][f"{method}-LC vs {method}-FC"],
                {"agreements": 0, "total": 0, "excluded": 0, "agreement_rate": None},
            )
        result = score(["none"], [("valid", "none")], [("valid", "none")])
        for method in ("PAIR", "SEQ"):
            self.assertEqual(
                result["settings"][f"{method}-LC"]["accuracy_when_fc_wrong"],
                {"correct": 0, "total": 0, "accuracy": None},
            )


if __name__ == "__main__":
    unittest.main()

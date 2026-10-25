"""Approved equal-question and trajectory-occurrence descriptive metrics."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from ..protocol.contracts import (
    ANNOTATION_SOURCES,
    CHANGE_LABELS,
    DATASET,
    MODEL_BACKED_SETTINGS,
    NO_IDENTIFIABLE,
    REFERENCE_SUPPORTED_LC_POLICY,
    UPSTREAM_ANSWER_MISMATCH,
    SETTINGS,
    TASKS,
    answer_choices,
)
from ..validation.validate_run import ValidatedRun
from ..protocol.task_rules import derive_answer
from ..protocol.task_selection import TASK_SELECTION_CONTRACT
from . import EVALUATOR_VERSION, SCHEMA_VERSION


def ratio(numerator: int, denominator: int) -> float | None:
    """Return a fraction, or JSON-compatible null for a zero denominator."""
    return numerator / denominator if denominator else None


def _accuracy(correct: int, total: int, field: str = "value") -> dict[str, Any]:
    """Keep exact correctness counts beside their accuracy."""
    return {"correct": correct, "total": total, field: ratio(correct, total)}


def _rate(count: int, total: int) -> dict[str, Any]:
    """Keep the failure count beside its rate."""
    return {"count": count, "rate": ratio(count, total)}


def _has_answer(row: dict[str, Any]) -> bool:
    """Require a valid, defined answer before correctness or agreement checks."""
    answer = row["predicted_answer"]
    return (
        row["status"] == "valid"
        and isinstance(answer, str)
        and bool(answer)
        and answer != NO_IDENTIFIABLE
    )


def _balanced_accuracy(
    cases: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Average reference-class recall, counting unsuccessful outputs as misses."""
    support = Counter(case["reference_answer"] for case in cases)
    correct = Counter(
        case["reference_answer"]
        for case, row in zip(cases, rows, strict=True)
        if _has_answer(row) and row["predicted_answer"] == case["reference_answer"]
    )
    per_class = {
        answer: {
            "support": support[answer],
            "correct": correct[answer],
            "recall": correct[answer] / support[answer],
        }
        for answer in sorted(support)
    }
    return {
        "value": (
            sum(entry["recall"] for entry in per_class.values()) / len(per_class)
            if per_class else None
        ),
        "per_class": per_class,
    }


def _distribution(
    values: Iterable[str], vocabulary: Iterable[str], count_name: str
) -> dict[str, Any]:
    """Include zero counts for each requested category in stable order."""
    counts = Counter(values)
    total = sum(counts.values())
    return {
        name: {count_name: counts[name], "proportion": ratio(counts[name], total)}
        for name in vocabulary
    }


def _direct_failures(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count submitted logical records once, excluding propagated LC skips."""
    statuses = Counter(
        row["status"] for row in rows if row["model_input_id"] is not None
    )
    total = sum(statuses.values())
    return {
        "submitted_logical_calls": total,
        "invalid_response": _rate(statuses["invalid_response"], total),
        "technical_failure": _rate(statuses["response_error"], total),
    }


def _change_metrics(
    trajectories: list[dict[str, Any]],
    cases_by_seq: dict[str, dict[str, Any]],
    run: ValidatedRun,
    method: str,
    timepoint_count: int,
) -> dict[str, Any]:
    """Score every interval occurrence and each complete trajectory once."""
    true_positives: Counter[str] = Counter()
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    error_counts: Counter[int] = Counter()
    interval_total = 0
    interval_correct = 0
    for trajectory in trajectories:
        seq_id = trajectory["prediction_id"]
        references = trajectory["reference_change_labels"]
        if method == "PAIR":
            # Dependencies already preserve chronological order. Reused calls
            # intentionally count again in every containing trajectory.
            dependencies = cases_by_seq[seq_id]["pair_dependency_ids"]
            predictions = [
                run.pair_predictions[identifier]["predicted_change_label"]
                if run.pair_predictions[identifier]["status"] == "valid"
                else None
                for identifier in dependencies
            ]
        else:
            row = run.seq_predictions[seq_id]
            predictions = (
                row["predicted_change_labels"]
                if row["status"] == "valid"
                else [None] * len(references)
            )
        errors = 0
        for reference, predicted in zip(references, predictions, strict=True):
            interval_total += 1
            if predicted == reference:
                interval_correct += 1
                true_positives[reference] += 1
            else:
                errors += 1
                false_negatives[reference] += 1
                # An unusable output has no predicted class. It contributes
                # a false negative only, never a fabricated false positive.
                if predicted is not None:
                    false_positives[predicted] += 1
        error_counts[errors] += 1

    per_class = {}
    for label in CHANGE_LABELS:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        per_class[label] = {
            "counts": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            },
            "metrics": {
                "precision": ratio(tp, tp + fp),
                "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
            },
        }
    total = len(trajectories)
    return {
        "interval_accuracy": _accuracy(interval_correct, interval_total),
        "whole_trajectory_accuracy": _accuracy(error_counts[0], total),
        "per_class": per_class,
        "interval_error_counts": {
            str(errors): {
                "trajectories": error_counts[errors],
                "proportion": ratio(error_counts[errors], total),
            }
            for errors in range(timepoint_count)
        },
    }


def _task_metrics(
    cases: list[dict[str, Any]],
    run: ValidatedRun,
    task: str,
    timepoint_count: int,
    trajectory_total: int,
    reference_eligible_total: int | None = None,
) -> dict[str, Any]:
    """Score selected questions equally and distinguish reference eligibility."""
    total = len(cases)
    reference_supported = (
        run.manifest.get("lc_inference_policy") == REFERENCE_SUPPORTED_LC_POLICY
    )
    settings = {}
    for setting in SETTINGS:
        rows = [run.results[setting][case["result_ids"][setting]] for case in cases]
        correct = sum(
            _has_answer(row)
            and row["predicted_answer"] == case["reference_answer"]
            for case, row in zip(cases, rows, strict=True)
        )
        metrics = {
            "final_answer_accuracy": _accuracy(correct, total, "accuracy"),
            "balanced_accuracy": _balanced_accuracy(cases, rows),
        }
        if setting in {"PAIR-FC", "SEQ-FC", "PAIR-LC", "SEQ-LC"}:
            statuses = Counter(row["status"] for row in rows)
            metrics["downstream_failure_rates"] = {
                ("selected_questions" if reference_eligible_total is not None
                 else "eligible_questions"): total,
                "invalid_response": _rate(
                    statuses["upstream_invalid"] + statuses[NO_IDENTIFIABLE], total
                ),
                "technical_failure": _rate(statuses["upstream_response_error"], total),
            }
        settings[setting] = metrics
        if reference_supported and setting in {"PAIR-LC", "SEQ-LC"}:
            metrics["downstream_failure_rates"]["upstream_answer_mismatch"] = _rate(
                statuses[UPSTREAM_ANSWER_MISMATCH], total
            )

    agreement = {}
    for method in ("PAIR", "SEQ"):
        agreements = 0
        group_totals = Counter()
        group_correct = Counter()
        fc_setting, lc_setting = f"{method}-FC", f"{method}-LC"
        for case in cases:
            fc = run.results[fc_setting][case["result_ids"][fc_setting]]
            lc = run.results[lc_setting][case["result_ids"][lc_setting]]
            fc_correct = (
                _has_answer(fc)
                and fc["predicted_answer"] == case["reference_answer"]
            )
            group = "fc_correct" if fc_correct else "fc_wrong"
            group_totals[group] += 1
            group_correct[group] += (
                _has_answer(lc)
                and lc["predicted_answer"] == case["reference_answer"]
            )
            agreements += (
                _has_answer(fc) and _has_answer(lc)
                and lc["predicted_answer"] == fc["predicted_answer"]
            )
        for group in ("fc_correct", "fc_wrong"):
            settings[lc_setting][f"accuracy_when_{group}"] = _accuracy(
                group_correct[group], group_totals[group], "accuracy"
            )
        agreement[f"{lc_setting} vs {fc_setting}"] = {
            "agreements": agreements,
            "total": total,
            "excluded": 0,
            "agreement_rate": ratio(agreements, total),
        }
    population = {
        "eligible_questions": total,
        "excluded_questions": trajectory_total - total,
        "reference_answers": _distribution(
            (case["reference_answer"] for case in cases),
            answer_choices(task, timepoint_count),
            "questions",
        ),
    }
    if reference_eligible_total is not None:
        findings = [case["target_finding"]["canonical_name"] for case in cases]
        population.pop("excluded_questions")
        population.update({
            "eligible_questions": reference_eligible_total,
            "ineligible_questions": trajectory_total - reference_eligible_total,
            "selected_questions": total,
            "unselected_eligible_questions": reference_eligible_total - total,
            "selected_patients": len({case["patient_id"] for case in cases}),
            "selected_target_findings": _distribution(
                findings, sorted(set(findings)), "questions"
            ),
        })
    return {
        "population": population,
        "settings": settings,
        "rule_based_answer_agreement": agreement,
    }


def summarize_run(run: ValidatedRun) -> dict[str, Any]:
    """Build the approved nested summary from one validated saved run."""
    trajectories_by_cohort = defaultdict(list)
    cases_by_group = defaultdict(list)
    cases_by_seq = {}
    assigned_mode = any("assigned_task" in row for row in run.prepared["seq_inputs"])
    for trajectory in run.prepared["seq_inputs"]:
        key = (trajectory["timepoint_count"], trajectory["annotation_source"])
        trajectories_by_cohort[key].append(trajectory)
    for case in run.prepared["final_cases"]:
        key = (case["timepoint_count"], case["annotation_source"], case["task"])
        cases_by_group[key].append(case)
        cases_by_seq[case["seq_dependency_id"]] = case

    cohorts = {}
    for count in (3, 4):
        for source in ANNOTATION_SOURCES:
            trajectories = trajectories_by_cohort.get((count, source))
            if not trajectories:
                continue
            interval_total = sum(
                len(t["reference_change_labels"]) for t in trajectories
            )
            findings = [t["target_finding"]["canonical_name"] for t in trajectories]
            cohorts[f"{count}tp_{source}"] = {
                "population": {
                    "patients": len({t["patient_id"] for t in trajectories}),
                    "trajectories": len(trajectories),
                    "interval_positions": interval_total,
                    "target_findings": _distribution(
                        findings, sorted(set(findings)), "trajectories"
                    ),
                    "reference_change_labels": _distribution(
                        (
                            label for t in trajectories
                            for label in t["reference_change_labels"]
                        ),
                        CHANGE_LABELS,
                        "interval_positions",
                    ),
                },
                "change_label_metrics": {
                    method: _change_metrics(
                        trajectories, cases_by_seq, run, method, count
                    )
                    for method in ("PAIR", "SEQ")
                },
                "tasks": {
                    task: _task_metrics(
                        cases_by_group[(count, source, task)], run, task, count,
                        len(trajectories),
                        sum(
                            derive_answer(task, row["reference_change_labels"])
                            != NO_IDENTIFIABLE for row in trajectories
                        ) if assigned_mode else None,
                    )
                    for task in TASKS
                },
            }
    direct_rows = {
        "PAIR": run.pair_predictions,
        "SEQ": run.seq_predictions,
        **{setting: run.results[setting] for setting in MODEL_BACKED_SETTINGS},
    }
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            **({"lc_inference_policy": run.manifest["lc_inference_policy"]}
               if "lc_inference_policy" in run.manifest else {}),
            **({"task_selection_contract": TASK_SELECTION_CONTRACT}
               if assigned_mode else {}),
            "run_name": run.manifest["experiment"]["run_name"],
            "benchmark_version": run.manifest["benchmark_version"],
            "dataset": DATASET,
            "cohorts": list(cohorts),
        },
        "direct_failure_rates": {
            stage: _direct_failures(rows.values())
            for stage, rows in direct_rows.items()
        },
        "cohorts": cohorts,
    }

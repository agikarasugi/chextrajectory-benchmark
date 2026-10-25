"""Deterministic patient-first sampling for the frozen pilot."""

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from experiments.temporal_reasoning_single_finding.protocol.contracts import (
    ANNOTATION_SOURCES,
    DATASET,
)

from .source_data import SourceCandidate


PILOT_SEED = 42
PILOT_QUOTA_PER_GROUP = 100


def patient_first_sample(
    candidates: Sequence[SourceCandidate],
    *,
    quota: int,
    seed: int = PILOT_SEED,
) -> tuple[SourceCandidate, ...]:
    """Sample in rounds with at most one trajectory per patient per round."""
    if type(quota) is not int or quota <= 0:
        raise ValueError("quota must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if len(candidates) < quota:
        raise ValueError(
            f"Eligible subgroup has {len(candidates)} trajectories, below quota {quota}"
        )
    group_keys = {
        (candidate.timepoint_count, candidate.annotation_source)
        for candidate in candidates
    }
    if len(group_keys) != 1:
        raise ValueError("patient_first_sample requires exactly one source subgroup")
    if len({candidate.source_sequence_id for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("Sampling candidates contain duplicate trajectory IDs")

    by_patient: dict[tuple[str, str], list[SourceCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_patient[(DATASET, candidate.patient_id)].append(candidate)

    patient_keys = sorted(by_patient)
    for patient_key in patient_keys:
        by_patient[patient_key].sort(
            key=lambda candidate: candidate.source_sequence_id
        )

    generator = random.Random(seed)
    generator.shuffle(patient_keys)
    for patient_key in patient_keys:
        generator.shuffle(by_patient[patient_key])

    selected: list[SourceCandidate] = []
    round_index = 0
    while len(selected) < quota:
        added_this_round = False
        for patient_key in patient_keys:
            patient_candidates = by_patient[patient_key]
            if round_index >= len(patient_candidates):
                continue
            selected.append(patient_candidates[round_index])
            added_this_round = True
            if len(selected) == quota:
                break
        if not added_this_round:
            raise AssertionError("Patient-first sampling exhausted candidates early")
        round_index += 1
    if len({candidate.source_sequence_id for candidate in selected}) != quota:
        raise AssertionError("Patient-first sampling selected a trajectory twice")
    return tuple(selected)


def select_pilot_candidates(
    candidates_by_timepoint: Mapping[int, Sequence[SourceCandidate]],
    *,
    quota_per_group: int = PILOT_QUOTA_PER_GROUP,
    seed: int = PILOT_SEED,
) -> dict[int, tuple[SourceCandidate, ...]]:
    """Select the four independent timepoint-by-annotation-source groups."""
    if set(candidates_by_timepoint) != {3, 4}:
        raise ValueError("Pilot construction requires exactly 3TP and 4TP sources")
    selected: dict[int, tuple[SourceCandidate, ...]] = {}
    for timepoint_count in (3, 4):
        source_candidates = candidates_by_timepoint[timepoint_count]
        if any(
            candidate.timepoint_count != timepoint_count
            for candidate in source_candidates
        ):
            raise ValueError("Candidate appears under the wrong timepoint group")
        group_selection: list[SourceCandidate] = []
        for annotation_source in ANNOTATION_SOURCES:
            subgroup = [
                candidate
                for candidate in source_candidates
                if candidate.timepoint_count == timepoint_count
                and candidate.annotation_source == annotation_source
            ]
            group_selection.extend(
                patient_first_sample(
                    subgroup,
                    quota=quota_per_group,
                    seed=seed,
                )
            )
        selected[timepoint_count] = tuple(group_selection)
    return selected

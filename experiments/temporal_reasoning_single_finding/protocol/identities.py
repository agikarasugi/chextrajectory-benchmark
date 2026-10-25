"""Canonical SHA-256 identities for predictions, cases, results, and inputs."""

import hashlib
import json
from typing import Any

from .contracts import (
    ANNOTATION_SOURCES,
    SETTINGS,
    validate_task,
)


def canonical_json(value: Any) -> str:
    """Serialize one identity value as canonical UTF-8 JSON text."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def pair_prediction_id(
    *,
    dataset: str,
    annotation_source: str,
    earlier_image_id: str,
    later_image_id: str,
    target_finding: str,
    model: str,
    pair_prompt_version: str,
    json_schema_version: str,
    benchmark_version: str,
) -> str:
    """Build the exact source-specific PAIR prediction identity."""
    if annotation_source not in ANNOTATION_SOURCES:
        raise ValueError(f"Unsupported annotation source: {annotation_source}")
    return canonical_sha256(
        {
            "annotation_source": annotation_source,
            "benchmark_version": benchmark_version,
            "dataset": dataset,
            "earlier_image_id": earlier_image_id,
            "json_schema_version": json_schema_version,
            "later_image_id": later_image_id,
            "model": model,
            "pair_prompt_version": pair_prompt_version,
            "prediction_type": "PAIR",
            "target_finding": target_finding,
        }
    )


def seq_prediction_id(
    *,
    source_sequence_id: str,
    model: str,
    seq_prompt_version: str,
    json_schema_version: str,
    benchmark_version: str,
) -> str:
    """Build the SEQ identity from the namespaced source sequence."""
    return canonical_sha256(
        {
            "benchmark_version": benchmark_version,
            "json_schema_version": json_schema_version,
            "model": model,
            "prediction_type": "SEQ",
            "seq_prompt_version": seq_prompt_version,
            "source_sequence_id": source_sequence_id,
        }
    )


def final_case_id(*, source_sequence_id: str, task: str) -> str:
    """Build a setting-neutral final-answer case identity."""
    validate_task(task)
    return canonical_sha256({"source_sequence_id": source_sequence_id, "task": task})


def result_id(*, case_id: str, setting: str) -> str:
    """Build one of six uniform setting-specific result identities."""
    if setting not in SETTINGS:
        raise ValueError(f"Unsupported setting: {setting}")
    return canonical_sha256({"final_case_id": case_id, "setting": setting})


def model_input_id(
    *,
    safe_model_input: list[dict[str, Any]],
    model: str,
    user_prompt_version: str,
    json_schema_version: str,
    benchmark_version: str,
) -> str:
    """Fingerprint exactly what the model receives without image bytes."""
    return canonical_sha256(
        {
            "benchmark_version": benchmark_version,
            "json_schema_version": json_schema_version,
            "model": model,
            "safe_model_input": safe_model_input,
            "user_prompt_version": user_prompt_version,
        }
    )

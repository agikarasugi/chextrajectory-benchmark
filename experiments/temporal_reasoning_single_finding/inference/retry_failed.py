"""Reopen terminal technical failures using the ordinary missing-record resume."""

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..artifact_io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from ..validation.validate_run import load_validated_run


_RESPONSE_IDS = {
    "responses/pair.jsonl": "prediction_id",
    "responses/seq.jsonl": "prediction_id",
    **{
        f"responses/{stage}.jsonl": "result_id"
        for stage in ("seq_dir", "ref_lc", "pair_lc", "seq_lc")
    },
}
_DERIVED_AND_REPORTS = {
    "derived/pair_fc.json", "derived/seq_fc.json",
    "validation/coverage.json", "evaluation.json",
}


def _apply_pending(run_dir: Path) -> dict[str, Any]:
    """Finish an idempotent cleanup before any inference can resume."""
    pending = run_dir / "retry_pending.json"
    plan = read_json(pending)
    removals = plan["remove_response_ids"]
    deletions = plan["remove_files"]
    if (
        not set(removals) <= _RESPONSE_IDS.keys()
        or not set(deletions) <= _DERIVED_AND_REPORTS
    ):
        raise ValueError("Retry plan contains an unexpected artifact path")
    for relative in deletions:
        (run_dir / relative).unlink(missing_ok=True)
    for relative, identifiers in removals.items():
        path = run_dir / relative
        excluded = set(identifiers)
        field = _RESPONSE_IDS[relative]
        write_jsonl(
            path, [row for row in read_jsonl(path) if row[field] not in excluded],
        )
    pending.unlink()
    return {
        key: plan[key]
        for key in ("response_errors", "downstream_results", "backup_directory")
    }


def prepare_failed_retry(
    run_dir: Path, *, retry_failed: bool = False,
) -> dict[str, Any] | None:
    """Back up and reopen failures in a validated complete run, or finish cleanup.

    Only allowlisted run text is copied. A pending plan survives interruption
    between atomic file updates and is replayed before normal inference.
    """
    if (run_dir / "retry_pending.json").exists():
        return _apply_pending(run_dir)
    if not retry_failed:
        return None
    validated = load_validated_run(run_dir)
    removals = {}
    for relative, field in _RESPONSE_IDS.items():
        identifiers = [
            row[field] for row in read_jsonl(run_dir / relative)
            if row["status"] == "response_error"
        ]
        if identifiers:
            removals[relative] = identifiers
    if not removals:
        return {"response_errors": 0, "downstream_results": 0}

    error_count = sum(map(len, removals.values()))
    deletions = {"validation/coverage.json", "evaluation.json"}
    downstream_count = 0
    for source in ("pair", "seq"):
        failed = set(removals.get(f"responses/{source}.jsonl", []))
        if not failed:
            continue
        deletions.add(f"derived/{source}_fc.json")
        # Technical upstream failures have no LC call or late model input yet.
        # Keep all other LC responses and model inputs exactly as saved.
        affected = [
            row["result_id"]
            for row in validated.results[f"{source.upper()}-LC"].values()
            if failed.intersection(row["dependency_ids"])
        ]
        downstream_count += len(affected)
        removals.setdefault(f"responses/{source}_lc.jsonl", []).extend(affected)

    archive = run_dir / "retries" / uuid4().hex
    archive.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for relative in sorted(set(removals) | deletions):
        path = run_dir / relative
        if path.exists():
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            checksums[relative] = sha256_file(target)
    plan = {
        "response_errors": error_count,
        "downstream_results": downstream_count,
        "backup_directory": str(archive.relative_to(run_dir)),
        "source_sha256": checksums,
        "remove_response_ids": removals,
        "remove_files": sorted(deletions),
    }
    write_json(archive / "manifest.json", plan)
    # Publish the recovery plan only once every original file has been saved.
    write_json(run_dir / "retry_pending.json", plan)
    return _apply_pending(run_dir)

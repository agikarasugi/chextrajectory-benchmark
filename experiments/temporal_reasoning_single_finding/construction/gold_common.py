"""Shared source loading and frozen publication for the two Gold constructors."""

import tempfile
from pathlib import Path
from typing import Any

from ..artifact_io import sha256_file, write_json
from .benchmark_artifacts import (
    build_manifest, source_files, validate_cross_source_references,
)
from .source_data import build_benchmark_item, load_audit_index, load_source_candidates


def _publish_or_reuse(staged: Path, destination: Path) -> None:
    """Publish the complete directory atomically or reuse identical frozen files."""
    if not destination.exists() and not destination.is_symlink():
        staged.rename(destination)
        return
    filenames = {"manifest.json", "3tp.json", "4tp.json"}
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or {path.name for path in destination.iterdir()} != filenames
        or any(
            (destination / name).is_symlink()
            or not (destination / name).is_file()
            or sha256_file(destination / name) != sha256_file(staged / name)
            for name in filenames
        )
    ):
        raise FileExistsError(
            f"Conflicting frozen benchmark artifacts at {destination}. "
            "Review source changes and use a new benchmark version."
        )


def load_gold_population(
    *, three_tp_source: Path, four_tp_source: Path, audit_sidecar: Path,
    benchmark_version: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Load every valid Gold candidate and separate tier-specific provenance."""
    audit = load_audit_index(audit_sidecar)
    sources = {
        count: load_source_candidates(
            path, timepoint_count=count, audit_index=audit
        )
        for count, path in ((3, three_tp_source), (4, four_tp_source))
    }
    validate_cross_source_references(sources)
    items = {
        count: sorted(
            (
                build_benchmark_item(
                    candidate, benchmark_version=benchmark_version
                )
                for candidate in sources[count].candidates
                if candidate.annotation_source == "gold"
            ),
            key=lambda item: item["trajectory_id"],
        )
        for count in (3, 4)
    }
    if any(not items[count] for count in (3, 4)):
        raise ValueError(
            "Gold construction requires valid Gold trajectories in both 3TP and 4TP"
        )
    provenance = source_files(
        three_tp_source=three_tp_source,
        four_tp_source=four_tp_source,
        audit_sidecar=audit_sidecar,
        audit_record_count=audit.record_count,
        source_results=sources,
    )
    counts = {
        f"{count}tp": sources[count].counts_by_annotation_source["gold"]
        for count in (3, 4)
    }
    return items, provenance, counts


def write_gold_benchmark(
    *, output_root: Path, directory_name: str, benchmark_version: str,
    items: dict[int, list[dict[str, Any]]], provenance: dict[str, Any],
    construction: dict[str, Any],
) -> dict[str, Any]:
    """Stage a complete validated Gold artifact set and publish or reuse it."""
    if directory_name not in {"gold_v1", "gold_pilot_v1"}:
        raise ValueError("Unsupported Gold output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / directory_name
    with tempfile.TemporaryDirectory(
        prefix=f".{directory_name}-", dir=output_root
    ) as temporary:
        staged = Path(temporary) / directory_name
        paths = {count: staged / f"{count}tp.json" for count in (3, 4)}
        for count in (3, 4):
            write_json(paths[count], items[count])
        manifest = build_manifest(
            benchmark_version=benchmark_version, items_by_timepoint=items,
            file_paths=paths, source_files=provenance, construction=construction,
        )
        write_json(staged / "manifest.json", manifest)
        _publish_or_reuse(staged, destination)
    return {
        "gold_manifest": manifest,
        "gold_manifest_path": str((destination / "manifest.json").resolve()),
    }

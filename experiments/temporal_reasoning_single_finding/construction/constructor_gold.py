"""Construct Gold v1 from every valid Gold source trajectory, without sampling."""

import json
from pathlib import Path
from typing import Any

from .benchmark_artifacts import (
    DEFAULT_AUDIT_SIDECAR, DEFAULT_BENCHMARK_ROOT,
    DEFAULT_FOUR_TP_SOURCE, DEFAULT_THREE_TP_SOURCE,
)
from .gold_common import load_gold_population, write_gold_benchmark
from .source_data import SOURCE_CONTRACT_RECONCILIATION


GOLD_BENCHMARK_VERSION = "temporal-reasoning-gold-v1"


def construct_gold(
    *,
    three_tp_source: Path = DEFAULT_THREE_TP_SOURCE,
    four_tp_source: Path = DEFAULT_FOUR_TP_SOURCE,
    audit_sidecar: Path = DEFAULT_AUDIT_SIDECAR,
    output_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> dict[str, Any]:
    """Validate all sources, retain all valid Gold rows, and freeze Gold v1."""
    items, provenance, counts = load_gold_population(
        three_tp_source=three_tp_source, four_tp_source=four_tp_source,
        audit_sidecar=audit_sidecar, benchmark_version=GOLD_BENCHMARK_VERSION,
    )
    return write_gold_benchmark(
        output_root=output_root, directory_name="gold_v1",
        benchmark_version=GOLD_BENCHMARK_VERSION,
        items=items, provenance=provenance,
        construction={
            "kind": "gold_only",
            "selection_rule": "all-valid-gold-v1",
            "gold_source_counts": counts,
            "source_contract_reconciliation": SOURCE_CONTRACT_RECONCILIATION,
        },
    )


def main() -> None:
    """Construct Gold v1 at the default artifact root without making model calls."""
    result = construct_gold()
    print(json.dumps({"gold_manifest": result["gold_manifest_path"]}, indent=2))


if __name__ == "__main__":
    main()

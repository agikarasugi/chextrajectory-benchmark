"""Evaluate one complete saved run offline and write its evaluation.json."""

import argparse
from collections.abc import Sequence
from typing import Any

from ..artifact_io import assert_no_image_payload, repository_root, write_nested_json
from ..inference.experiment_config import RUN_NAME_PATTERN, RUNS_RELATIVE_ROOT
from ..validation.validate_run import load_validated_run
from .metrics import summarize_run


def evaluate_run(run_name: str) -> dict[str, Any]:
    """Validate and evaluate exactly one canonical run without model calls."""
    if not isinstance(run_name, str) or RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError("run_name must match lowercase [a-z0-9][a-z0-9._-]*")
    run_dir = repository_root() / RUNS_RELATIVE_ROOT / run_name
    run = load_validated_run(run_dir)
    if run.manifest["experiment"]["run_name"] != run_name:
        raise ValueError("Selected run name conflicts with its saved manifest")
    summary = summarize_run(run)
    assert_no_image_payload(summary)
    write_nested_json(run_dir / "evaluation.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Select a run by name and print only the saved summary location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args(argv)
    try:
        evaluate_run(args.run_name)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    output = repository_root() / RUNS_RELATIVE_ROOT / args.run_name / "evaluation.json"
    print(f"Saved evaluation: {output}")


if __name__ == "__main__":
    main()

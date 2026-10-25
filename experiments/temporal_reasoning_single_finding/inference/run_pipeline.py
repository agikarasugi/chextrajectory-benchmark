"""Run the frozen single-finding inference stages in dependency order."""

import argparse
import json
from pathlib import Path
from typing import Any

from .derive_fixed_code import derive_fixed_code
from .experiment_config import (
    load_experiment,
    load_private_paths,
    require_api_key,
    resolve_benchmark_path,
    validate_worker_count,
)
from .model_client import create_client, require_exact_model
from .prepare_run import prepare_run
from .retry_failed import prepare_failed_retry
from .run_change_labels import run_change_labels
from .run_lc import run_predicted_lc, run_ref_lc
from .run_seq_dir import run_seq_dir
from ..validation.preflight import load_and_validate_benchmark, preflight
from ..validation.validate_run import validate_run


def run_pipeline(
    experiment_path: Path,
    paths_path: Path,
    *,
    workers: int = 1,
    progress: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Preflight, execute, derive, and coverage-check one resumable run."""
    validate_worker_count(workers)
    preflight_summary = preflight(
        experiment_path, paths_path,
    )
    experiment = load_experiment(experiment_path)
    private_paths = load_private_paths(paths_path)
    benchmark_path = resolve_benchmark_path(experiment)
    benchmark = load_and_validate_benchmark(benchmark_path)
    prepared = prepare_run(
        experiment,
        benchmark_path,
        benchmark.manifest,
        benchmark.items,
    )
    run_dir = prepared["run_dir"]
    client = create_client(
        str(experiment["base_url"]), require_api_key(),
        timeout=experiment.get("request_timeout_seconds", 300),
    )
    require_exact_model(client, str(experiment["model"]))

    retry_summary = prepare_failed_retry(run_dir, retry_failed=retry_failed)
    if retry_summary is not None and progress:
        print(json.dumps({"retry_failed": retry_summary}, sort_keys=True), flush=True)

    change_summary = run_change_labels(
        run_dir,
        experiment,
        private_paths["chexpert_plus_image_root"],
        workers=workers,
        progress=progress,
        client=client,
    )
    seq_dir_summary = run_seq_dir(
        run_dir,
        experiment,
        private_paths["chexpert_plus_image_root"],
        workers=workers,
        progress=progress,
        client=client,
    )
    ref_lc_summary = run_ref_lc(
        run_dir,
        experiment,
        workers=workers,
        progress=progress,
        client=client,
    )
    fixed_summary = derive_fixed_code(run_dir)
    pair_lc_summary = run_predicted_lc(
        run_dir,
        experiment,
        "PAIR",
        workers=workers,
        progress=progress,
        client=client,
    )
    seq_lc_summary = run_predicted_lc(
        run_dir,
        experiment,
        "SEQ",
        workers=workers,
        progress=progress,
        client=client,
    )
    coverage = validate_run(run_dir)
    return {
        "status": "passed",
        "run_dir": str(run_dir.resolve()),
        "preflight": preflight_summary,
        "stages": {
            "change_labels": change_summary,
            "seq_dir": seq_dir_summary,
            "ref_lc": ref_lc_summary,
            "fixed_code": fixed_summary,
            "pair_lc": pair_lc_summary,
            "seq_lc": seq_lc_summary,
        },
        "coverage": coverage,
        **({"retry_failed": retry_summary} if retry_summary is not None else {}),
    }


def parse_args() -> argparse.Namespace:
    """Parse the frozen pipeline runtime-only options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Back up and retry response_error records in a completed run",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete inference workflow and print a safe summary."""
    args = parse_args()
    summary = run_pipeline(
        args.experiment,
        args.paths,
        workers=args.workers,
        progress=args.progress,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

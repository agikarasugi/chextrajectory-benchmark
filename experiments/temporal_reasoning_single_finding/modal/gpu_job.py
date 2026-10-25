"""Validate and submit a GPU benchmark with an explicit resource profile."""

import argparse
from pathlib import Path

from ..artifact_io import read_json
from ..inference.experiment_config import validate_worker_count
from .config import load_job, source_digest


def main() -> None:
    """Validate local inputs before importing the GPU App or starting paid work."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--serving", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    _, experiment, profile = load_job(args.serving, args.experiment, read_json(args.profile))
    validate_worker_count(args.workers)
    source_digest()
    import modal

    from .app import PROFILE, app, main as launch

    if profile != PROFILE:
        raise ValueError("GPU profile changed after local validation")
    with modal.enable_output(), app.run(name=experiment["run_name"], detach=True):
        launch(args.serving, args.experiment, args.workers, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()

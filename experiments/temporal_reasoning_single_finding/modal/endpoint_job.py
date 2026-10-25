"""Validate locally and optionally submit detached CPU endpoint evaluation."""

import argparse

from .config import source_digest
from .endpoint_config import load_job


def submit(
    experiment: dict, profile: dict, code_sha256: str, workers: int, wait: bool,
    *, retry_failed: bool = False,
) -> None:
    """Start an asynchronous invocation whose lifetime is independent of local waiting."""
    import modal

    from .endpoint_app import PROFILE, app, run_benchmark

    if profile != PROFILE:
        raise ValueError("Endpoint profile changed after local validation")
    with modal.enable_output(), app.run(name=experiment["run_name"], detach=True):
        call = run_benchmark.spawn(
            experiment, profile, code_sha256, workers, retry_failed=retry_failed,
        )
        print(f"Submitted endpoint benchmark: {call.object_id}", flush=True)
        if wait:
            print(call.get())


def main() -> None:
    """Check configuration and source before importing or starting the Modal App."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True, help="Resource-profile JSON for this invocation")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate locally without Modal")
    parser.add_argument("--no-wait", action="store_true", help="Return after submission")
    args = parser.parse_args()
    experiment, profile = load_job(args.experiment, args.workers)
    code_sha256 = source_digest()
    if args.check:
        print("Endpoint configuration and source validated locally; no API calls made")
        return
    submit(
        experiment, profile, code_sha256, args.workers, not args.no_wait,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()

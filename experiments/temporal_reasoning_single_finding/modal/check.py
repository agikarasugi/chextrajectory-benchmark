"""Validate local Modal configuration without SDK authentication or cloud calls."""

import argparse
from pathlib import Path

from ..artifact_io import read_json
from .config import load_job, source_digest


def main() -> None:
    """Validate the two job JSONs and infrastructure profile offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    load_job(args.serving, args.experiment, read_json(args.profile))
    print("Configuration valid. Source SHA-256: " + source_digest())


if __name__ == "__main__":
    main()

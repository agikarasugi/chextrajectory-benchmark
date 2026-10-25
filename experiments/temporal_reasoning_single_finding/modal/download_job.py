"""Submit the CPU checkpoint downloader using the selected Modal profile."""

import argparse


def main() -> None:
    """Start a detached CPU download without transferring weights locally."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", required=True, help="Resource-profile JSON for this invocation")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--revision",
        help="Full commit SHA. Omit to select latest main for a new destination "
        "or reuse the recorded commit at an existing destination.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--secret", default="")
    args = parser.parse_args()
    from .config import validate_download_arguments

    validate_download_arguments(args.revision, args.model_path)
    import modal

    from .download_app import app, download

    with modal.enable_output(), app.run(detach=True):
        download(args.repo_id, args.revision, args.model_path, args.secret)


if __name__ == "__main__":
    main()

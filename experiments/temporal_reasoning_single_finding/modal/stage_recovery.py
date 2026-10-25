"""Hash unindexed images on a CPU container without downloading image payloads."""


def make_hasher():
    """Create a self-contained closure that Modal serializes by value."""
    def hash_paths(names: list[str], root: str = "/staged") -> dict:
        """Return sizes and hashes, or null for absent files, from a mounted Volume."""
        import hashlib
        from pathlib import Path, PurePosixPath

        base = Path(root).resolve()
        results = {}
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
                raise ValueError("Invalid recovery path")
            path = (base / name).resolve()
            if path == base or not path.is_relative_to(base):
                raise ValueError("Recovery path escapes Volume")
            try:
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
            except FileNotFoundError:
                results[name] = None
            else:
                results[name] = {"bytes": size, "sha256": digest.hexdigest()}
        return results

    return hash_paths


hash_paths = make_hasher()


def recover_images(volume: object, names: list[str]) -> dict:
    """Run bounded CPU checksum requests and return only file metadata locally."""
    if not names:
        return {}
    import modal

    app = modal.App("temporal-single-staging-recovery")
    check = app.function(
        image=modal.Image.debian_slim(python_version="3.12"),
        volumes={"/staged": volume.with_mount_options(read_only=True)},
        serialized=True,
        timeout=86400,
        retries=0,
    )(hash_paths)
    results = {}
    with app.run():
        for offset in range(0, len(names), 256):
            selected = names[offset : offset + 256]
            received = check.remote(selected)
            if not isinstance(received, dict) or set(received) != set(selected):
                raise ValueError("Incomplete remote checksum response")
            results.update(received)
            print(
                f"Remote image paths checked on Modal: {len(results)}/{len(names)}",
                flush=True,
            )
    return results

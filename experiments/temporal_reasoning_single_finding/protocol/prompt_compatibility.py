"""Recognize the fixture revision used before explanations became permanent."""

from pathlib import Path
from typing import Any

from ..artifact_io import sha256_file


LEGACY_FIXTURE_SHA256 = "cb197399779e1a8cd9e854f2c981720a5df21969b134390489141d0d1daa1bb5"
FIXTURE_KEY = "golden_prompts/sha256.json"


def current_fixture_checksums() -> dict[str, str]:
    """Fingerprint the small prompt fixture, never model or image payloads."""
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures" / FIXTURE_KEY
    return {FIXTURE_KEY: sha256_file(fixture)}


def normalize_prompt_fixture(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare known fixture revisions without changing the saved manifest.

    The legacy fixture covered answer-only snapshots even for explanation-enabled
    runs. Exact prepared inputs still determine whether inference can resume.
    """
    current = current_fixture_checksums()
    saved = manifest.get("prompt_fixture_checksums")
    if saved not in (current, {FIXTURE_KEY: LEGACY_FIXTURE_SHA256}):
        raise ValueError("Run prompt-fixture checksum is unsupported")
    return {**manifest, "prompt_fixture_checksums": current}

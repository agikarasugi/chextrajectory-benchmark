"""Reconstruct answer-only inputs exclusively for historical offline scoring."""

from collections.abc import Callable, Mapping
from typing import Any

from ..protocol.prompts import SHORT_EXPLANATION_PREFIX


def render_answer_only(renderer: Callable, *args: Any) -> list[dict[str, Any]]:
    """Restore the historical output instruction in freshly rendered messages."""
    messages = renderer(*args)
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            message["content"] = content.replace(
                SHORT_EXPLANATION_PREFIX, "Return exactly one JSON object "
            )
        else:
            for part in content:
                if part["type"] == "text":
                    part["text"] = part["text"].replace(
                        SHORT_EXPLANATION_PREFIX, "Return exactly one JSON object "
                    )
    return messages


def build_answer_only_registries(
    experiment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the old complete registry for offline consistency checks."""
    from ..inference.prepare_run import _build_registries

    return _build_registries(experiment, manifest, items, render_answer_only)

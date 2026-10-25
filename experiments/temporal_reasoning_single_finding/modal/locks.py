"""Persistent writer locks for single-input Modal FunctionCall submissions."""

from typing import Any


def acquire_call_lock(locks: Any, key: str, owner: str, message: str) -> None:
    """Acquire a key or resume its owning call after a container restart."""
    if not isinstance(owner, str) or not owner:
        raise RuntimeError("A Modal FunctionCall ID is required to acquire a writer lock")
    for _ in range(3):
        if locks.put(key, owner, skip_if_exists=True):
            return
        existing = locks.get(key)
        if existing == owner:
            # Modal reschedules the same input after worker loss. Its persistent
            # ownership survives even when the previous worker cannot clean up.
            return
        if existing is not None:
            raise RuntimeError(
                f"{message}; key={key!r}, owner={existing!r}, current_call={owner!r}. "
                "A different call cannot take over automatically. Confirm the owner "
                "has stopped before removing only this key, with other launchers stopped."
            )
        # The previous owner released between put and get. Compete atomically
        # for the now-empty key instead of assuming the missing key is ours.
    raise RuntimeError(f"{message}; ownership changed while acquiring {key!r}; retry")

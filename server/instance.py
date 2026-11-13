"""Pick the intended instance from a potentially stale JarvisLabs list."""

from __future__ import annotations

import os


def select_instance(instances: list[dict], fallback: dict | None) -> dict | None:
    """Honor JARVIS_MACHINE_ID when set; otherwise keep the adapter's choice."""
    wanted = os.environ.get("JARVIS_MACHINE_ID", "")
    if not wanted:
        return fallback
    return next(
        (item for item in instances if str(item.get("machine_id")) == wanted),
        fallback,
    )

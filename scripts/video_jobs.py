"""Video job parsing, captioning and quota helpers for the Telegram bot."""

from __future__ import annotations

import re
import time

VIDEO_POLL_SECONDS = 1800
_video_usage: dict[int, tuple[str, int]] = {}


def allow_video(owner_id: int, limit: int) -> bool:
    """In-process daily video quota; restarts reset it."""
    if limit <= 0:
        return True
    today = time.strftime("%Y-%m-%d")
    day, count = _video_usage.get(owner_id, (today, 0))
    if day != today:
        count = 0
    if count >= limit:
        return False
    _video_usage[owner_id] = (today, count + 1)
    return True


def extract_video_seconds(logs: list[str] | None) -> float | None:
    match = re.search(r"generated in ([0-9.]+)s", "\n".join(logs or []))
    return float(match.group(1)) if match else None


def format_video_caption(job: dict, total_seconds: float | None = None) -> str:
    prompt = (job.get("prompt") or "")[:500]
    metadata = job.get("metadata") or {}
    model = metadata.get("model_size", "1.3B")
    duration = metadata.get("duration", 3)
    gpu = extract_video_seconds(job.get("logs", []))
    timing = []
    if gpu is not None:
        timing.append(f"{gpu:.1f}s GPU")
    if total_seconds is not None:
        timing.append(f"{total_seconds:.0f}s total")
    suffix = f"\n⏱ {' · '.join(timing)}" if timing else ""
    return f"{prompt}\nWan 2.1 {model} · {duration}s · 480p{suffix}"

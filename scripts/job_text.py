"""Clean user-facing labels for image jobs.

Character jobs show the character name instead of the composed appearance
prompt; prompt-only jobs keep their prompt.
"""

from __future__ import annotations


def character_name(characters: list[dict], job: dict) -> str:
    if not job.get("character_id"):
        return ""
    for character in characters:
        if character.get("id") == job["character_id"]:
            return character.get("name") or "Character"
    return "Character"


def _timing(job: dict) -> str:
    seconds = job.get("generation_seconds")
    if seconds is None:
        start = job.get("started_at") or job.get("created_at") or 0
        end = job.get("updated_at") or 0
        seconds = round(end - start, 1) if end and start else None
    return f"\n⏱ {seconds:.1f}s" if seconds is not None else ""


def image_caption(job: dict, characters: list[dict], limit: int = 900) -> str:
    timing = _timing(job)
    name = character_name(characters, job)
    if name:
        return f"{name}{timing}"
    prompt = (job.get("prompt") or "")[:limit]
    return f"{prompt}{timing}"


def job_label(job: dict, characters: list[dict], limit: int = 60) -> str:
    return character_name(characters, job) or (job.get("prompt") or "")[:limit]

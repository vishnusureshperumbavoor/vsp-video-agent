"""Load and plan a micro-story for the Level 1 video renderer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_SCENES = 8
MAX_TEXT = 800
IDENTITY = "Keep the same woman's face, hair, skin tone and identity exactly."
STYLE = "vertical 9:16 Instagram story frame, cinematic, fully clothed"


class StoryError(ValueError):
    """The story file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Scene:
    prompt: str
    narration: str


@dataclass(frozen=True, slots=True)
class Story:
    title: str
    scenes: list[Scene]


def _text(value, field: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise StoryError(f"{field} is required")
    if len(text) > MAX_TEXT:
        raise StoryError(f"{field} must be at most {MAX_TEXT} characters")
    return text


def load_story(path: str | Path) -> Story:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoryError(f"cannot read story: {exc}") from exc
    title = _text(data.get("title", "Untitled"), "title")
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not 1 <= len(raw_scenes) <= MAX_SCENES:
        raise StoryError(f"scenes must be a list of 1 to {MAX_SCENES} entries")
    scenes = [
        Scene(
            prompt=_text(item.get("prompt"), f"scenes[{index}].prompt"),
            narration=_text(item.get("narration"), f"scenes[{index}].narration"),
        )
        for index, item in enumerate(raw_scenes)
        if isinstance(item, dict)
    ]
    if len(scenes) != len(raw_scenes):
        raise StoryError("every scene must be an object with prompt and narration")
    return Story(title=title, scenes=scenes)


def scene_prompt(scene: Scene) -> str:
    prompt = scene.prompt.rstrip(" .,;")
    return f"{IDENTITY} Change the scene to: {prompt}, {STYLE}."


def estimate_seconds(scene: Scene, words_per_second: float = 2.5, gap: float = 0.7) -> float:
    words = max(1, len(scene.narration.split()))
    return max(2.0, words / words_per_second + gap)


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(scenes: list[Scene], durations: list[float]) -> str:
    if len(scenes) != len(durations):
        raise StoryError("scenes and durations must have the same length")
    lines: list[str] = []
    cursor = 0.0
    for index, (scene, duration) in enumerate(zip(scenes, durations, strict=True), 1):
        start = cursor
        cursor += duration
        lines.extend(
            [
                str(index),
                f"{format_timestamp(start)} --> {format_timestamp(cursor)}",
                scene.narration,
                "",
            ]
        )
    return "\n".join(lines)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "story"

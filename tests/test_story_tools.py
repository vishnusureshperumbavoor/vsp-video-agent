"""Pure helpers for the micro-story Level 1 tooling.

These tests never call the GPU, Piper, MusicGen or FFmpeg; the renderer itself
is exercised manually after a `--dry-run`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from story.model import (  # noqa: E402
    Scene,
    StoryError,
    build_srt,
    estimate_seconds,
    format_timestamp,
    load_story,
    scene_prompt,
)


def _story_file(tmp_path: Path, scenes: list[dict]) -> Path:
    path = tmp_path / "story.json"
    path.write_text(json.dumps({"title": "The Second Cup", "scenes": scenes}), encoding="utf-8")
    return path


def test_load_story_reads_scenes(tmp_path: Path) -> None:
    path = _story_file(
        tmp_path,
        [
            {"prompt": "a cafe", "narration": "Gabriela ordered one coffee."},
            {"prompt": "a window", "narration": "She stayed for a second cup."},
        ],
    )

    story = load_story(path)

    assert story.title == "The Second Cup"
    assert len(story.scenes) == 2
    assert story.scenes[0].prompt == "a cafe"


def test_load_story_rejects_an_empty_scene_list(tmp_path: Path) -> None:
    path = _story_file(tmp_path, [])

    with pytest.raises(StoryError):
        load_story(path)


def test_load_story_rejects_missing_narration(tmp_path: Path) -> None:
    path = _story_file(tmp_path, [{"prompt": "a cafe"}])

    with pytest.raises(StoryError):
        load_story(path)


def test_scene_prompt_keeps_identity_and_vertical_style() -> None:
    prompt = scene_prompt(Scene("a cafe", "text"))

    assert "same woman" in prompt.lower()
    assert "9:16" in prompt
    assert "fully clothed" in prompt


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(61.5) == "00:01:01,500"


def test_build_srt_uses_cumulative_durations() -> None:
    scenes = [Scene("a", "one"), Scene("b", "two")]

    srt = build_srt(scenes, [2.5, 3.0])

    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "00:00:02,500 --> 00:00:05,500" in srt
    assert "one" in srt and "two" in srt


def test_estimate_seconds_is_positive() -> None:
    assert estimate_seconds(Scene("a", "a short line")) >= 2.0


def test_render_cli_parses_its_flags() -> None:
    from story_video import parse_args

    args = parse_args(
        [
            "render",
            "--script", "story.json",
            "--reference", "face.png",
            "--no-music",
            "--steps", "8",
        ]
    )

    assert args.command == "render"
    assert args.script == "story.json"
    assert args.no_music is True
    assert args.steps == 8


def test_find_piper_falls_back_to_the_interpreter_directory(tmp_path: Path, monkeypatch) -> None:
    from story import tts

    monkeypatch.setattr(tts.shutil, "which", lambda name: None)
    fake = tmp_path / "piper"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")

    assert tts.find_piper(tmp_path / "python") == str(fake)


def test_find_piper_returns_none_when_absent(tmp_path: Path, monkeypatch) -> None:
    from story import tts

    monkeypatch.setattr(tts.shutil, "which", lambda name: None)

    assert tts.find_piper(tmp_path / "python") is None

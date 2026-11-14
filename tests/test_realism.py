"""Pure tests for the photorealism prompt helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import realism  # noqa: E402
from minimax_h3.workflow import compose_prompt  # noqa: E402


def test_dialogue_wraps_the_line_in_a_tag() -> None:
    assert realism.dialogue("Hello there") == "<d>[English] Hello there</d>"
    assert realism.dialogue("Bonjour", "French") == "<d>[French] Bonjour</d>"


def test_dialogue_strips_quotes_and_rejects_empty() -> None:
    assert realism.dialogue('"Hi"') == "<d>[English] Hi</d>"
    with pytest.raises(ValueError):
        realism.dialogue("   ")
    with pytest.raises(ValueError):
        realism.dialogue("<d>[English] already tagged</d>")


def test_enhance_adds_the_still_suffix_once() -> None:
    once = realism.enhance("a woman at a table", "still")
    twice = realism.enhance(once, "still")

    assert "raw candid photograph" in once
    assert once == twice


def test_enhance_uses_a_different_suffix_for_video() -> None:
    text = realism.enhance("a woman at a table", "video")

    assert "shot on 35mm film" in text
    assert "raw candid photograph" not in text


def test_audit_flags_plastic_words_and_missing_cues() -> None:
    warnings = realism.audit("beautiful instagram portrait, smooth skin")

    assert any("instagram" in warning for warning in warnings)
    assert any("light/lens/texture" in warning for warning in warnings)


def test_audit_is_quiet_for_a_good_prompt() -> None:
    prompt = (
        "raw candid photograph, 85mm lens, natural skin texture, "
        "soft window light, fully clothed"
    )

    assert realism.audit(prompt) == []


def test_compose_prompt_combines_scene_dialogue_and_film_cues() -> None:
    prompt = compose_prompt(
        "a woman at a kitchen table",
        dialogue_line="Call me back",
        raw=True,
    )

    assert "<d>[English] Call me back</d>" in prompt
    assert "shot on 35mm film" in prompt


def test_compose_prompt_leaves_plain_scenes_alone() -> None:
    assert compose_prompt("  a saxophonist  ") == "a saxophonist"
    with pytest.raises(ValueError):
        compose_prompt("   ")


def test_dialogue_switches_steps_and_disables_turbo() -> None:
    from minimax_h3.workflow import resolve_settings

    assert resolve_settings("") == (8, 1.0)
    assert resolve_settings("a line") == (20, 0.0)


def test_explicit_settings_win_over_the_dialogue_default() -> None:
    from minimax_h3.workflow import resolve_settings

    assert resolve_settings("a line", steps=12, lora_strength=0.5) == (12, 0.5)

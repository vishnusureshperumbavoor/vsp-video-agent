"""Pure parsing tests for the Telegram bot.

The bot itself is a network loop, so the testable layer is command parsing and
user-facing formatting. Network calls stay out of the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from telegram_bot import BotError, describe_characters, parse_character, parse_command  # noqa: E402


def test_plain_text_is_not_a_command() -> None:
    assert parse_command("a rooftop at sunset") == ("", "a rooftop at sunset")


def test_command_suffix_is_removed() -> None:
    assert parse_command("/imagine@AvaBot a rooftop") == ("imagine", "a rooftop")


def test_character_uses_the_first_pipe_only() -> None:
    assert parse_character("Ava | silver hair | green eyes") == (
        "Ava",
        "silver hair | green eyes",
    )


def test_character_requires_a_pipe() -> None:
    with pytest.raises(BotError):
        parse_character("Ava silver hair")


def test_character_requires_both_halves() -> None:
    with pytest.raises(BotError):
        parse_character("Ava |   ")


def test_describe_characters_explains_how_to_start() -> None:
    assert "/character" in describe_characters([])


def test_describe_characters_lists_ids() -> None:
    text = describe_characters([{"id": "abc123", "name": "Ava"}])

    assert "Ava" in text
    assert "abc123" in text

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

from telegram_bot import (  # noqa: E402
    BotError,
    describe_characters,
    is_allowed,
    parse_character,
    parse_command,
)


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


def test_empty_allowlist_accepts_any_chat() -> None:
    assert is_allowed(111, 111, 0) is True


def test_allowlist_accepts_the_configured_chat() -> None:
    assert is_allowed(855227717, 999, 855227717) is True


def test_allowlist_accepts_the_configured_user_in_a_group() -> None:
    assert is_allowed(-100123, 855227717, 855227717) is True


def test_allowlist_rejects_other_chats() -> None:
    assert is_allowed(111, 222, 855227717) is False

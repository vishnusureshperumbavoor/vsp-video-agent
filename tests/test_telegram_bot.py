"""Pure parsing tests for the Telegram bot.

The bot itself is a network loop, so the testable layer is command parsing and
user-facing formatting. Network calls stay out of the suite.
"""

from __future__ import annotations

import io
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import telegram_bot  # noqa: E402
from telegram_bot import (  # noqa: E402
    BotError,
    character_commands_allowed,
    describe_characters,
    format_caption,
    http_json,
    is_allowed,
    parse_character,
    parse_command,
)
from video_jobs import extract_video_seconds, format_video_caption  # noqa: E402


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


def test_caption_includes_generation_time() -> None:
    text = format_caption({"prompt": "a cat", "generation_seconds": 12.3})

    assert "a cat" in text
    assert "12.3s" in text


def test_caption_falls_back_to_the_job_timestamps() -> None:
    text = format_caption(
        {"prompt": "a cat", "started_at": 100.0, "updated_at": 104.5}
    )

    assert "4.5s" in text


def test_empty_allowlist_accepts_any_chat() -> None:
    assert is_allowed(111, 111, 0) is True


def test_allowlist_accepts_the_configured_chat() -> None:
    assert is_allowed(855227717, 999, 855227717) is True


def test_allowlist_accepts_the_configured_user_in_a_group() -> None:
    assert is_allowed(-100123, 855227717, 855227717) is True


def test_allowlist_rejects_other_chats() -> None:
    assert is_allowed(111, 222, 855227717) is False


def test_character_commands_are_private_when_an_owner_is_set() -> None:
    assert character_commands_allowed(855227717, 855227717) is True
    assert character_commands_allowed(111, 855227717) is False


def test_character_commands_are_open_without_an_owner() -> None:
    assert character_commands_allowed(111, 0) is True


def test_backend_detail_is_shown_without_http_noise(monkeypatch) -> None:
    def boom(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"detail":"create and select a character first"}'),
        )

    monkeypatch.setattr(telegram_bot.urllib.request, "urlopen", boom)

    with pytest.raises(BotError, match="create and select a character first"):
        http_json("Backend", "http://backend.test/api/images", {})


def test_network_timeout_is_wrapped_as_a_bot_error(monkeypatch) -> None:
    def boom(request, timeout=0):
        raise TimeoutError("timed out")

    monkeypatch.setattr(telegram_bot.urllib.request, "urlopen", boom)

    with pytest.raises(BotError, match="unreachable"):
        http_json("Telegram", "http://telegram.test/getUpdates", {})


def test_extract_video_seconds_reads_the_success_marker() -> None:
    logs = ["Loading...", "=== JOB_SUCCESS: out.mp4 generated in 58.7s (273000 bytes) ==="]

    assert extract_video_seconds(logs) == 58.7


def test_extract_video_seconds_handles_missing_marker() -> None:
    assert extract_video_seconds(["still running"]) is None


def test_video_caption_includes_model_and_timing() -> None:
    job = {
        "prompt": "a beach at sunset",
        "logs": ["=== JOB_SUCCESS: out.mp4 generated in 58.7s (273000 bytes) ==="],
        "metadata": {"model_size": "1.3B", "duration": 3},
    }

    caption = format_video_caption(job, total_seconds=80)

    assert "a beach at sunset" in caption
    assert "Wan 2.1 1.3B" in caption
    assert "58.7s GPU" in caption
    assert "80s total" in caption

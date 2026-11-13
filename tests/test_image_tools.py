"""Pure helpers for the image operator tools.

No network and no GPU: these pin down the formatting and the benchmark matrix,
so the CLIs stay predictable when their HTTP paths change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from image_admin import format_job, format_seconds  # noqa: E402
from image_bench import build_configs, parse_list  # noqa: E402
from reference_edit import build_edit_payload  # noqa: E402


def test_parse_list_accepts_spaces() -> None:
    assert parse_list("1024, 768,512") == [1024, 768, 512]


def test_parse_list_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_list("1024,0")


def test_build_configs_keeps_size_major_order() -> None:
    assert build_configs([1024, 768], [50, 30]) == [
        (1024, 50),
        (1024, 30),
        (768, 50),
        (768, 30),
    ]


def test_format_seconds_handles_missing_values() -> None:
    assert format_seconds(63.4) == "63.4s"
    assert format_seconds(None) == "—"


def test_format_job_includes_status_prompt_and_time() -> None:
    line = format_job(
        {
            "status": "completed",
            "prompt": "a cat on a windowsill",
            "generation_seconds": 62.8,
        }
    )

    assert "completed" in line
    assert "a cat on a windowsill" in line
    assert "62.8s" in line


def test_reference_payload_carries_the_edit_settings() -> None:
    payload = build_edit_payload(
        "keep the same woman",
        "base64-reference",
        size=768,
        steps=4,
        guidance=1.0,
        seed=42,
    )

    assert payload["reference_b64"] == "base64-reference"
    assert payload["width"] == 768
    assert payload["height"] == 768
    assert payload["steps"] == 4
    assert payload["guidance_scale"] == 1.0


def test_reference_payload_can_be_vertical() -> None:
    payload = build_edit_payload("keep the same woman", "ref", size=768, height=1344)

    assert payload["width"] == 768
    assert payload["height"] == 1344

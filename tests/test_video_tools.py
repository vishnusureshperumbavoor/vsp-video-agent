"""Pure helpers for the video operator CLI and job formatting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from video_admin import (  # noqa: E402
    build_i2v_payload,
    build_t2v_payload,
    format_job,
    format_seconds,
    parse_args,
)


def test_t2v_payload_defaults_to_1_3b() -> None:
    payload = build_t2v_payload("a beach at sunset")

    assert payload["model_size"] == "1.3B"
    assert payload["duration"] == 3
    assert payload["steps"] == 25
    assert payload["aspect_ratio"] == "16:9"
    assert "seed" not in payload


def test_t2v_payload_keeps_an_explicit_seed() -> None:
    payload = build_t2v_payload("x", seed=7)

    assert payload["seed"] == 7


def test_i2v_payload_defaults_to_vertical() -> None:
    payload = build_i2v_payload("she smiles")

    assert payload["duration"] == 5
    assert payload["aspect_ratio"] == "9:16"


def test_t2v_cli_accepts_model_and_no_wait() -> None:
    args = parse_args(["t2v", "a beach", "--model", "14B", "--no-wait"])

    assert args.command == "t2v"
    assert args.model == "14B"
    assert args.no_wait is True


def test_i2v_cli_accepts_an_image_and_prompt() -> None:
    args = parse_args(["i2v", "face.png", "she smiles"])

    assert args.command == "i2v"
    assert args.image == "face.png"
    assert args.prompt == "she smiles"


def test_format_seconds_handles_missing_values() -> None:
    assert format_seconds(58.7) == "58.7s"
    assert format_seconds(None) == "—"


def test_format_job_reads_the_success_marker() -> None:
    job = {
        "status": "completed",
        "prompt": "a beach at sunset",
        "logs": ["=== JOB_SUCCESS: out.mp4 generated in 58.7s (273000 bytes) ==="],
    }

    line = format_job(job)

    assert "completed" in line
    assert "a beach at sunset" in line
    assert "58.7s" in line

"""Tests for the synthetic character bootstrap tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from character.curate import cosine  # noqa: E402
from character.scenes import Scene, default_scenes, load_scenes  # noqa: E402
from character.train import TrainSettings, build_command  # noqa: E402
from character_cli import parse_args  # noqa: E402


def test_default_sheet_has_30_unique_scenes() -> None:
    scenes = default_scenes()

    assert len(scenes) == 30
    assert len({scene.id for scene in scenes}) == 30
    assert all("Keep the same woman" in scene.prompt for scene in scenes)


def test_caption_starts_with_the_trigger() -> None:
    scene = Scene("01", "close-up, white shirt", "prompt")

    assert scene.caption("gbrla") == "gbrla, close-up, white shirt"


def test_load_scenes_reads_a_custom_json(tmp_path: Path) -> None:
    path = tmp_path / "scenes.json"
    path.write_text(
        json.dumps([{"id": "a", "description": "d", "prompt": "p"}]),
        encoding="utf-8",
    )

    assert load_scenes(str(path))[0].id == "a"


def test_cosine_is_one_for_identical_vectors() -> None:
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_train_command_uses_flux2_klein_defaults() -> None:
    command = build_command(TrainSettings())

    assert "train_dreambooth_lora_flux2_klein.py" in command
    assert any("FLUX.2-klein-4B" in part for part in command)
    assert "--max_train_steps=800" in command
    assert "--do_fp8_training" not in command


def test_train_cli_accepts_dry_run() -> None:
    args = parse_args(["train", "--dry-run", "--steps", "500"])

    assert args.command == "train"
    assert args.dry_run is True
    assert args.steps == 500

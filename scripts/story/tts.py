"""Piper TTS for the micro-story renderer (CPU, no GPU needed)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

from huggingface_hub import hf_hub_download

from story.model import StoryError

VOICE_REPO = "rhasspy/piper-voices"


def _voice_prefix(voice: str) -> str:
    parts = voice.split("-")
    if len(parts) != 3:
        raise StoryError("voice must look like en_US-amy-medium")
    language, name, quality = parts
    region = language.split("_")
    if len(region) != 2:
        raise StoryError("voice language must look like en_US")
    return f"{region[0]}/{language}/{name}/{quality}/{voice}"


def voice_paths(voice: str, cache_dir: str | Path) -> tuple[str, str]:
    prefix = _voice_prefix(voice)
    model = hf_hub_download(
        repo_id=VOICE_REPO,
        filename=f"{prefix}.onnx",
        cache_dir=str(cache_dir),
    )
    config = hf_hub_download(
        repo_id=VOICE_REPO,
        filename=f"{prefix}.onnx.json",
        cache_dir=str(cache_dir),
    )
    return model, config


def synthesize(
    text: str,
    out_path: str | Path,
    voice: str = "en_US-amy-medium",
    cache_dir: str | Path = "data/voices",
) -> Path:
    piper = shutil.which("piper")
    if not piper:
        raise StoryError(
            "piper is not installed; run `pip install -r requirements-story.txt`"
        )
    model, config = voice_paths(voice, cache_dir)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [piper, "--model", model, "--config", config, "--output_file", str(target)],
        input=text.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise StoryError(f"piper failed: {result.stderr.decode('utf-8', 'ignore')[-400:]}")
    return target


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())

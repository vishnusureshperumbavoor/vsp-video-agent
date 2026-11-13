"""FFmpeg assembly for the Level 1 micro-story renderer."""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

from story.model import StoryError

WIDTH = 1080
HEIGHT = 1920
FPS = 30
RATE = 22050


def _ffmpeg(args: list[str], label: str) -> None:
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-y", *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "ignore")[-600:]
        raise StoryError(f"{label} failed: {tail}")


def make_segment(image: str | Path, duration: float, out: str | Path) -> Path:
    frames = max(1, int(round(duration * FPS)))
    filters = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(zoom+0.0006,1.06)':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )
    target = Path(out)
    _ffmpeg(
        [
            "-i", str(image),
            "-vf", filters,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-an",
            str(target),
        ],
        "video segment",
    )
    return target


def concat_segments(segments: list[Path], out: str | Path) -> Path:
    target = Path(out)
    list_path = target.parent / "segments.txt"
    list_path.write_text(
        "".join(f"file '{segment.resolve()}'\n" for segment in segments),
        encoding="utf-8",
    )
    _ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(target)],
        "segment concat",
    )
    return target


def write_narration(wavs: list[Path], gap: float, out: str | Path) -> Path:
    target = Path(out)
    with wave.open(str(wavs[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for path in wavs[1:]:
        with wave.open(str(path), "rb") as handle:
            frames.append(handle.readframes(handle.getnframes()))
    silence = b"\x00" * int(params.framerate * gap) * params.sampwidth * params.nchannels
    with wave.open(str(target), "wb") as handle:
        handle.setparams(params)
        for chunk in frames:
            handle.writeframes(chunk)
            handle.writeframes(silence)
    return target


def ambient_bed(duration: float, out: str | Path) -> Path:
    try:
        import numpy as np
    except ImportError as exc:
        raise StoryError("numpy is required for the ambient bed") from exc
    target = Path(out)
    count = max(1, int(duration * RATE))
    time_axis = np.arange(count) / RATE
    chord = (
        0.05 * np.sin(2 * np.pi * 220.0 * time_axis)
        + 0.04 * np.sin(2 * np.pi * 277.18 * time_axis)
        + 0.04 * np.sin(2 * np.pi * 329.63 * time_axis)
    )
    fade = np.minimum(1.0, time_axis / 2.0) * np.minimum(1.0, (duration - time_axis) / 2.0)
    samples = (chord * np.clip(fade, 0, 1) * 32767).astype(np.int16)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.tobytes())
    return target


def _srt_filter(path: str | Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    style = (
        "FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=70"
    )
    return f"subtitles={escaped}:force_style='{style}'"


def mix(
    video: str | Path,
    narration: str | Path,
    music: str | Path | None,
    srt: str | Path | None,
    out: str | Path,
    burn_subtitles: bool,
) -> Path:
    args = ["-i", str(video), "-i", str(narration)]
    if music:
        args += ["-i", str(music)]
        args += [
            "-filter_complex",
            "[2:a]volume=0.16[m];[1:a][m]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "0:v", "-map", "[a]",
        ]
    else:
        args += ["-map", "0:v", "-map", "1:a"]
    if burn_subtitles and srt:
        args += ["-vf", _srt_filter(srt), "-c:v", "libx264", "-crf", "18"]
    else:
        args += ["-c:v", "copy"]
    args += [
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    _ffmpeg(args, "audio mix")
    return Path(out)


def poster(video: str | Path, out: str | Path) -> Path:
    target = Path(out)
    _ffmpeg(
        ["-ss", "0.2", "-i", str(video), "-vframes", "1", "-q:v", "2", str(target)],
        "poster",
    )
    return target

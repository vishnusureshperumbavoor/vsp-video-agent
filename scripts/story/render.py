"""Level 1 micro-story renderer: stills, narration, music, FFmpeg."""

from __future__ import annotations

from pathlib import Path

from gpu_client import request_image
from reference_edit import build_edit_payload, encode_reference

from story.assemble import (
    ambient_bed,
    concat_segments,
    make_segment,
    mix,
    poster,
    write_narration,
)
from story.model import Story, build_srt, scene_prompt
from story.tts import synthesize, wav_duration

VOICE_CACHE = Path("data/voices")


def generate_stills(
    story: Story,
    reference: str,
    out_dir: Path,
    url: str,
    token: str,
    width: int = 768,
    height: int = 1344,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
) -> list[Path]:
    encoded = encode_reference(reference)
    paths: list[Path] = []
    for index, scene in enumerate(story.scenes):
        payload = build_edit_payload(
            scene_prompt(scene),
            encoded,
            width,
            steps,
            guidance,
            seed + index,
            height,
        )
        elapsed, raw = request_image(url, token, "/edit", payload, 1800.0)
        target = out_dir / f"scene_{index + 1:02d}.png"
        target.write_bytes(raw)
        print(f"still {index + 1}/{len(story.scenes)}: {elapsed:.1f}s -> {target}")
        paths.append(target)
    return paths


def render_story(
    story: Story,
    reference: str,
    out_path: Path,
    work_dir: Path,
    url: str,
    token: str,
    voice: str = "en_US-amy-medium",
    width: int = 768,
    height: int = 1344,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
    music: bool = True,
    subtitles: bool = True,
) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    stills = generate_stills(
        story, reference, work_dir, url, token, width, height, steps, guidance, seed
    )

    wavs: list[Path] = []
    durations: list[float] = []
    for index, scene in enumerate(story.scenes):
        wav = synthesize(
            scene.narration,
            work_dir / f"scene_{index + 1:02d}.wav",
            voice,
            VOICE_CACHE,
        )
        wavs.append(wav)
        durations.append(wav_duration(wav) + 0.7)
        print(f"voice {index + 1}/{len(story.scenes)}: {durations[-1]:.1f}s")

    narration = write_narration(wavs, 0.7, work_dir / "narration.wav")
    srt_path = work_dir / "captions.srt"
    srt_path.write_text(build_srt(story.scenes, durations), encoding="utf-8")

    segments: list[Path] = []
    for index, (still, duration) in enumerate(zip(stills, durations, strict=True), 1):
        segment = make_segment(still, duration, work_dir / f"segment_{index:02d}.mp4")
        segments.append(segment)

    silent = concat_segments(segments, work_dir / "silent.mp4")
    total = sum(durations)
    music_path = ambient_bed(total, work_dir / "ambient.wav") if music else None
    final = mix(silent, narration, music_path, srt_path, out_path, subtitles)
    poster(final, out_path.with_name(f"{out_path.stem}_poster.jpg"))
    return final, srt_path

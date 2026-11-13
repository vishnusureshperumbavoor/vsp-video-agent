#!/usr/bin/env python3
"""Micro-story video CLI (Level 1).

Turns a JSON storyboard and one character reference into a 9:16 MP4 with
narration, an ambient bed, and burned-in captions.

    python scripts/story_video.py plan --script story.json
    python scripts/story_video.py render --script story.json --reference face.png \
        --out public/videos/story.mp4

`plan` and `render --dry-run` never call the GPU, TTS or FFmpeg.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli_common import add_media_options, add_options  # noqa: E402
from gpu_client import GPUError, resolve_service  # noqa: E402
from story.model import StoryError, estimate_seconds, load_story  # noqa: E402
from telegram_bot import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def print_plan(story) -> None:
    print(story.title)
    total = 0.0
    for index, scene in enumerate(story.scenes, 1):
        seconds = estimate_seconds(scene)
        total += seconds
        print(f"{index}. ~{seconds:.1f}s  {scene.prompt}")
        print(f"   {scene.narration}")
    print(f"estimated total: ~{total:.1f}s")


def cmd_plan(args) -> int:
    try:
        story = load_story(args.script)
    except StoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_plan(story)
    return 0


def cmd_render(args) -> int:
    try:
        story = load_story(args.script)
    except StoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print_plan(story)
        return 0
    if not Path(args.reference).exists():
        print(f"error: reference not found: {args.reference}", file=sys.stderr)
        return 2
    load_dotenv(str(ROOT / ".env"))
    try:
        url, token = resolve_service(None, None)
    except GPUError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out or ROOT / "public" / "videos" / f"story_{stamp}.mp4")
    work_dir = Path(args.work_dir or ROOT / "public" / "stories" / f"story_{stamp}")
    try:
        from story.render import render_story

        final, srt = render_story(
            story,
            args.reference,
            out_path,
            work_dir,
            url,
            token,
            args.voice,
            args.width,
            args.height,
            args.steps,
            args.guidance,
            args.seed,
            not args.no_music,
            not args.no_subtitles,
        )
    except (StoryError, GPUError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"video: {final}")
    print(f"captions: {srt}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-story video (Level 1)")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Validate and estimate a story")
    plan.add_argument("--script", required=True)
    plan.set_defaults(func=cmd_plan)

    render = sub.add_parser("render", help="Render stills, narration, music and video")
    add_options(
        render,
        [
            ("--script", {"required": True}),
            ("--reference", {"required": True}),
            ("--out"),
            ("--work-dir"),
            ("--voice", {"default": "en_US-amy-medium"}),
            ("--no-music", {"action": "store_true"}),
            ("--no-subtitles", {"action": "store_true"}),
        ],
    )
    add_media_options(render)
    render.set_defaults(func=cmd_render)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

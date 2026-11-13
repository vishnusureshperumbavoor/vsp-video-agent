#!/usr/bin/env python3
"""Synthetic character bootstrap CLI.

    python scripts/character_cli.py sheet --reference face.png --out sheet/
    python scripts/character_cli.py curate --input sheet/ --reference face.png --out dataset/
    python scripts/character_cli.py caption --manifest sheet/manifest.json
    python scripts/character_cli.py train --dataset dataset/ --dry-run

Nothing here starts the multi-hour training run unless you pass its generated
script to the GPU yourself.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from character.curate import curate  # noqa: E402
from character.scenes import load_scenes  # noqa: E402
from character.sheet import generate_sheet  # noqa: E402
from character.train import TrainSettings, build_command, write_script  # noqa: E402
from cli_common import add_options  # noqa: E402
from gpu_client import GPUError, add_service_args, resolve_service  # noqa: E402
from telegram_bot import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def cmd_sheet(args) -> int:
    scenes = load_scenes(args.scenes)
    if args.dry_run:
        for scene in scenes:
            print(f"{scene.id}: {scene.caption(args.trigger)}")
        return 0
    try:
        url, token = resolve_service(args.url, args.token)
    except GPUError as exc:
        print(exc, file=sys.stderr)
        return 2
    entries = generate_sheet(
        args.reference,
        scenes,
        args.out,
        url,
        token,
        args.size,
        args.steps,
        args.guidance,
        args.seed,
        args.timeout,
        args.trigger,
        args.resume,
    )
    print(f"{len(entries)} scenes -> {Path(args.out) / 'manifest.json'}")
    return 0


def cmd_curate(args) -> int:
    report = curate(args.input, args.reference, args.out, args.threshold, args.dedupe)
    kept = sum(1 for item in report if item["kept"])
    print(f"{kept} kept -> {Path(args.out) / 'report.json'}")
    return 0


def cmd_caption(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else Path(args.manifest).parent
    for entry in manifest.get("entries", []):
        caption = entry.get("caption")
        if not caption:
            continue
        stem = Path(entry["file"]).stem
        (out_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    print(f"captions written to {out_dir}")
    return 0


def cmd_train(args) -> int:
    settings = TrainSettings(
        dataset=args.dataset,
        output=args.output,
        trigger=args.trigger,
        steps=args.steps,
        rank=args.rank,
        resolution=args.resolution,
        batch=args.batch,
        grad_accum=args.grad_accum,
        seed=args.seed,
    )
    command = build_command(settings)
    if args.write_script:
        path = write_script(settings, args.write_script)
        print(f"wrote {path}; upload it to /home/app and run it there")
        return 0
    print(shlex.join(command))
    print("\nRun this on the GPU box after installing the training deps and trainer.")
    return 0


def add_identity_options(parser: argparse.ArgumentParser, seed_default: int) -> None:
    parser.add_argument("--trigger", default="gbrla")
    parser.add_argument("--seed", type=int, default=seed_default)
    parser.add_argument("--dry-run", action="store_true")


def add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic character bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)

    sheet = sub.add_parser("sheet", help="Generate reference edits with FLUX.2 klein")
    add_options(
        sheet,
        [
            ("--reference", {"required": True}),
            ("--out", {"default": "sheet"}),
            ("--scenes", {"help": "JSON scene list; defaults to the built-in 30"}),
        ],
    )
    add_generation_options(sheet)
    add_identity_options(sheet, 42)
    sheet.add_argument("--resume", action="store_true")
    add_service_args(sheet, 1800.0)
    sheet.set_defaults(func=cmd_sheet)

    curation = sub.add_parser("curate", help="Face-embedding curation")
    add_options(
        curation,
        [
            ("--input", {"required": True}),
            ("--reference", {"required": True}),
            ("--out", {"default": "dataset"}),
            ("--threshold", {"type": float, "default": 0.45}),
            ("--dedupe", {"type": float, "default": 0.93}),
        ],
    )
    curation.set_defaults(func=cmd_curate)

    caption = sub.add_parser("caption", help="Write captions from a sheet manifest")
    add_options(
        caption,
        [
            ("--manifest", {"required": True}),
            ("--out", {}),
        ],
    )
    caption.set_defaults(func=cmd_caption)

    train = sub.add_parser("train", help="Build the FLUX.2 klein LoRA command")
    add_options(
        train,
        [
            ("--dataset", {"default": "/home/jl_fs/character/dataset"}),
            ("--output", {"default": "/home/jl_fs/character/lora"}),
            ("--steps", {"type": int, "default": 800}),
            ("--rank", {"type": int, "default": 16}),
            ("--resolution", {"type": int, "default": 1024}),
            ("--batch", {"type": int, "default": 1}),
            ("--grad-accum", {"type": int, "default": 4}),
            ("--write-script", {"help": "Write a shell script to run on the GPU"}),
        ],
    )
    add_identity_options(train, 0)
    train.set_defaults(func=cmd_train)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(str(ROOT / ".env"))
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

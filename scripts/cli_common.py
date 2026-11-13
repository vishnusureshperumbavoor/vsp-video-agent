"""Small argparse helpers shared by the agent-facing CLIs."""

from __future__ import annotations

import argparse


def add_options(parser: argparse.ArgumentParser, options: list[tuple[str, dict]]) -> None:
    for flags, kwargs in options:
        parser.add_argument(flags, **kwargs)


def add_media_options(
    parser: argparse.ArgumentParser,
    width: int = 768,
    height: int = 1344,
    steps: int = 4,
) -> None:
    parser.add_argument("--width", type=int, default=width)
    parser.add_argument("--height", type=int, default=height)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

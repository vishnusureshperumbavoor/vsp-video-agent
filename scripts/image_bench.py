#!/usr/bin/env python3
"""Benchmark the GPU image service across resolution and step counts.

Calls the service directly, so it bypasses the per-user quota and bills GPU
time. Start with the default 1024/50 to compare against a known number.

    python scripts/image_bench.py --dry-run
    python scripts/image_bench.py --sizes 1024,768 --steps 50,30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_client import GPUError, add_service_args, request_image, resolve_service  # noqa: E402
from telegram_bot import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def build_configs(sizes: list[int], steps: list[int]) -> list[tuple[int, int]]:
    return [(size, step) for size in sizes for step in steps]


def run_one(
    url: str,
    token: str,
    prompt: str,
    negative_prompt: str,
    size: int,
    steps: int,
    seed: int,
    timeout: float,
) -> tuple[float, int]:
    payload = {
        "prompt": prompt,
        "width": size,
        "height": size,
        "steps": steps,
        "seed": seed,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    elapsed, raw = request_image(url, token, "/generate", payload, timeout)
    return elapsed, len(raw)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the GPU image service")
    parser.add_argument("--prompt", default="a photorealistic portrait in a sunlit cafe")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--sizes", type=parse_list, default=[1024, 768])
    parser.add_argument("--steps", type=parse_list, default=[50, 30])
    parser.add_argument("--seed", type=int, default=42)
    add_service_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(str(ROOT / ".env"))
    args = parse_args(argv)
    try:
        url, token = resolve_service(args.url, args.token)
    except GPUError as exc:
        print(exc, file=sys.stderr)
        return 2
    configs = build_configs(args.sizes, args.steps)
    print(f"{'size':>6} {'steps':>6} {'seconds':>9} {'KB':>8}")
    for size, step in configs:
        if args.dry_run:
            print(f"{size:>6} {step:>6} {'dry':>9} {'—':>8}")
            continue
        try:
            elapsed, size_bytes = run_one(
                url,
                token,
                args.prompt,
                args.negative_prompt,
                size,
                step,
                args.seed,
                args.timeout,
            )
        except GPUError as exc:
            print(f"{size:>6} {step:>6} failed: {exc}", file=sys.stderr)
            return 1
        print(f"{size:>6} {step:>6} {elapsed:>8.1f}s {size_bytes / 1024:>7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

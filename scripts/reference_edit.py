#!/usr/bin/env python3
"""Reference-conditioned edit against the GPU service.

    python scripts/reference_edit.py --reference face.png \
        --prompt "Keep the same woman; put her in a sunlit kitchen" --out kitchen.png

This calls FLUX.2 klein on the GPU (4 steps by default). It is an operator tool:
the result lands on disk, not in the Telegram gallery.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_client import GPUError, add_service_args, request_image, resolve_service  # noqa: E402
from telegram_bot import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def encode_reference(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_edit_payload(
    prompt: str,
    reference_b64: str,
    size: int = 1024,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
) -> dict:
    return {
        "prompt": prompt,
        "reference_b64": reference_b64,
        "width": size,
        "height": size,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference edit on the GPU service")
    parser.add_argument("--reference", required=True, help="Local reference image")
    parser.add_argument("--prompt", required=True, help="What to change")
    parser.add_argument("--out", default="reference_edit.png")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    add_service_args(parser, 1800.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(str(ROOT / ".env"))
    args = parse_args(argv)
    try:
        url, token = resolve_service(args.url, args.token)
    except GPUError as exc:
        print(exc, file=sys.stderr)
        return 2
    payload = build_edit_payload(
        args.prompt,
        encode_reference(args.reference),
        args.size,
        args.steps,
        args.guidance,
        args.seed,
    )
    try:
        elapsed, raw = request_image(url, token, "/edit", payload, args.timeout)
    except GPUError as exc:
        print(f"edit failed: {exc}", file=sys.stderr)
        return 1
    Path(args.out).write_bytes(raw)
    print(f"edit: {elapsed:.1f}s -> {args.out} ({len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

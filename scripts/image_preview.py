#!/usr/bin/env python3
"""Inspect generated images as text, for operators driving this repo from a phone.

The app has no web UI and `public/images/` is git-ignored, so a run that finishes
on a GPU box leaves files that a mobile client cannot display. This tool gives
three text-only answers:

    python scripts/image_preview.py              # listing + a URL for your phone
    python scripts/image_preview.py --ascii      # ASCII preview in the log
    python scripts/image_preview.py --serve      # browse them from a phone

`--serve` binds to the local network only and prints the address your phone can
open. It is an operator convenience, not part of the app: never port-forward it,
and never put it behind a public tunnel — these files can be adult content.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "public" / "images"
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
RAMP = " .:-=+*#%@"


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES)


def describe(path: Path) -> dict:
    """Cheap facts about an image: enough to spot a blank or broken render."""
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        grey = image.convert("L")
        stats = ImageStat.Stat(grey)
        mean = float(stats.mean[0])
        deviation = float(stats.stddev[0])
        return {
            "name": path.name,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "kb": round(path.stat().st_size / 1024, 1),
            "mean": round(mean, 1),
            "blank": deviation < 2.0,
        }


def ascii_art(path: Path, width: int = 40) -> str:
    """Render luminance as text. Half-height keeps characters roughly square."""
    from PIL import Image

    with Image.open(path) as image:
        height = max(1, round(image.height / image.width * width * 0.5))
        small = image.convert("L").resize((width, height))
        # Pillow 14 removes getdata(); prefer its replacement when present.
        read = getattr(small, "get_flattened_data", None) or small.getdata
        pixels = list(read())

    rows = []
    for row in range(height):
        line = "".join(
            RAMP[min(len(RAMP) - 1, pixels[row * width + col] * len(RAMP) // 256)]
            for col in range(width)
        )
        rows.append(line.rstrip())
    return "\n".join(rows)


def lan_ip() -> str:
    """Best-effort LAN address; a UDP socket sends no packets."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.168.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def serve(directory: Path, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    address = lan_ip()
    print(f"serving {directory}", flush=True)
    print(f"  phone (same Wi-Fi):  http://{address}:{port}/", flush=True)
    print(f"  this machine:        http://127.0.0.1:{port}/", flush=True)
    print("LAN only. Ctrl-C to stop; never expose this port publicly.", flush=True)
    with ThreadingHTTPServer(("0.0.0.0", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text views of generated images")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--width", type=int, default=40, help="ASCII columns")
    parser.add_argument("--ascii", action="store_true", help="print an ASCII preview")
    parser.add_argument("--json", action="store_true", help="machine-readable listing")
    parser.add_argument("--serve", action="store_true", help="serve the folder over LAN")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = Path(args.dir).expanduser().resolve()
    images = list_images(directory)

    if args.serve:
        if not images:
            print(f"nothing to serve: {directory} has no images yet")
            return 1
        serve(directory, args.port)
        return 0

    if not images:
        print(f"no images in {directory}")
        print("run the generator first, e.g. scripts/moss_image_runner.py")
        return 1

    try:
        details = [describe(path) for path in images]
    except ImportError:
        print("pillow is required: .venv/bin/pip install pillow")
        return 1
    if args.json:
        print(json.dumps(details, indent=2))
        return 0

    for item in details:
        flag = "  BLANK?" if item["blank"] else ""
        print(
            f"{item['name']}  {item['width']}x{item['height']} "
            f"{item['mode']} {item['kb']} KB  mean={item['mean']}{flag}"
        )
        if args.ascii:
            print(ascii_art(directory / item["name"], args.width))
            print()

    print()
    print(f"{len(images)} image(s) in {directory}")
    print(f"view on a phone (same Wi-Fi): http://{lan_ip()}:{args.port}/")
    print(f"  start it with:  python scripts/image_preview.py --serve --port {args.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

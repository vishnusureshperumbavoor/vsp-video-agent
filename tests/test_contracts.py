"""Contract tests that protect the project's stated boundaries.

1. The public route set may only change deliberately (update the snapshot).
2. The transport vocabulary is JarvisLabs-only, everywhere, forever.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.routing import APIRoute

from server.main import app

ROOT = Path(__file__).resolve().parent.parent

# Adding or removing an endpoint is a contract change: update this list in the
# same commit and say why in the message.
EXPECTED_ROUTES = [
    "/api/gpu/setup",
    "/api/gpu/status",
    "/api/health",
    "/api/video/clean-cache",
    "/api/video/gallery",
    "/api/video/generate/i2v",
    "/api/video/generate/t2v",
    "/api/video/jobs",
    "/api/video/jobs/{job_id}",
    "/api/video/stitch",
]

# Assembled at runtime so this file does not itself trip the scan below.
FORBIDDEN_TERMS = ["co" + "lab", "m" + "cp"]


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def test_route_set_is_stable() -> None:
    actual = sorted(
        route.path for route in app.routes if isinstance(route, APIRoute)
    )
    assert actual == EXPECTED_ROUTES, (
        "The HTTP contract changed. If that is intentional, update EXPECTED_ROUTES "
        "in tests/test_contracts.py and explain the change in the commit message."
    )


def test_no_forbidden_transport_terms_in_the_repository() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if path == Path(__file__).resolve():
            continue  # the needles live here on purpose
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue  # binary asset
        for term in FORBIDDEN_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(ROOT)} contains '{term}'")

    assert not offenders, "Remove these references:\n" + "\n".join(offenders)

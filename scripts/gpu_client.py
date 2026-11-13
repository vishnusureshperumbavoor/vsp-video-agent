"""Small HTTP client shared by the GPU-side operator tools."""

from __future__ import annotations

import argparse
import base64
import os
import time

import requests


class GPUError(RuntimeError):
    """The GPU service could not be reached or returned no image."""


def add_service_args(parser: argparse.ArgumentParser, default_timeout: float = 900.0) -> None:
    parser.add_argument("--url", help="GPU service URL (default: IMAGE_API_URL)")
    parser.add_argument("--token", help="Bearer token (default: IMAGE_API_TOKEN)")
    parser.add_argument("--timeout", type=float, default=default_timeout)


def resolve_service(url: str | None, token: str | None) -> tuple[str, str]:
    resolved_url = (url or os.environ.get("IMAGE_API_URL", "")).rstrip("/")
    resolved_token = token or os.environ.get("IMAGE_API_TOKEN", "")
    if not resolved_url:
        raise GPUError("IMAGE_API_URL is not set")
    if not resolved_token:
        raise GPUError("IMAGE_API_TOKEN is not set")
    return resolved_url, resolved_token


def request_image(
    url: str,
    token: str,
    path: str,
    payload: dict,
    timeout: float,
) -> tuple[float, bytes]:
    """POST one image request and return (elapsed seconds, PNG bytes)."""
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{url.rstrip('/')}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GPUError(str(exc)[:300]) from exc
    elapsed = time.perf_counter() - started
    try:
        return elapsed, base64.b64decode(response.json()["image_b64"])
    except (KeyError, ValueError, TypeError) as exc:
        raise GPUError("GPU service returned no valid image") from exc

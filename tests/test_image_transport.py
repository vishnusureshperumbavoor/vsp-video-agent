"""Transport tests for the GPU image client.

The real endpoint returns 404 from the JarvisLabs proxy while the instance is
paused, which must read as "offline", not as a broken API.
"""

from __future__ import annotations

import base64

import pytest
import requests

from server.transport.image_api import ImageAPIClient, ImageAPIError


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = status_code < 400

    def json(self) -> dict:
        return self._payload


def client() -> ImageAPIClient:
    return ImageAPIClient("http://gpu.test", "gpu-token", timeout=5)


@pytest.mark.parametrize("status", [404, 502, 503, 504])
def test_paused_or_proxied_gpu_is_reported_as_offline(monkeypatch, status: int) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(status))

    with pytest.raises(ImageAPIError, match="offline"):
        client().generate("a rooftop", 512, 4, 42)


def test_rejected_token_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(401))

    with pytest.raises(ImageAPIError, match="IMAGE_API_TOKEN"):
        client().generate("a rooftop", 512, 4, 42)


def test_other_http_errors_keep_the_status(monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(500))

    with pytest.raises(ImageAPIError, match="HTTP 500"):
        client().generate("a rooftop", 512, 4, 42)


def test_success_returns_the_decoded_png(monkeypatch) -> None:
    encoded = base64.b64encode(b"png-bytes").decode("ascii")
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: FakeResponse(200, {"image_b64": encoded})
    )

    assert client().generate("a rooftop", 512, 4, 42) == b"png-bytes"


def test_character_mode_uses_the_character_endpoint(monkeypatch) -> None:
    captured = {}
    encoded = base64.b64encode(b"png-bytes").decode("ascii")

    def fake_post(url, **kwargs):
        captured["url"] = url
        return FakeResponse(200, {"image_b64": encoded})

    monkeypatch.setenv("IMAGE_USE_CHARACTER", "1")
    monkeypatch.setattr(requests, "post", fake_post)

    client().generate("a rooftop", 512, 8, 42)

    assert captured["url"].endswith("/character")

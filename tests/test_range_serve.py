"""Tests for byte-range serving.

Video on a phone depends on this: without `206 Partial Content`, iOS refuses to
play and no browser can seek. The tests drive a real server on an ephemeral port
rather than asserting on the handler internals.
"""

from __future__ import annotations

import http.client
import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import range_serve  # noqa: E402
from range_serve import RangeRequestHandler  # noqa: E402

BODY = bytes(range(256)) * 4  # 1024 deterministic bytes


@pytest.fixture
def server(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(BODY)
    handler = partial(RangeRequestHandler, directory=str(tmp_path))
    handler.log_message = lambda *a, **kw: None  # keep pytest output clean
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def request(port: int, path: str = "/clip.mp4", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_plain_request_is_a_full_200(server) -> None:
    status, headers, body = request(server)

    assert status == 200
    assert body == BODY
    assert headers["Accept-Ranges"] == "bytes"


def test_prefix_range_returns_206(server) -> None:
    status, headers, body = request(server, headers={"Range": "bytes=0-9"})

    assert status == 206
    assert body == BODY[:10]
    assert headers["Content-Range"] == f"bytes 0-9/{len(BODY)}"
    assert headers["Content-Length"] == "10"


def test_open_ended_range_runs_to_the_end(server) -> None:
    status, _, body = request(server, headers={"Range": "bytes=1000-"})

    assert status == 206
    assert body == BODY[1000:]


def test_suffix_range_returns_the_tail(server) -> None:
    status, _, body = request(server, headers={"Range": "bytes=-5"})

    assert status == 206
    assert body == BODY[-5:]


def test_range_past_the_end_is_clamped(server) -> None:
    status, headers, body = request(server, headers={"Range": "bytes=0-99999"})

    assert status == 206
    assert body == BODY
    assert headers["Content-Range"] == f"bytes 0-{len(BODY) - 1}/{len(BODY)}"


def test_unsatisfiable_range_is_416(server) -> None:
    status, _, _ = request(server, headers={"Range": "bytes=5000-"})

    assert status == 416


def test_multi_range_falls_back_to_a_full_response(server) -> None:
    status, _, body = request(server, headers={"Range": "bytes=0-9,20-29"})

    assert status == 200
    assert body == BODY


def test_parse_range_rejects_a_zero_suffix() -> None:
    with pytest.raises(ValueError):
        range_serve.parse_range("-0", 100)

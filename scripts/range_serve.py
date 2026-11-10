#!/usr/bin/env python3
"""A static file handler that supports HTTP byte ranges.

`http.server.SimpleHTTPRequestHandler` always answers `200 OK` for a `Range`
request, which breaks two things an operator actually needs:

* iOS Safari refuses to play video served without `206 Partial Content`
* no client can seek: the browser either downloads the whole file or stalls

This adds single-range support (`bytes=0-`, `bytes=100-199`, `bytes=-500`) and
advertises `Accept-Ranges: bytes`. Multi-range requests fall back to a normal
full response, which is legal.
"""

from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler
from typing import BinaryIO


class LimitedReader:
    """Reads at most `remaining` bytes, so copyfileobj stops at the range end."""

    def __init__(self, handle: BinaryIO, remaining: int):
        self.handle = handle
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self.remaining
        chunk = self.handle.read(min(size, self.remaining))
        self.remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.handle.close()


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus byte-range responses."""

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        header = self.headers.get("Range", "")
        if not header.startswith("bytes=") or "," in header:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        try:
            size = os.path.getsize(path)
            start, end = parse_range(header[len("bytes="):], size)
        except ValueError:
            self.send_error(416, "Requested range not satisfiable")
            return None

        handle = open(path, "rb")
        handle.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return LimitedReader(handle, end - start + 1)


def parse_range(spec: str, size: int) -> tuple[int, int]:
    """Turn a `bytes=` value into an inclusive (start, end) pair.

    Raises ValueError when the range is unsatisfiable, which the caller turns
    into a 416.
    """
    start_text, _, end_text = spec.partition("-")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        # Suffix range: the last N bytes.
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("empty suffix range")
        start = max(0, size - suffix)
        end = size - 1

    if start < 0 or start > end or start >= size:
        raise ValueError("unsatisfiable range")
    return start, min(end, size - 1)

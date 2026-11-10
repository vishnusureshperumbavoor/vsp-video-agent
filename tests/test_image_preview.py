"""Tests for the phone-friendly image viewer.

The point of the tool is that a mobile operator can tell a good render from a
blank one without displaying the image, so the checks are about the text
output: dimensions, the blank flag, and the ASCII grid shape.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import image_preview as preview  # noqa: E402


def request_on(port: int, path: str):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def write_png(path: Path, size=(64, 32), color=(128, 128, 128)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_list_images_ignores_non_images(tmp_path) -> None:
    write_png(tmp_path / "b.png")
    write_png(tmp_path / "a.webp")
    (tmp_path / "notes.txt").write_text("ignore me")

    names = [p.name for p in preview.list_images(tmp_path)]

    assert names == ["a.webp", "b.png"]


def test_list_images_on_a_missing_folder_is_empty(tmp_path) -> None:
    assert preview.list_images(tmp_path / "nope") == []


def test_describe_reports_dimensions_and_size(tmp_path) -> None:
    path = tmp_path / "shot.png"
    # A gradient, so the blank check has real contrast to measure.
    Image.linear_gradient("L").resize((1024, 768)).save(path)

    info = preview.describe(path)

    assert (info["width"], info["height"]) == (1024, 768)
    assert info["kb"] > 0
    assert info["blank"] is False


def test_describe_flags_a_flat_image_as_blank(tmp_path) -> None:
    path = write_png(tmp_path / "flat.png", color=(0, 0, 0))

    assert preview.describe(path)["blank"] is True


def test_ascii_art_has_one_row_per_half_height(tmp_path) -> None:
    path = write_png(tmp_path / "art.png", size=(80, 80))

    rows = preview.ascii_art(path, width=20).splitlines()

    # 80/80 * 20 * 0.5 == 10 rows.
    assert len(rows) == 10
    assert all(len(row) <= 20 for row in rows)


def test_ascii_art_uses_darker_characters_for_dark_pixels(tmp_path) -> None:
    path = write_png(tmp_path / "dark.png", color=(0, 0, 0))

    art = preview.ascii_art(path, width=8)

    assert set(art.replace("\n", "")) <= {" "}


def test_lan_ip_is_an_address_string() -> None:
    address = preview.lan_ip()

    assert address.count(".") == 3
    assert all(part.isdigit() for part in address.split("."))


def run_main(monkeypatch, argv, capsys):
    monkeypatch.setattr(sys, "argv", ["image_preview.py", *argv])
    code = preview.main()
    return code, capsys.readouterr().out


def test_main_lists_images_and_the_phone_url(tmp_path, monkeypatch, capsys) -> None:
    write_png(tmp_path / "one.png")

    code, out = run_main(monkeypatch, ["--dir", str(tmp_path)], capsys)

    assert code == 0
    assert "one.png" in out
    assert "http://" in out and ":8080/" in out


def test_main_json_is_machine_readable(tmp_path, monkeypatch, capsys) -> None:
    write_png(tmp_path / "one.png", size=(16, 16))

    code, out = run_main(monkeypatch, ["--dir", str(tmp_path), "--json"], capsys)

    assert code == 0
    assert json.loads(out)[0]["width"] == 16


def test_main_explains_an_empty_folder(tmp_path, monkeypatch, capsys) -> None:
    code, out = run_main(monkeypatch, ["--dir", str(tmp_path / "empty")], capsys)

    assert code == 1
    assert "no images" in out


def test_has_content_accepts_a_folder_of_subfolders(tmp_path) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip.mp4").write_bytes(b"x")

    assert preview.has_content(tmp_path) is True


def test_has_content_rejects_an_empty_folder(tmp_path) -> None:
    (tmp_path / "empty").mkdir()

    assert preview.has_content(tmp_path / "empty") is False
    assert preview.has_content(tmp_path / "missing") is False


def test_make_server_honours_the_bind_address(tmp_path) -> None:
    write_png(tmp_path / "one.png")
    httpd = preview.make_server(tmp_path, 0, "127.0.0.1")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = request_on(httpd.server_address[1], "/one.png")
        assert status == 200
        assert body[:4] == b"\x89PNG"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_make_server_refuses_a_port_already_taken(tmp_path) -> None:
    write_png(tmp_path / "one.png")
    first = preview.make_server(tmp_path, 0, "127.0.0.1")
    port = first.server_address[1]
    try:
        with pytest.raises(OSError):
            preview.make_server(tmp_path, port, "127.0.0.1")
    finally:
        first.server_close()

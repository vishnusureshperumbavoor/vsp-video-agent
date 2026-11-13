"""HTTP contract for the Telegram-facing image product.

The transport is replaced with FakeImageAPI and the database with a temporary
SQLite file, so nothing here touches the GPU or the network.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from server import main, runtime
from tests.fakes import FakeImageAPI

TOKEN = "test-bot-token"


def auth(owner_id: int = 7) -> dict[str, str]:
    return {"X-Telegram-Id": str(owner_id), "X-Bot-Token": TOKEN}


def configure(tmp_path, monkeypatch, auto_dispatch: str) -> None:
    monkeypatch.setenv("IMAGE_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("BOT_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("IMAGE_AUTO_DISPATCH", auto_dispatch)
    monkeypatch.setenv("DAILY_IMAGE_LIMIT", "10")
    runtime.set_image_service(runtime.build_service(api=FakeImageAPI()))


@pytest.fixture
def client(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, "0")
    with TestClient(main.app) as test_client:
        yield test_client
    runtime.set_image_service(None)


@pytest.fixture
def auto_client(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, "1")
    with TestClient(main.app) as test_client:
        yield test_client
    runtime.set_image_service(None)


def test_missing_or_wrong_bot_credentials_are_rejected(client) -> None:
    assert client.get("/api/characters").status_code == 401
    wrong = {"X-Telegram-Id": "7", "X-Bot-Token": "not-the-token"}
    assert client.get("/api/characters", headers=wrong).status_code == 401


def test_character_creation_and_selection(client) -> None:
    response = client.post(
        "/api/characters",
        headers=auth(),
        json={"name": "Ava", "appearance": "silver hair, green eyes"},
    )

    assert response.status_code == 200
    character = response.json()
    assert character["name"] == "Ava"
    assert client.get("/api/characters", headers=auth()).json()[0]["id"] == character["id"]

    selected = client.post(
        f"/api/characters/{character['id']}/select", headers=auth()
    )
    assert selected.status_code == 200
    active = client.get("/api/characters/active", headers=auth())
    assert active.json()["id"] == character["id"]


def test_active_character_is_404_before_selection(client) -> None:
    assert client.get("/api/characters/active", headers=auth()).status_code == 404


def test_image_job_runs_and_appears_in_the_gallery(client) -> None:
    character = client.post(
        "/api/characters",
        headers=auth(),
        json={"name": "Ava", "appearance": "silver hair"},
    ).json()
    client.post(f"/api/characters/{character['id']}/select", headers=auth())

    response = client.post(
        "/api/images", headers=auth(), json={"prompt": "a rooftop at sunset"}
    )
    assert response.status_code == 201
    job = response.json()
    assert job["status"] == "queued"

    service = runtime.get_image_service()
    asyncio.run(service.process(job["id"]))

    done = client.get(f"/api/images/{job['id']}", headers=auth()).json()
    assert done["status"] == "completed"
    assert done["image_url"] == f"/images/7/{job['id']}.png"
    assert client.get("/api/images", headers=auth()).json()[0]["id"] == job["id"]


def test_jobs_are_scoped_to_their_owner(client) -> None:
    character = client.post(
        "/api/characters",
        headers=auth(),
        json={"name": "Ava", "appearance": "silver hair"},
    ).json()
    job = client.post(
        "/api/images",
        headers=auth(),
        json={"prompt": "a rooftop", "character_id": character["id"]},
    ).json()

    assert client.get(f"/api/images/{job['id']}", headers=auth(8)).status_code == 404
    assert client.get("/api/images", headers=auth(8)).json() == []


def test_unknown_job_returns_404(client) -> None:
    assert client.get("/api/images/does-not-exist", headers=auth()).status_code == 404


def test_prompt_only_image_needs_no_character(client) -> None:
    response = client.post("/api/images", headers=auth(), json={"prompt": "a cat"})

    assert response.status_code == 201
    job = response.json()
    assert job["character_id"] == ""
    assert job["prompt"] == "a cat"


def test_auto_dispatch_finishes_on_the_server_event_loop(auto_client) -> None:
    """Regression: a sync endpoint used to call create_task with no running loop."""
    character = auto_client.post(
        "/api/characters",
        headers=auth(),
        json={"name": "Ava", "appearance": "silver hair"},
    ).json()
    auto_client.post(f"/api/characters/{character['id']}/select", headers=auth())

    job = auto_client.post("/api/images", headers=auth(), json={"prompt": "a rooftop"})
    assert job.status_code == 201

    status = "queued"
    deadline = time.time() + 5
    while time.time() < deadline:
        status = auto_client.get(
            f"/api/images/{job.json()['id']}", headers=auth()
        ).json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert status == "completed"

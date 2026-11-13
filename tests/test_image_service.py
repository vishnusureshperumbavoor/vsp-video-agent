"""Service-level tests: queue, quota, storage and failure handling.

Everything runs in-process against FakeImageAPI and a temporary SQLite file, so
the whole image lifecycle is verified for free.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from server.services.image_jobs import ImageJobService, NotFound, QuotaExceeded
from server.settings import Settings
from server.storage import LocalImageStorage
from server.store import Store
from server.transport.image_api import ImageAPIError
from tests.fakes import FakeImageAPI


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "db_path": tmp_path / "app.db",
        "images_dir": tmp_path / "images",
        "image_api_url": "http://127.0.0.1:6006",
        "image_api_token": "gpu-token",
        "bot_internal_token": "bot-token",
        "public_base_url": "",
        "daily_image_limit": 10,
        "global_daily_limit": 100,
        "default_steps": 4,
        "request_timeout": 10.0,
        "auto_dispatch": False,
    }
    values.update(overrides)
    return Settings(**values)


def make_service(
    tmp_path: Path, api: FakeImageAPI | None = None, **overrides
) -> tuple[ImageJobService, FakeImageAPI]:
    settings = make_settings(tmp_path, **overrides)
    fake = api or FakeImageAPI()
    service = ImageJobService(
        Store(settings.db_path),
        fake,
        LocalImageStorage(settings.images_dir),
        settings,
    )
    return service, fake


def ready_character(service: ImageJobService) -> str:
    character = service.create_character(
        owner_id=7,
        name="Ava",
        appearance="silver hair, green eyes",
        style="cinematic portrait",
    )
    service.select_character(7, character.id)
    return character.id


def test_completed_job_is_saved_and_served(tmp_path: Path) -> None:
    service, fake = make_service(tmp_path)
    ready_character(service)

    job = service.create_job(owner_id=7, scene="a rooftop at sunset")
    assert job.status == "queued"
    asyncio.run(service.process(job.id))

    stored = service.get_job(7, job.id)
    assert stored.status == "completed"
    assert stored.image_url == f"/images/7/{job.id}.png"
    assert (tmp_path / "images" / "7" / f"{job.id}.png").exists()
    assert fake.calls[0]["prompt"] == (
        "silver hair, green eyes, cinematic portrait, a rooftop at sunset"
    )


def test_processing_twice_does_not_call_the_gpu_twice(tmp_path: Path) -> None:
    service, fake = make_service(tmp_path)
    ready_character(service)
    job = service.create_job(owner_id=7, scene="a rooftop")

    asyncio.run(service.process(job.id))
    asyncio.run(service.process(job.id))

    assert len(fake.calls) == 1


def test_daily_quota_is_enforced(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, daily_image_limit=1)
    ready_character(service)
    service.create_job(owner_id=7, scene="first")

    with pytest.raises(QuotaExceeded):
        service.create_job(owner_id=7, scene="second")


def test_gpu_failure_marks_the_job_failed(tmp_path: Path) -> None:
    service, _ = make_service(
        tmp_path,
        api=FakeImageAPI(error=ImageAPIError("GPU image service is offline.")),
    )
    ready_character(service)
    job = service.create_job(owner_id=7, scene="a rooftop")

    asyncio.run(service.process(job.id))

    stored = service.get_job(7, job.id)
    assert stored.status == "failed"
    assert "offline" in stored.error


def test_users_cannot_read_each_others_jobs(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    ready_character(service)
    job = service.create_job(owner_id=7, scene="a rooftop")

    with pytest.raises(NotFound):
        service.get_job(8, job.id)


def test_selecting_an_unknown_character_fails(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)

    with pytest.raises(NotFound):
        service.select_character(7, "does-not-exist")


def test_prompt_only_job_uses_the_raw_prompt(tmp_path: Path) -> None:
    service, fake = make_service(tmp_path)

    job = service.create_job(owner_id=7, scene="a cat on a rooftop")
    asyncio.run(service.process(job.id))

    stored = service.get_job(7, job.id)
    assert stored.status == "completed"
    assert stored.character_id == ""
    assert fake.calls[0]["prompt"] == "a cat on a rooftop"


def test_global_daily_cap_stops_runaway_public_use(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, daily_image_limit=0, global_daily_limit=1)

    service.create_job(owner_id=7, scene="first")

    with pytest.raises(QuotaExceeded):
        service.create_job(owner_id=8, scene="second")


def test_start_requeues_jobs_left_queued_by_a_restart(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    ready_character(service)
    job = service.create_job(owner_id=7, scene="a rooftop")
    assert job.status == "queued"

    restarted, fake = make_service(tmp_path)

    async def run() -> None:
        restarted.start()
        await restarted.drain()
        await restarted.stop()

    asyncio.run(run())

    assert restarted.get_job(7, job.id).status == "completed"
    assert len(fake.calls) == 1

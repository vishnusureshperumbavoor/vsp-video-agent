"""End-to-end job lifecycle against FakeGPU.

This is the test that makes the pipeline safe to change: queue -> stage ->
spawn detached -> poll log -> download -> metadata, with no GPU and no SSH.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from server import main
from tests.fakes import FakeGPU


@pytest.fixture
def gpu(tmp_path, monkeypatch):
    fake = FakeGPU(tmp_path / "remote")
    monkeypatch.setattr(main, "internal_gpu_client", fake)
    videos = tmp_path / "videos"
    videos.mkdir()
    monkeypatch.setattr(main, "VIDEOS_DIR", videos)
    monkeypatch.setattr(main, "METADATA_FILE", videos / "metadata.json")
    main.jobs.clear()
    # run_detached_job sleeps 5s between polls; make the test instant.
    monkeypatch.setattr(main.asyncio, "sleep", _noop_sleep)
    return fake


async def _noop_sleep(*_args, **_kwargs) -> None:
    return None


def _queue(job_id: str = "job-1", filename: str = "out.mp4") -> dict:
    job = {
        "id": job_id,
        "type": "T2V",
        "status": "queued",
        "created_at": 0.0,
        "prompt": "a test prompt",
        "metadata": {"filename": filename, "prompt": "a test prompt"},
        "logs": [],
    }
    main.jobs[job_id] = job
    return job


def test_successful_job_downloads_video_and_writes_metadata(gpu) -> None:
    job = _queue()

    asyncio.run(
        main.run_detached_job(
            "job-1",
            "print('hello')",
            "out.mp4",
            dict(job["metadata"]),
        )
    )

    assert job["status"] == "completed"
    assert job["video_url"] == "/videos/out.mp4"
    assert (main.VIDEOS_DIR / "out.mp4").exists()
    assert gpu.spawned, "the job must be spawned detached, not run inline"
    assert "setsid" not in gpu.spawned[0]  # setsid lives in the transport

    saved = json.loads(main.METADATA_FILE.read_text())
    assert saved[0]["filename"] == "out.mp4"
    assert "size_mb" in saved[0]


def test_job_stages_script_under_the_remote_job_dir(gpu) -> None:
    _queue()
    asyncio.run(main.run_detached_job("job-1", "print(1)", "out.mp4", {}))

    staged = [p for p in gpu.remote_files if p.endswith("job-1.py")]
    assert staged == ["/home/vsp_jobs/job-1.py"]


def test_i2v_uploads_the_reference_image(gpu, tmp_path) -> None:
    _queue("job-2", "i2v.mp4")
    image = tmp_path / "character.png"
    image.write_bytes(b"png-bytes")

    asyncio.run(
        main.run_detached_job(
            "job-2",
            "print(1)",
            "i2v.mp4",
            {},
            str(image),
            "/home/vsp_jobs/uploads/character.png",
        )
    )

    assert gpu.uploaded == [(str(image), "/home/vsp_jobs/uploads/character.png")]


def test_traceback_in_log_marks_the_job_failed(tmp_path, monkeypatch) -> None:
    fake = FakeGPU(
        tmp_path / "remote",
        log="loading model\nTraceback (most recent call last):\nValueError: boom",
    )
    monkeypatch.setattr(main, "internal_gpu_client", fake)
    videos = tmp_path / "videos"
    videos.mkdir()
    monkeypatch.setattr(main, "VIDEOS_DIR", videos)
    monkeypatch.setattr(main, "METADATA_FILE", videos / "metadata.json")
    monkeypatch.setattr(main.asyncio, "sleep", _noop_sleep)
    main.jobs.clear()

    job = _queue("job-3")
    asyncio.run(main.run_detached_job("job-3", "print(1)", "out.mp4", {}))

    assert job["status"] == "failed"
    assert "ValueError: boom" in job["error"]


def test_download_failure_marks_the_job_failed(gpu, monkeypatch) -> None:
    _queue("job-4")

    async def broken_download(remote_path: str, local_path: str) -> bool:
        return False

    monkeypatch.setattr(gpu, "download_file", broken_download)
    job = main.jobs["job-4"]
    asyncio.run(main.run_detached_job("job-4", "print(1)", "out.mp4", {}))

    assert job["status"] == "failed"
    assert "Download failed" in job["error"]

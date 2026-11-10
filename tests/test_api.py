"""HTTP contract tests.

They do not touch a GPU: the transport is replaced with FakeGPU. What they pin
down is the part other agents depend on - status codes, the A100 guardrail and
the route set.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import main
from tests.fakes import FakeGPU


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeGPU(tmp_path / "remote")
    monkeypatch.setattr(main, "internal_gpu_client", fake)
    main.jobs.clear()
    return TestClient(main.app)


@pytest.fixture
def no_gpu_client(tmp_path, monkeypatch):
    fake = FakeGPU(tmp_path / "remote", connected=False)
    monkeypatch.setattr(main, "internal_gpu_client", fake)
    main.jobs.clear()
    return TestClient(main.app)


@pytest.fixture
def l4_client(tmp_path, monkeypatch):
    fake = FakeGPU(tmp_path / "remote", gpu_name="NVIDIA L4")
    monkeypatch.setattr(main, "internal_gpu_client", fake)
    main.jobs.clear()
    return TestClient(main.app)


def test_health_reports_a100_ready(client) -> None:
    body = client.get("/api/health").json()

    assert body["status"] == "healthy"
    assert body["transport"] == "jarvislabs-ssh"
    assert body["gpu_connected"] is True
    assert body["a100_ready"] is True
    assert "A100" in body["gpu"]


def test_health_degrades_without_an_instance(no_gpu_client) -> None:
    body = no_gpu_client.get("/api/health").json()

    assert body["gpu_connected"] is False
    assert body["a100_ready"] is False


def test_gpu_status_is_always_200(client) -> None:
    response = client.get("/api/gpu/status")

    assert response.status_code == 200
    assert response.json()["connected"] is True


def test_t2v_refuses_to_run_without_an_instance(no_gpu_client) -> None:
    response = no_gpu_client.post(
        "/api/video/generate/t2v", json={"prompt": "x", "duration": 3}
    )

    assert response.status_code == 400
    assert "No running JarvisLabs GPU" in response.json()["detail"]


def test_t2v_refuses_to_run_on_a_non_a100(l4_client) -> None:
    response = l4_client.post(
        "/api/video/generate/t2v", json={"prompt": "x", "duration": 3}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "A100" in detail
    assert "NVIDIA L4" in detail


def test_stitch_rejects_a_single_clip(client) -> None:
    response = client.post("/api/video/stitch", json={"video_filenames": ["a.mp4"]})

    assert response.status_code == 400


def test_unknown_job_returns_404(client) -> None:
    assert client.get("/api/video/jobs/does-not-exist").status_code == 404


def test_gallery_is_a_list(client) -> None:
    response = client.get("/api/video/gallery")

    assert response.status_code == 200
    assert isinstance(response.json(), list)

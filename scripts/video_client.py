"""HTTP client for the video API, usable from any agent or script."""

from __future__ import annotations

from pathlib import Path

import requests


class VideoError(RuntimeError):
    """The backend rejected the request or is unreachable."""


class VideoClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, payload: dict | None = None):
        try:
            response = requests.request(
                method, f"{self.base_url}{path}", json=payload, timeout=60
            )
        except requests.RequestException as exc:
            raise VideoError("Backend is unreachable.") from exc
        if not response.ok:
            raise VideoError(
                f"Backend returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()

    def create_t2v(self, payload: dict) -> dict:
        return self._request("POST", "/api/video/generate/t2v", payload)

    def create_i2v(self, image_path: str, data: dict) -> dict:
        try:
            with open(image_path, "rb") as handle:
                response = requests.post(
                    f"{self.base_url}/api/video/generate/i2v",
                    data=data,
                    files={"image": (Path(image_path).name, handle, "image/png")},
                    timeout=120,
                )
        except (OSError, requests.RequestException) as exc:
            raise VideoError(f"Image upload failed: {exc}") from exc
        if not response.ok:
            raise VideoError(
                f"Backend returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json()

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/api/video/jobs/{job_id}")

    def list_jobs(self) -> list[dict]:
        return self._request("GET", "/api/video/jobs")

    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def gpu_status(self) -> dict:
        return self._request("GET", "/api/gpu/status")

    def clean_cache(self) -> dict:
        return self._request("POST", "/api/video/clean-cache", {})

"""HTTP transport to the GPU image service.

The GPU runs ``scripts/image_server.py`` and is reachable only with a bearer
token. Nothing here knows about SQLite, FastAPI or the Telegram bot.
"""

from __future__ import annotations

import base64

import requests


class ImageAPIError(RuntimeError):
    """The GPU service is down, rejected the token, or returned no image."""


class ImageAPIClient:
    def __init__(self, base_url: str, token: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        size: int,
        steps: int,
        seed: int,
        negative_prompt: str = "",
    ) -> bytes:
        payload = {
            "prompt": prompt,
            "width": size,
            "height": size,
            "steps": steps,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ImageAPIError(
                "GPU image service is offline. Resume the instance and retry."
            ) from exc

        if response.status_code == 401:
            raise ImageAPIError("GPU image service rejected IMAGE_API_TOKEN.")
        if not response.ok:
            raise ImageAPIError(
                f"GPU image service failed with HTTP {response.status_code}."
            )
        try:
            encoded = response.json()["image_b64"]
            return base64.b64decode(encoded, validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise ImageAPIError("GPU image service returned no valid image.") from exc

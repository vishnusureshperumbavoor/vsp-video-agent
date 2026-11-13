"""Where generated images live.

``LocalImageStorage`` is the beta implementation: files under ``public/images``
served by the API. The protocol is the seam for R2/S3 later, so the service and
the routers do not change when storage moves off the box.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

_SAFE_ID = re.compile(r"^[a-f0-9]{32}$")


class ImageStorage(Protocol):
    """Persist one image and return the URL to serve it from."""

    def save(self, owner_id: int, job_id: str, data: bytes) -> str: ...


class LocalImageStorage:
    def __init__(self, root: str | Path, public_base_url: str = "") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")

    def save(self, owner_id: int, job_id: str, data: bytes) -> str:
        if not data:
            raise ValueError("refusing to store an empty image")
        if owner_id <= 0:
            raise ValueError("owner_id must be positive")
        if not _SAFE_ID.match(job_id):
            raise ValueError("job_id is not a safe file name")
        folder = self.root / str(owner_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{job_id}.png").write_bytes(data)
        relative = f"/images/{owner_id}/{job_id}.png"
        return f"{self.public_base_url}{relative}" if self.public_base_url else relative

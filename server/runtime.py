"""Lazily built singletons for the image product.

Import-time work is deliberately avoided: the database is opened and the model
settings are read on the first request. Tests can therefore point the product at
a temporary directory by setting environment variables and calling
:func:`set_image_service`.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from server.services.image_jobs import ImageJobService
from server.settings import Settings
from server.storage import ImageStorage, LocalImageStorage
from server.store import Store
from server.transport.image_api import ImageAPIClient

_lock = threading.Lock()
_service: ImageJobService | None = None


@asynccontextmanager
async def image_lifespan(_app):
    """Resume queued jobs on startup, but stay inert when the product is off."""
    settings = Settings.from_env()
    if not settings.bot_internal_token:
        yield
        return
    service = get_image_service()
    service.start()
    try:
        yield
    finally:
        await service.stop()


def build_service(
    settings: Settings | None = None,
    *,
    api: ImageAPIClient | None = None,
    storage: ImageStorage | None = None,
    store: Store | None = None,
) -> ImageJobService:
    resolved = settings or Settings.from_env()
    return ImageJobService(
        store=store or Store(resolved.db_path),
        api=api
        or ImageAPIClient(
            resolved.image_api_url,
            resolved.image_api_token,
            resolved.request_timeout,
        ),
        storage=storage or LocalImageStorage(resolved.images_dir, resolved.public_base_url),
        settings=resolved,
    )


def get_image_service() -> ImageJobService:
    global _service
    with _lock:
        if _service is None:
            _service = build_service()
        return _service


def set_image_service(service: ImageJobService | None) -> None:
    global _service
    with _lock:
        _service = service

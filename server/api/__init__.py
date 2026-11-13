"""Mount the girlfriend image product on the main FastAPI app.

``server/main.py`` keeps one line for this so the wiring file does not grow with
every feature: routers live beside this file, and static image serving is mounted
here too.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.settings import Settings


def register_image_app(app: FastAPI) -> None:
    from server.api.characters import router as characters_router
    from server.api.images import router as images_router

    images_dir = Settings.from_env().images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    app.include_router(characters_router)
    app.include_router(images_router)
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

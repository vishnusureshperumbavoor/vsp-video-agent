"""Runtime settings for the image product, all overridable from the environment.

Defaults are safe for local development: SQLite under ``data/``, images under
``public/images/``, and no shared secret (which means the API rejects callers
until ``BOT_INTERNAL_TOKEN`` is set).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from server.config import DATA_DIR, IMAGES_DIR


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the image product needs, resolved once per request."""

    db_path: Path
    images_dir: Path
    image_api_url: str
    image_api_token: str
    bot_internal_token: str
    public_base_url: str
    daily_image_limit: int
    global_daily_limit: int
    default_steps: int
    request_timeout: float
    auto_dispatch: bool

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=Path(os.environ.get("IMAGE_DB_PATH", DATA_DIR / "girlfriend.db")),
            images_dir=Path(os.environ.get("IMAGES_DIR", IMAGES_DIR)),
            image_api_url=os.environ.get("IMAGE_API_URL", "http://127.0.0.1:6006"),
            image_api_token=os.environ.get("IMAGE_API_TOKEN", ""),
            bot_internal_token=os.environ.get("BOT_INTERNAL_TOKEN", ""),
            public_base_url=os.environ.get("PUBLIC_BASE_URL", "").rstrip("/"),
            daily_image_limit=int(os.environ.get("DAILY_IMAGE_LIMIT", "10")),
            global_daily_limit=int(os.environ.get("GLOBAL_DAILY_LIMIT", "100")),
            default_steps=int(os.environ.get("IMAGE_DEFAULT_STEPS", "4")),
            request_timeout=float(os.environ.get("IMAGE_TIMEOUT", "180")),
            auto_dispatch=_env_bool("IMAGE_AUTO_DISPATCH", True),
        )

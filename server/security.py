"""Authentication for the bot-facing API.

Users authenticate with Telegram. The bot is the only caller, so it presents a
shared secret plus the Telegram id it already verified. The browser and the GPU
never see either value.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException

from server.settings import Settings


def require_user(
    x_telegram_id: Annotated[int, Header(alias="X-Telegram-Id")] = 0,
    x_bot_token: Annotated[str, Header(alias="X-Bot-Token")] = "",
) -> int:
    expected = Settings.from_env().bot_internal_token
    if not expected or not hmac.compare_digest(x_bot_token, expected):
        raise HTTPException(status_code=401, detail="invalid bot credentials")
    if x_telegram_id <= 0:
        raise HTTPException(status_code=401, detail="invalid Telegram id")
    return x_telegram_id

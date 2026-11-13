"""The character bible: the stable identity behind every generated image.

Pure Python. No FastAPI, no database, no transport. The API, the queue and the
Telegram bot all compose prompts through this module so "the same girl" means
the same thing everywhere.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass

from server.domain.records import build_from_row

MAX_NAME = 60
MAX_TEXT = 1000
DEFAULT_SEED = 42


class InvalidCharacter(ValueError):
    """A character field failed validation."""


def clean_text(value: str, field_name: str, limit: int = MAX_TEXT) -> str:
    """Collapse whitespace and enforce a length limit."""
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) > limit:
        raise InvalidCharacter(f"{field_name} must be at most {limit} characters")
    return text


def required_text(value: str, field_name: str, limit: int = MAX_TEXT) -> str:
    text = clean_text(value, field_name, limit)
    if not text:
        raise InvalidCharacter(f"{field_name} is required")
    return text


@dataclass(frozen=True, slots=True)
class Character:
    """A fictional persona with a repeatable look."""

    id: str
    owner_id: int
    name: str
    appearance: str
    style: str = ""
    negative_prompt: str = ""
    seed: int = DEFAULT_SEED
    created_at: float = 0.0

    @classmethod
    def create(
        cls,
        owner_id: int,
        name: str,
        appearance: str,
        style: str = "",
        negative_prompt: str = "",
        seed: int = DEFAULT_SEED,
        created_at: float = 0.0,
    ) -> Character:
        if owner_id <= 0:
            raise InvalidCharacter("owner_id must be a positive Telegram id")
        return cls(
            id=uuid.uuid4().hex,
            owner_id=owner_id,
            name=required_text(name, "name", MAX_NAME),
            appearance=required_text(appearance, "appearance"),
            style=clean_text(style, "style"),
            negative_prompt=clean_text(negative_prompt, "negative_prompt"),
            seed=int(seed),
            created_at=created_at,
        )

    def compose_prompt(self, scene: str) -> str:
        """Join the fixed identity with the requested scene, in a stable order."""
        scene_text = required_text(scene, "prompt")
        parts = (self.appearance, self.style, scene_text)
        return ", ".join(part for part in parts if part)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> Character:
        return build_from_row(cls, row)

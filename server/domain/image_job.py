"""Image job lifecycle. Pure Python: no HTTP, no database, no GPU.

The state machine lives here so the queue, the API and the tests agree on what
"queued", "running", "completed" and "failed" mean.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass

from server.domain.records import build_from_row

MIN_SIZE = 256
MAX_SIZE = 1536
MIN_STEPS = 1
MAX_STEPS = 50
MAX_PROMPT = 1000
STATUSES = ("queued", "running", "completed", "failed")
ACTIVE_STATUSES = ("queued", "running")


class InvalidJob(ValueError):
    """A job request failed validation."""


def validate_prompt(prompt: str) -> str:
    text = " ".join((prompt or "").split())
    if not text:
        raise InvalidJob("prompt is required")
    if len(text) > MAX_PROMPT:
        raise InvalidJob(f"prompt must be at most {MAX_PROMPT} characters")
    return text


def validate_size(size: int) -> int:
    value = int(size)
    if value % 16:
        raise InvalidJob("size must be a multiple of 16")
    if not MIN_SIZE <= value <= MAX_SIZE:
        raise InvalidJob(f"size must be between {MIN_SIZE} and {MAX_SIZE}")
    return value


def validate_steps(steps: int) -> int:
    value = int(steps)
    if not MIN_STEPS <= value <= MAX_STEPS:
        raise InvalidJob(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    return value


@dataclass(slots=True)
class ImageJob:
    """One request for one picture."""

    id: str
    owner_id: int
    character_id: str
    prompt: str
    size: int
    steps: int
    seed: int
    status: str = "queued"
    image_url: str | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    generation_seconds: float | None = None

    @classmethod
    def create(
        cls,
        owner_id: int,
        character_id: str,
        prompt: str,
        size: int = 1024,
        steps: int = 4,
        seed: int = 42,
        now: float | None = None,
    ) -> ImageJob:
        if owner_id <= 0:
            raise InvalidJob("owner_id must be positive")
        moment = time.time() if now is None else now
        return cls(
            id=uuid.uuid4().hex,
            owner_id=owner_id,
            character_id=character_id,
            prompt=validate_prompt(prompt),
            size=validate_size(size),
            steps=validate_steps(steps),
            seed=int(seed),
            created_at=moment,
            updated_at=moment,
        )

    def mark_running(self, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        if self.started_at is None:
            self.started_at = moment
        self._transition("running", moment)

    def mark_completed(self, image_url: str, now: float | None = None) -> None:
        if not image_url:
            raise InvalidJob("image_url is required")
        moment = time.time() if now is None else now
        self._record_duration(moment)
        self.image_url = image_url
        self.error = None
        self._transition("completed", moment)

    def mark_failed(self, error: str, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        self._record_duration(moment)
        self.error = error or "generation failed"
        self._transition("failed", moment)

    def _record_duration(self, moment: float) -> None:
        if self.started_at is not None and self.generation_seconds is None:
            self.generation_seconds = round(moment - self.started_at, 1)

    def _transition(self, status: str, now: float | None = None) -> None:
        if status not in STATUSES:
            raise InvalidJob(f"unknown status: {status}")
        self.status = status
        self.updated_at = time.time() if now is None else now

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> ImageJob:
        return build_from_row(cls, row)

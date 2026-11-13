"""Domain rules for the character bible and the image job state machine."""

from __future__ import annotations

import pytest

from server.domain.character import Character, InvalidCharacter
from server.domain.image_job import ImageJob, InvalidJob, validate_size, validate_steps


def test_prompt_keeps_identity_before_the_scene() -> None:
    character = Character.create(
        owner_id=7,
        name="Ava",
        appearance="silver hair, green eyes",
        style="cinematic portrait",
    )

    assert character.compose_prompt("a rooftop at sunset") == (
        "silver hair, green eyes, cinematic portrait, a rooftop at sunset"
    )


def test_character_requires_a_name() -> None:
    with pytest.raises(InvalidCharacter):
        Character.create(owner_id=7, name="  ", appearance="silver hair")


def test_character_requires_an_appearance() -> None:
    with pytest.raises(InvalidCharacter):
        Character.create(owner_id=7, name="Ava", appearance="")


def test_character_owner_must_be_a_telegram_id() -> None:
    with pytest.raises(InvalidCharacter):
        Character.create(owner_id=0, name="Ava", appearance="silver hair")


@pytest.mark.parametrize("size", [256, 512, 1024, 1536])
def test_valid_sizes_are_multiples_of_16(size: int) -> None:
    assert validate_size(size) == size


@pytest.mark.parametrize("size", [250, 1000, 1552])
def test_invalid_sizes_are_rejected(size: int) -> None:
    with pytest.raises(InvalidJob):
        validate_size(size)


@pytest.mark.parametrize("steps", [1, 4, 50])
def test_valid_step_counts(steps: int) -> None:
    assert validate_steps(steps) == steps


@pytest.mark.parametrize("steps", [0, 51])
def test_invalid_step_counts_are_rejected(steps: int) -> None:
    with pytest.raises(InvalidJob):
        validate_steps(steps)


def test_job_moves_through_its_states() -> None:
    job = ImageJob.create(
        owner_id=7,
        character_id="character",
        prompt="a rooftop",
        now=10.0,
    )

    assert job.status == "queued"
    assert job.created_at == 10.0
    job.mark_running(now=11.0)
    assert job.status == "running"
    assert job.updated_at == 11.0
    job.mark_completed("/images/7/a.png", now=12.0)
    assert job.status == "completed"
    assert job.image_url == "/images/7/a.png"
    job.mark_failed("GPU offline", now=13.0)
    assert job.status == "failed"
    assert job.error == "GPU offline"


def test_job_rejects_an_empty_prompt() -> None:
    with pytest.raises(InvalidJob):
        ImageJob.create(owner_id=7, character_id="character", prompt="   ")

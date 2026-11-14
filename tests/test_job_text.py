"""Captions and status labels for image jobs.

Character jobs must not echo the composed appearance prompt back to the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from job_text import character_name, image_caption, job_label  # noqa: E402

GABRIELA = {"id": "a0f5", "name": "Gabriela"}
APPEARANCE = (
    "photorealistic beautiful blonde Mexican woman in her mid-20s, "
    "thick curvy hourglass figure, smiling at the camera in a sunlit cafe"
)


def test_character_caption_is_only_the_name_and_time() -> None:
    job = {
        "character_id": "a0f5",
        "prompt": APPEARANCE,
        "generation_seconds": 17.2,
    }

    assert image_caption(job, [GABRIELA]) == "Gabriela\n⏱ 17.2s"


def test_prompt_only_caption_keeps_the_prompt() -> None:
    job = {"prompt": "a cat", "generation_seconds": 12.3}

    assert image_caption(job, [GABRIELA]) == "a cat\n⏱ 12.3s"


def test_caption_without_seconds_has_no_timing() -> None:
    job = {"character_id": "a0f5", "prompt": APPEARANCE}

    assert image_caption(job, [GABRIELA]) == "Gabriela"


def test_caption_falls_back_to_job_timestamps() -> None:
    job = {
        "character_id": "a0f5",
        "prompt": APPEARANCE,
        "started_at": 100.0,
        "updated_at": 104.5,
    }

    assert image_caption(job, [GABRIELA]) == "Gabriela\n⏱ 4.5s"


def test_unknown_character_id_still_hides_the_prompt() -> None:
    job = {"character_id": "gone", "prompt": APPEARANCE}

    assert character_name([GABRIELA], job) == "Character"
    assert "photorealistic" not in image_caption(job, [GABRIELA])


def test_status_label_uses_the_character_name() -> None:
    job = {"character_id": "a0f5", "prompt": APPEARANCE}

    assert job_label(job, [GABRIELA]) == "Gabriela"


def test_status_label_keeps_the_prompt_for_prompt_jobs() -> None:
    assert job_label({"prompt": "a rooftop at sunset"}, [GABRIELA]) == "a rooftop at sunset"

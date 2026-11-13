"""Synthetic character bootstrap: scene list and captions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TRIGGER = "gbrla"

FRAMINGS = [
    "close-up portrait, neutral expression, looking at the camera",
    "close-up portrait, soft smile, looking at the camera",
    "three-quarter view, looking slightly left, natural expression",
    "three-quarter view, looking slightly right, confident smile",
    "side profile, looking away, calm expression",
    "half-body, arms crossed, relaxed smile",
    "half-body, holding a coffee cup, candid laugh",
    "full-body, walking toward the camera, smiling",
    "full-body, standing with one hand in her pocket, relaxed",
    "full-body, sitting on a park bench, looking at the camera",
]

SETTINGS = [
    ("wearing a white linen shirt and blue jeans", "sunlit kitchen, morning light"),
    ("wearing a beige knit sweater and dark trousers", "cozy living room, soft window light"),
    ("wearing a floral summer dress", "city street at golden hour"),
]


@dataclass(frozen=True, slots=True)
class Scene:
    id: str
    description: str
    prompt: str

    def caption(self, trigger: str = TRIGGER) -> str:
        return f"{trigger}, {self.description}"


def default_scenes() -> list[Scene]:
    scenes: list[Scene] = []
    for setting_index, (outfit, setting) in enumerate(SETTINGS, 1):
        for framing_index, framing in enumerate(FRAMINGS, 1):
            description = f"{framing}, {outfit}, {setting}"
            prompt = (
                "Keep the same woman's face, hair, skin tone and identity exactly. "
                f"Change the scene to: she is {description}, "
                "candid Instagram lifestyle photo, fully clothed."
            )
            scenes.append(
                Scene(f"{setting_index:02d}_{framing_index:02d}", description, prompt)
            )
    return scenes


def load_scenes(path: str | None) -> list[Scene]:
    if not path:
        return default_scenes()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Scene(item["id"], item["description"], item["prompt"]) for item in data]


def write_manifest(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

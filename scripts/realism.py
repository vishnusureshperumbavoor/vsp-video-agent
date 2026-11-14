"""Prompt recipes for photoreal output.

Diffusion models default to a polished, slightly plastic look. These helpers add
the cues that push toward a photograph — and flag the words that pull the other
way. Used by the H3 video CLI and documented in ``docs/ai-realism.md``.

Keep this pure: no torch, no HTTP, no model imports.
"""

from __future__ import annotations

STILL_SUFFIX = (
    "raw candid photograph, 85mm lens, f/1.8, natural skin texture, "
    "visible pores, subtle freckles, film grain, soft directional window light, "
    "no retouching"
)
VIDEO_SUFFIX = (
    "shot on 35mm film, shallow depth of field, natural skin texture, "
    "practical lighting, subtle grain, handheld micro-movement, no digital smoothing"
)
DIALOGUE_TAG = "<d>[{language}] {line}</d>"

# Words that reliably make output look generated.
PLASTIC_WORDS = (
    "instagram",
    "airbrushed",
    "smooth skin",
    "perfect skin",
    "beauty filter",
    "hyperrealistic",
    "8k",
    "ultra hd",
    "glossy",
)

LIGHT_CUES = ("light", "lens", "film", "grain", "pores", "texture", "shadow")


def dialogue(line: str, language: str = "English") -> str:
    """Wrap a spoken line in H3's dialogue tag.

    H3 only renders speech it is told about explicitly; prose like "she talks"
    produces mouth movement and room tone, not words.
    """
    text = line.strip().strip('"').strip()
    if not text:
        raise ValueError("dialogue is empty")
    if "</d>" in text or "<d>" in text:
        raise ValueError("dialogue already contains a tag")
    return DIALOGUE_TAG.format(language=language, line=text)


def enhance(prompt: str, kind: str = "still") -> str:
    """Append the photorealism suffix unless the prompt already has one."""
    suffix = STILL_SUFFIX if kind == "still" else VIDEO_SUFFIX
    lowered = prompt.lower()
    if "raw candid photograph" in lowered or "shot on 35mm film" in lowered:
        return prompt
    return f"{prompt.rstrip()} {suffix}"


def audit(prompt: str) -> list[str]:
    """Return short warnings about cues that hurt realism."""
    lowered = prompt.lower()
    warnings = [
        f"remove {word!r}: it pushes the AI-portrait look" for word in PLASTIC_WORDS if word in lowered
    ]
    if not any(cue in lowered for cue in LIGHT_CUES):
        warnings.append("add light/lens/texture cues (or --raw)")
    if "fully clothed" not in lowered and "clothed" not in lowered:
        warnings.append("state clothing explicitly to avoid the model drifting")
    return warnings

"""Generate a synthetic reference sheet with FLUX.2 klein edits."""

from __future__ import annotations

from pathlib import Path

from gpu_client import request_image
from reference_edit import build_edit_payload, encode_reference

from character.scenes import Scene, write_manifest


def generate_sheet(
    reference: str,
    scenes: list[Scene],
    out_dir: str,
    url: str,
    token: str,
    size: int = 1024,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
    timeout: float = 1800.0,
    trigger: str = "gbrla",
    resume: bool = False,
) -> list[dict]:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    encoded = encode_reference(reference)
    entries: list[dict] = []

    for index, scene in enumerate(scenes):
        image_path = target_dir / f"{scene.id}.png"
        caption = scene.caption(trigger)
        (target_dir / f"{scene.id}.txt").write_text(caption + "\n", encoding="utf-8")
        if resume and image_path.exists():
            entries.append(
                {
                    "id": scene.id,
                    "file": image_path.name,
                    "caption": caption,
                    "skipped": True,
                }
            )
            continue
        payload = build_edit_payload(
            scene.prompt, encoded, size, steps, guidance, seed + index
        )
        elapsed, raw = request_image(url, token, "/edit", payload, timeout)
        image_path.write_bytes(raw)
        entries.append(
            {
                "id": scene.id,
                "file": image_path.name,
                "caption": caption,
                "prompt": scene.prompt,
                "seed": seed + index,
                "seconds": round(elapsed, 1),
            }
        )
        print(f"{scene.id}: {elapsed:.1f}s -> {image_path}", flush=True)

    write_manifest(
        target_dir / "manifest.json",
        {
            "reference": str(reference),
            "size": size,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "entries": entries,
        },
    )
    return entries

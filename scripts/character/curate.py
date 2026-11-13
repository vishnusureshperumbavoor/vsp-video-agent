"""Face-embedding curation for a synthetic character sheet."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _load_insightface():
    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "Install curation deps first: "
            "pip install insightface onnxruntime opencv-python-headless"
        ) from exc
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app, cv2, np


def _embedding(app, cv2, np, path: Path) -> list[float] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        return None
    face = max(
        faces,
        key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
    )
    return np.asarray(face.normed_embedding, dtype=float).tolist()


def curate(
    input_dir: str,
    reference: str,
    out_dir: str,
    threshold: float = 0.45,
    dedupe: float = 0.93,
) -> list[dict]:
    app, cv2, np = _load_insightface()
    source = Path(input_dir)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    reference_embedding = _embedding(app, cv2, np, Path(reference))
    if reference_embedding is None:
        raise SystemExit(f"no face found in reference image: {reference}")

    report: list[dict] = []
    kept: list[tuple[list[float], float]] = []
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        embedding = _embedding(app, cv2, np, path)
        if embedding is None:
            report.append({"image": path.name, "kept": False, "reason": "no face"})
            continue
        score = cosine(reference_embedding, embedding)
        if score < threshold:
            report.append(
                {"image": path.name, "kept": False, "score": round(score, 4), "reason": "low identity"}
            )
            continue
        if any(cosine(embedding, other) > dedupe for other, _ in kept):
            report.append(
                {"image": path.name, "kept": False, "score": round(score, 4), "reason": "duplicate"}
            )
            continue
        kept.append((embedding, score))
        shutil.copy2(path, target / path.name)
        caption = path.with_suffix(".txt")
        if caption.exists():
            shutil.copy2(caption, target / caption.name)
        report.append(
            {"image": path.name, "kept": True, "score": round(score, 4), "reason": "ok"}
        )

    (target / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"kept {sum(1 for item in report if item['kept'])} of {len(report)}", flush=True)
    return report

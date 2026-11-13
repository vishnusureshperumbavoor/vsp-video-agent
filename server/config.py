"""Filesystem locations shared by the API, services and storage.

Keeping these in one module lets ``server/main.py`` stay wiring-only and lets
tests point the whole product at a temporary directory.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"
VIDEOS_DIR = PUBLIC_DIR / "videos"
UPLOADS_DIR = PUBLIC_DIR / "uploads"
IMAGES_DIR = PUBLIC_DIR / "images"
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = VIDEOS_DIR / "metadata.json"

for _directory in (VIDEOS_DIR, UPLOADS_DIR, IMAGES_DIR, DATA_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

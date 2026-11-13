"""SQLite persistence for users, characters and image jobs.

One connection per call, WAL mode, and a lock around writes. That is enough for
a single-process beta and keeps the repository dependency-free; swap the Store
for Postgres when the app runs on more than one worker.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path

from server.domain.character import Character
from server.domain.image_job import ACTIVE_STATUSES, ImageJob

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    telegram_id INTEGER PRIMARY KEY,
    active_character_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    owner_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    appearance TEXT NOT NULL,
    style TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS characters_owner ON characters (owner_id, created_at);
CREATE TABLE IF NOT EXISTS image_jobs (
    id TEXT PRIMARY KEY,
    owner_id INTEGER NOT NULL,
    character_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    size INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    image_url TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    generation_seconds REAL
);
CREATE INDEX IF NOT EXISTS image_jobs_owner ON image_jobs (owner_id, created_at);
CREATE INDEX IF NOT EXISTS image_jobs_status ON image_jobs (status, created_at);
"""


class Store:
    """A tiny repository; domain objects go in and come out."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after the first beta database was created."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(image_jobs)")}
        for name in ("started_at", "generation_seconds"):
            if name not in columns:
                conn.execute(f"ALTER TABLE image_jobs ADD COLUMN {name} REAL")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connection(self):
        with self._lock, closing(self._connect()) as conn, conn:
            yield conn

    # ------------------------------------------------------------------ users
    def ensure_user(self, owner_id: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (telegram_id, created_at) VALUES (?, ?)",
                (owner_id, time.time()),
            )

    def set_active_character(self, owner_id: int, character_id: str) -> None:
        self.ensure_user(owner_id)
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET active_character_id = ? WHERE telegram_id = ?",
                (character_id, owner_id),
            )

    def clear_active_character(self, owner_id: int) -> None:
        self.ensure_user(owner_id)
        with self._connection() as conn:
            conn.execute(
                "UPDATE users SET active_character_id = NULL WHERE telegram_id = ?",
                (owner_id,),
            )

    def get_active_character_id(self, owner_id: int) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT active_character_id FROM users WHERE telegram_id = ?",
                (owner_id,),
            ).fetchone()
        return row["active_character_id"] if row else None

    # ------------------------------------------------------------- characters
    def create_character(self, character: Character) -> None:
        self.ensure_user(character.owner_id)
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO characters "
                "(id, owner_id, name, appearance, style, negative_prompt, seed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    character.id,
                    character.owner_id,
                    character.name,
                    character.appearance,
                    character.style,
                    character.negative_prompt,
                    character.seed,
                    character.created_at,
                ),
            )

    def get_character(self, owner_id: int, character_id: str) -> Character | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM characters WHERE id = ? AND owner_id = ?",
                (character_id, owner_id),
            ).fetchone()
        return Character.from_row(row) if row else None

    def list_characters(self, owner_id: int) -> list[Character]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM characters WHERE owner_id = ? ORDER BY created_at",
                (owner_id,),
            ).fetchall()
        return [Character.from_row(row) for row in rows]

    # ------------------------------------------------------------------- jobs
    def create_job(self, job: ImageJob) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO image_jobs "
                "(id, owner_id, character_id, prompt, size, steps, seed, status, "
                "image_url, error, created_at, updated_at, started_at, generation_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.owner_id,
                    job.character_id,
                    job.prompt,
                    job.size,
                    job.steps,
                    job.seed,
                    job.status,
                    job.image_url,
                    job.error,
                    job.created_at,
                    job.updated_at,
                    job.started_at,
                    job.generation_seconds,
                ),
            )

    def update_job(self, job: ImageJob) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE image_jobs SET status = ?, image_url = ?, error = ?, updated_at = ?, "
                "started_at = ?, generation_seconds = ? WHERE id = ?",
                (
                    job.status,
                    job.image_url,
                    job.error,
                    job.updated_at,
                    job.started_at,
                    job.generation_seconds,
                    job.id,
                ),
            )

    def get_job(self, job_id: str) -> ImageJob | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM image_jobs WHERE id = ?", (job_id,)).fetchone()
        return ImageJob.from_row(row) if row else None

    def get_job_for(self, owner_id: int, job_id: str) -> ImageJob | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_jobs WHERE id = ? AND owner_id = ?",
                (job_id, owner_id),
            ).fetchone()
        return ImageJob.from_row(row) if row else None

    def list_jobs(self, owner_id: int, limit: int = 20) -> list[ImageJob]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_jobs WHERE owner_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (owner_id, int(limit)),
            ).fetchall()
        return [ImageJob.from_row(row) for row in rows]

    def count_jobs_since(self, owner_id: int, since: float) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM image_jobs "
                "WHERE owner_id = ? AND created_at >= ?",
                (owner_id, since),
            ).fetchone()
        return int(row["total"])

    def count_all_jobs_since(self, since: float) -> int:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM image_jobs WHERE created_at >= ?",
                (since,),
            ).fetchone()
        return int(row["total"])

    def pending_job_ids(self) -> list[str]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT id FROM image_jobs WHERE status IN ({placeholders}) "
                "ORDER BY created_at",
                ACTIVE_STATUSES,
            ).fetchall()
        return [row["id"] for row in rows]

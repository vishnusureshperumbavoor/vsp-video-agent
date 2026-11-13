"""Application service for the image product.

Owns the single-GPU queue, the per-user quota and the job lifecycle. Routers
call this; this calls the Store, the GPU transport and the image storage.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from server.domain.character import Character
from server.domain.image_job import ImageJob
from server.settings import Settings
from server.storage import ImageStorage
from server.store import Store
from server.transport.image_api import ImageAPIClient

DAY_SECONDS = 86400


class ImageServiceError(RuntimeError):
    """Base class for the errors an API can map to a status code."""


class NotFound(ImageServiceError):
    """The character or job does not exist for this owner."""


class QuotaExceeded(ImageServiceError):
    """The user has hit the daily image cap."""


class ImageJobService:
    def __init__(
        self,
        store: Store,
        api: ImageAPIClient,
        storage: ImageStorage,
        settings: Settings,
    ) -> None:
        self._store = store
        self._api = api
        self._storage = storage
        self._settings = settings
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._resumed = False

    # ------------------------------------------------------------- characters
    def create_character(
        self,
        owner_id: int,
        name: str,
        appearance: str,
        style: str = "",
        negative_prompt: str = "",
        seed: int = 42,
    ) -> Character:
        character = Character.create(
            owner_id=owner_id,
            name=name,
            appearance=appearance,
            style=style,
            negative_prompt=negative_prompt,
            seed=seed,
            created_at=time.time(),
        )
        self._store.create_character(character)
        return character

    def list_characters(self, owner_id: int) -> list[Character]:
        return self._store.list_characters(owner_id)

    def get_character(self, owner_id: int, character_id: str) -> Character:
        character = self._store.get_character(owner_id, character_id)
        if character is None:
            raise NotFound("character not found")
        return character

    def select_character(self, owner_id: int, character_id: str) -> Character:
        character = self.get_character(owner_id, character_id)
        self._store.set_active_character(owner_id, character.id)
        return character

    def get_active_character(self, owner_id: int) -> Character:
        character_id = self._store.get_active_character_id(owner_id)
        if not character_id:
            raise NotFound("no active character")
        return self.get_character(owner_id, character_id)

    # ------------------------------------------------------------------- jobs
    def create_job(
        self,
        owner_id: int,
        scene: str,
        character_id: str | None = None,
        size: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
        dispatch: bool | None = None,
    ) -> ImageJob:
        self._resume_once()
        character = self._resolve_character(owner_id, character_id)
        self._check_quota(owner_id)
        job = ImageJob.create(
            owner_id=owner_id,
            character_id=character.id,
            prompt=character.compose_prompt(scene),
            size=size,
            steps=self._settings.default_steps if steps is None else steps,
            seed=character.seed if seed is None else seed,
        )
        self._store.create_job(job)
        should_dispatch = self._settings.auto_dispatch if dispatch is None else dispatch
        if should_dispatch:
            self.dispatch(job.id)
        return job

    def get_job(self, owner_id: int, job_id: str) -> ImageJob:
        job = self._store.get_job_for(owner_id, job_id)
        if job is None:
            raise NotFound("job not found")
        return job

    def list_jobs(self, owner_id: int, limit: int = 20) -> list[ImageJob]:
        return self._store.list_jobs(owner_id, limit)

    async def process(self, job_id: str) -> None:
        """Run one queued job to completion. Safe to call directly in tests."""
        job = self._store.get_job(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return
        job.mark_running()
        self._store.update_job(job)
        try:
            character = self._store.get_character(job.owner_id, job.character_id)
            negative = character.negative_prompt if character else ""
            data = await asyncio.to_thread(
                self._api.generate,
                job.prompt,
                job.size,
                job.steps,
                job.seed,
                negative,
            )
            job.mark_completed(self._storage.save(job.owner_id, job.id, data))
        except Exception as exc:  # job boundary: every failure is user-visible
            job.mark_failed(str(exc)[:300] or exc.__class__.__name__)
        self._store.update_job(job)

    def dispatch(self, job_id: str) -> None:
        self._resume_once(skip=job_id)
        self._queue.put_nowait(job_id)
        self._start_worker()

    def start(self) -> None:
        """Requeue jobs left behind by a restart and start the single worker."""
        self._resume_once()
        self._start_worker()

    def _start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.worker())

    async def worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self.process(job_id)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def storage_path(self, job: ImageJob) -> Path | None:
        """Where a completed job's file lives, when the storage is local."""
        root = getattr(self._storage, "root", None)
        if root is None or not job.image_url:
            return None
        return Path(root) / str(job.owner_id) / f"{job.id}.png"

    # ---------------------------------------------------------------- helpers
    def _resolve_character(self, owner_id: int, character_id: str | None) -> Character:
        wanted = character_id or self._store.get_active_character_id(owner_id)
        if not wanted:
            raise NotFound("create and select a character first")
        return self.get_character(owner_id, wanted)

    def _check_quota(self, owner_id: int) -> None:
        limit = self._settings.daily_image_limit
        if limit <= 0:
            return
        used = self._store.count_jobs_since(owner_id, time.time() - DAY_SECONDS)
        if used >= limit:
            raise QuotaExceeded(f"daily limit of {limit} images reached")

    def _resume_once(self, skip: str | None = None) -> None:
        """Requeue jobs that were left queued or running by a restart."""
        if self._resumed:
            return
        self._resumed = True
        for job_id in self._store.pending_job_ids():
            if job_id != skip:
                self._queue.put_nowait(job_id)

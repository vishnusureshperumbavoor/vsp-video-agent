"""Image generation endpoints: create a job, poll it, list a user's gallery."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from server.domain.character import InvalidCharacter
from server.domain.image_job import InvalidJob
from server.runtime import get_image_service
from server.security import require_user
from server.services.image_jobs import ImageJobService, NotFound, QuotaExceeded

router = APIRouter(prefix="/api/images", tags=["images"])

User = Annotated[int, Depends(require_user)]
Service = Annotated[ImageJobService, Depends(get_image_service)]


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    character_id: str | None = None
    size: int = Field(default=1024, ge=256, le=1536)
    steps: int | None = Field(default=None, ge=1, le=50)
    seed: int | None = None


@router.post("", status_code=201)
async def create_image(request: ImageRequest, owner_id: User, service: Service) -> dict:
    try:
        job = service.create_job(
            owner_id=owner_id,
            scene=request.prompt,
            character_id=request.character_id,
            size=request.size,
            steps=request.steps,
            seed=request.seed,
        )
    except (InvalidCharacter, InvalidJob) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return job.to_dict()


@router.get("")
def list_images(
    owner_id: User,
    service: Service,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    return [job.to_dict() for job in service.list_jobs(owner_id, limit)]


@router.get("/{job_id}")
def get_image(job_id: str, owner_id: User, service: Service) -> dict:
    try:
        return service.get_job(owner_id, job_id).to_dict()
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

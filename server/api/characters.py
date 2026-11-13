"""Character endpoints for the Telegram-facing image product."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from server.domain.character import InvalidCharacter
from server.runtime import get_image_service
from server.security import require_user
from server.services.image_jobs import ImageJobService, NotFound

router = APIRouter(prefix="/api/characters", tags=["characters"])

User = Annotated[int, Depends(require_user)]
Service = Annotated[ImageJobService, Depends(get_image_service)]


class CharacterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    appearance: str = Field(min_length=1, max_length=1000)
    style: str = Field(default="", max_length=1000)
    negative_prompt: str = Field(default="", max_length=1000)
    seed: int = 42


@router.post("")
def create_character(
    request: CharacterRequest,
    owner_id: User,
    service: Service,
) -> dict:
    try:
        character = service.create_character(owner_id=owner_id, **request.model_dump())
    except InvalidCharacter as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return character.to_dict()


@router.get("")
def list_characters(owner_id: User, service: Service) -> list[dict]:
    return [character.to_dict() for character in service.list_characters(owner_id)]


@router.get("/active")
def active_character(owner_id: User, service: Service) -> dict:
    try:
        return service.get_active_character(owner_id).to_dict()
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/active")
def clear_active_character(owner_id: User, service: Service) -> dict:
    service.clear_character(owner_id)
    return {"active": None}


@router.post("/{character_id}/select")
def select_character(character_id: str, owner_id: User, service: Service) -> dict:
    try:
        return service.select_character(owner_id, character_id).to_dict()
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

#!/usr/bin/env python3
"""GPU-side image service: one warm model, one request at a time.

Runs on the JarvisLabs A100 container and is called only by the backend, with a
bearer token. Keep it behind that token: the JarvisLabs endpoint is public.

    IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-SFT \
    IMAGE_CUSTOM_PIPELINE=pipeline_moss_image IMAGE_API_TOKEN=... \
      uvicorn image_server:app --host 0.0.0.0 --port 6006

For FLUX, drop ``IMAGE_CUSTOM_PIPELINE`` and set the model id to a diffusers
repo such as ``black-forest-labs/FLUX.1-schnell``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "")
CUSTOM_PIPELINE = os.environ.get("IMAGE_CUSTOM_PIPELINE", "")
API_TOKEN = os.environ.get("IMAGE_API_TOKEN", "")
NO_OFFLOAD = os.environ.get("IMAGE_NO_OFFLOAD", "") == "1"
SKIP_LAYERS = [
    int(value)
    for value in os.environ.get("IMAGE_SKIP_LAYERS", "22").split(",")
    if value.strip()
]

pipe = None
lock = asyncio.Lock()


def load_pipeline():
    """Import the heavy stack only on the GPU box, never at module import."""
    import torch
    from diffusers import DiffusionPipeline

    if not MODEL_ID:
        raise RuntimeError("set IMAGE_MODEL_ID to a Hugging Face model id")
    dtype = torch.bfloat16
    if os.environ.get("IMAGE_FP32_VAE", "") == "1":
        dtype = {"default": torch.bfloat16, "vae": torch.float32}
    options = {"torch_dtype": dtype}
    if CUSTOM_PIPELINE:
        options.update(custom_pipeline=CUSTOM_PIPELINE, trust_remote_code=True)
    pipeline = DiffusionPipeline.from_pretrained(MODEL_ID, **options)
    if NO_OFFLOAD:
        pipeline.to("cuda")
    else:
        pipeline.enable_model_cpu_offload()
    return pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not API_TOKEN:
        raise RuntimeError("set IMAGE_API_TOKEN before starting the service")
    global pipe
    pipe = await asyncio.to_thread(load_pipeline)
    yield
    pipe = None


app = FastAPI(title="Girlfriend image service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)
    steps: int = Field(default=4, ge=1, le=50)
    seed: int = 42


def _validate(request: GenerateRequest) -> None:
    if request.width % 16 or request.height % 16:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16")


def _render(request: GenerateRequest) -> bytes:
    import torch

    kwargs = {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": request.steps,
        "generator": torch.Generator("cpu").manual_seed(request.seed),
    }
    parameters = inspect.signature(pipe.__call__).parameters
    if request.negative_prompt and "negative_prompt" in parameters:
        kwargs["negative_prompt"] = request.negative_prompt
    if "uncond_skip_layers" in parameters:
        kwargs["uncond_skip_layers"] = [] if request.negative_prompt else SKIP_LAYERS
    image = pipe(**kwargs).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/health")
async def health() -> dict:
    return {"ok": pipe is not None, "model": MODEL_ID}


@app.post("/generate")
async def generate(
    request: GenerateRequest,
    authorization: Annotated[str, Header()] = "",
) -> dict:
    if not secrets.compare_digest(authorization, f"Bearer {API_TOKEN}"):
        raise HTTPException(status_code=401, detail="unauthorized")
    _validate(request)
    async with lock:
        data = await asyncio.to_thread(_render, request)
    return {"image_b64": base64.b64encode(data).decode("ascii")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "6006")))

import os
import sys
import uuid
import time
import json
import base64
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import vspagent Colab router
try:
    from vspagent.colab.integrations.fastapi import get_colab_router
    from vspagent.colab.client import internal_colab_client
    VSPAGENT_AVAILABLE = True
except Exception as e:
    print(f"Warning: vspagent import note: {e}")
    VSPAGENT_AVAILABLE = False
    internal_colab_client = None

from server.wan_worker import (
    generate_colab_setup_code,
    generate_t2v_execution_code,
    generate_i2v_execution_code
)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "public" / "videos"
UPLOADS_DIR = BASE_DIR / "public" / "uploads"
METADATA_FILE = VIDEOS_DIR / "metadata.json"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Vibe Video Studio API",
    description="Video Generation Platform with React, FastAPI, and vspagent-powered A100 GPU",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folders for video playback & uploads
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

if VSPAGENT_AVAILABLE:
    app.include_router(get_colab_router())

# In-memory job tracker
jobs: Dict[str, Dict[str, Any]] = {}

def load_metadata() -> List[Dict[str, Any]]:
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_metadata(items: List[Dict[str, Any]]):
    with open(METADATA_FILE, "w") as f:
        json.dump(items, f, indent=2)

class T2VRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the scene and motion")
    negative_prompt: str = Field(default="low quality, blurry, distorted, artifacts", description="Negative prompt")
    aspect_ratio: str = Field(default="16:9", description="16:9, 9:16, or 1:1")
    duration: int = Field(default=5, description="Duration in seconds (approx 16 fps)")
    steps: int = Field(default=30, description="Inference steps")
    seed: int = Field(default=42, description="Random seed")
    model_size: str = Field(default="1.3B", description="1.3B or 14B")

class StitchRequest(BaseModel):
    video_filenames: List[str] = Field(..., description="Ordered list of video files to concatenate")
    output_filename: Optional[str] = Field(default="final_movie.mp4")

@app.get("/api/health")
async def health_check():
    colab_status = internal_colab_client.status if internal_colab_client else "unknown"
    return {
        "status": "online",
        "service": "Vibe Video Studio",
        "vspagent": "ready" if VSPAGENT_AVAILABLE else "unavailable",
        "colab_status": colab_status,
        "videos_count": len(load_metadata())
    }

@app.get("/api/video/gallery")
async def get_gallery():
    meta_list = load_metadata()
    # Scan filesystem to ensure consistency
    files = list(VIDEOS_DIR.glob("*.mp4"))
    file_map = {m.get("filename"): m for m in meta_list if "filename" in m}
    
    gallery = []
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        fn = f.name
        info = file_map.get(fn, {})
        gallery.append({
            "filename": fn,
            "url": f"/videos/{fn}",
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "created_at": f.stat().st_mtime,
            "prompt": info.get("prompt", fn.replace(".mp4", "").replace("_", " ")),
            "aspect_ratio": info.get("aspect_ratio", "16:9"),
            "duration": info.get("duration", 5),
            "type": info.get("type", "T2V")
        })
    return gallery

@app.get("/api/video/jobs")
async def list_jobs():
    return list(jobs.values())

@app.get("/api/video/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def run_video_job(job_id: str, code: str, metadata: Dict[str, Any]):
    job = jobs[job_id]
    job["status"] = "running"
    job["logs"].append("Dispatching execution payload to Colab A100 GPU...")
    
    if not internal_colab_client or internal_colab_client.status != "connected":
        job["status"] = "failed"
        job["error"] = "Colab runtime is not connected. Please connect via /colab/connect first."
        return

    try:
        res = await internal_colab_client.execute_code(code=code, keep_cell=True, timeout=1200.0)
        output_text = res.get("output", "")
        job["logs"].append(f"Execution finished. Output size: {len(output_text)} chars")

        # Parse base64 video output
        if "__VSP_VIDEO_OUTPUT_START__" in output_text and "__VSP_VIDEO_OUTPUT_END__" in output_text:
            b64_segment = output_text.split("__VSP_VIDEO_OUTPUT_START__")[1].split("__VSP_VIDEO_OUTPUT_END__")[0].strip()
            video_bytes = base64.b64decode(b64_segment)
            
            output_filename = metadata["filename"]
            target_path = VIDEOS_DIR / output_filename
            with open(target_path, "wb") as f:
                f.write(video_bytes)
            
            # Save metadata
            meta_list = load_metadata()
            meta_list.insert(0, metadata)
            save_metadata(meta_list)

            job["status"] = "completed"
            job["video_url"] = f"/videos/{output_filename}"
            job["logs"].append(f"Video saved successfully: {output_filename} ({len(video_bytes)/(1024*1024):.2f} MB)")
        else:
            job["status"] = "failed"
            job["error"] = "Video generation script executed, but video output marker was not found in Colab output."
            job["logs"].append(f"Tail log: {output_text[-500:]}")
            
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["logs"].append(f"Error during dispatch: {e}")

@app.post("/api/video/setup-colab")
async def setup_colab(bg: BackgroundTasks):
    job_id = f"setup-{uuid.uuid4().hex[:6]}"
    jobs[job_id] = {
        "id": job_id,
        "type": "SETUP",
        "status": "queued",
        "created_at": time.time(),
        "logs": ["Queued Colab environment setup..."]
    }
    code = generate_colab_setup_code()
    bg.add_task(run_video_job, job_id, code, {"filename": "setup.log"})
    return {"job_id": job_id, "status": "queued"}

@app.post("/api/video/generate/t2v")
async def generate_t2v(req: T2VRequest, bg: BackgroundTasks):
    job_id = f"t2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"t2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"
    
    # Map aspect ratio to pixel dimensions (multiples of 16/32 for video diffusion)
    dimensions = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640)
    }
    width, height = dimensions.get(req.aspect_ratio, (832, 480))
    num_frames = max(49, req.duration * 16) # 16 fps
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" if req.model_size == "1.3B" else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    
    code = generate_t2v_execution_code(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=req.steps,
        seed=req.seed,
        model_id=model_id,
        output_filename=out_filename
    )
    
    metadata = {
        "filename": out_filename,
        "prompt": req.prompt,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration,
        "model_size": req.model_size,
        "type": "T2V",
        "created_at": time.time()
    }
    
    jobs[job_id] = {
        "id": job_id,
        "type": "T2V",
        "status": "queued",
        "created_at": time.time(),
        "prompt": req.prompt,
        "metadata": metadata,
        "logs": [f"Job queued for prompt: '{req.prompt}'"]
    }
    
    bg.add_task(run_video_job, job_id, code, metadata)
    return {"job_id": job_id, "status": "queued", "filename": out_filename}

@app.post("/api/video/generate/i2v")
async def generate_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("low quality, blurry, distorted, artifacts"),
    aspect_ratio: str = Form("16:9"),
    duration: int = Form(5),
    steps: int = Form(30),
    seed: int = Form(42),
    bg: BackgroundTasks = BackgroundTasks()
):
    job_id = f"i2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"i2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"
    
    # Save input image locally
    img_bytes = await image.read()
    img_ext = Path(image.filename or "image.jpg").suffix or ".jpg"
    img_filename = f"input_{uuid.uuid4().hex[:8]}{img_ext}"
    img_path = UPLOADS_DIR / img_filename
    with open(img_path, "wb") as f:
        f.write(img_bytes)
        
    b64_image = base64.b64encode(img_bytes).decode("utf-8")
    
    dimensions = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640)
    }
    width, height = dimensions.get(aspect_ratio, (832, 480))
    num_frames = max(49, duration * 16)
    
    code = generate_i2v_execution_code(
        image_b64=b64_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        seed=seed,
        model_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        output_filename=out_filename
    )
    
    metadata = {
        "filename": out_filename,
        "input_image": f"/uploads/{img_filename}",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "type": "I2V",
        "created_at": time.time()
    }
    
    jobs[job_id] = {
        "id": job_id,
        "type": "I2V",
        "status": "queued",
        "created_at": time.time(),
        "prompt": prompt,
        "metadata": metadata,
        "logs": [f"I2V job queued with reference image: '{image.filename}'"]
    }
    
    bg.add_task(run_video_job, job_id, code, metadata)
    return {"job_id": job_id, "status": "queued", "filename": out_filename}

@app.post("/api/video/stitch")
async def stitch_videos(req: StitchRequest):
    if len(req.video_filenames) < 2:
        raise HTTPException(status_code=400, detail="At least 2 videos are required to stitch.")

    # Create temporary concat list for ffmpeg
    list_file_path = VIDEOS_DIR / f"concat_{uuid.uuid4().hex[:6]}.txt"
    final_output = req.output_filename or f"movie_{int(time.time())}.mp4"
    target_path = VIDEOS_DIR / final_output

    with open(list_file_path, "w") as f:
        for fn in req.video_filenames:
            video_file = VIDEOS_DIR / fn
            if video_file.exists():
                f.write(f"file '{video_file.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file_path),
        "-c", "copy",
        str(target_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        if list_file_path.exists():
            list_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"FFmpeg stitching failed: {e}")

    if list_file_path.exists():
        list_file_path.unlink()

    # Save to metadata
    meta_list = load_metadata()
    meta_list.insert(0, {
        "filename": final_output,
        "prompt": f"Stitched Movie ({len(req.video_filenames)} scenes)",
        "duration": len(req.video_filenames) * 5,
        "type": "MOVIE",
        "created_at": time.time()
    })
    save_metadata(meta_list)

    return {
        "status": "success",
        "filename": final_output,
        "url": f"/videos/{final_output}",
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)

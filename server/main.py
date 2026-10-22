import os
import sys
import uuid
import time
import json
import base64
import asyncio
import subprocess
import re
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
    generate_cache_cleanup_code,
    generate_t2v_script,
    generate_i2v_script
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

if VSPAGENT_AVAILABLE:
    app.include_router(get_colab_router(), prefix="/colab", tags=["Google Colab"])

app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

jobs: Dict[str, Dict[str, Any]] = {}

class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, blurry, distorted, flickering, artifacts"
    aspect_ratio: str = Field(default="16:9", description="16:9, 9:16, 1:1")
    duration: int = Field(default=5, description="Video duration in seconds (1-10)")
    steps: int = Field(default=25, description="Inference steps (15-50)")
    seed: int = Field(default=42, description="Random seed")
    model_size: str = Field(default="1.3B", description="1.3B (fast) or 14B (cinematic)")

class StitchRequest(BaseModel):
    video_filenames: List[str]
    output_filename: Optional[str] = None

def load_metadata() -> List[Dict[str, Any]]:
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_metadata(data: List[Dict[str, Any]]):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "vspagent_available": VSPAGENT_AVAILABLE,
        "colab_connected": internal_colab_client.is_connected if internal_colab_client else False
    }

@app.get("/api/video/gallery")
async def get_gallery():
    metadata = {item["filename"]: item for item in load_metadata()}
    gallery = []
    for file in sorted(VIDEOS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
        info = metadata.get(file.name, {})
        poster_name = f"{file.stem}_poster.jpg"
        has_poster = (VIDEOS_DIR / poster_name).exists()
        gallery.append({
            "filename": file.name,
            "url": f"/videos/{file.name}",
            "poster_url": f"/videos/{poster_name}" if has_poster else None,
            "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
            "created_at": file.stat().st_mtime,
            "prompt": info.get("prompt", "AI Generated Video"),
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

async def run_detached_job(
    job_id: str,
    script_content: str,
    output_filename: str,
    metadata: Dict[str, Any],
    local_upload_path: Optional[str] = None,
    remote_upload_path: Optional[str] = None
):
    """
    Production-grade asynchronous detached runner:
    1. Uploads source file in safe chunks (if I2V).
    2. Writes job script to Colab disk.
    3. Spawns job detached via subprocess.Popen (cell returns in < 0.5s).
    4. Decoupled polling loop monitors log file without blocking the notebook kernel.
    5. Downloads completed video via safe chunked transfer.
    """
    job = jobs[job_id]
    job["status"] = "running"
    job["logs"].append("Initiating detached asynchronous GPU job...")

    if not internal_colab_client or not internal_colab_client.is_connected:
        job["status"] = "failed"
        job["error"] = "Google Colab is not connected. Please pair in the studio first."
        return

    try:
        # Step 1: Upload input image if I2V
        if local_upload_path and remote_upload_path:
            job["logs"].append(f"Uploading reference image to Colab at {remote_upload_path}...")
            await internal_colab_client.upload_file_to_colab(local_upload_path, remote_upload_path)
            job["logs"].append("Reference image successfully uploaded to Colab disk.")

        # Step 2: Write script on Colab disk
        remote_script_path = f"/content/{job_id}.py"
        remote_log_path = f"/content/{job_id}.log"
        remote_video_path = f"/content/{output_filename}"

        job["logs"].append(f"Staging execution script at {remote_script_path}...")
        stage_code = f"""
import os
with open({remote_script_path!r}, 'w') as f:
    f.write({script_content!r})
print('SCRIPT_STAGED:', os.path.getsize({remote_script_path!r}))
"""
        stage_res = await internal_colab_client.execute_code(stage_code, keep_cell=False)
        if stage_res.get("isError"):
            raise RuntimeError(f"Failed staging script on Colab: {stage_res.get('output')}")

        # Step 3: Spawn detached process on Colab
        job["logs"].append("Spawning detached GPU process on A100...")
        spawn_code = f"""
import subprocess, os
if os.path.exists({remote_log_path!r}):
    os.remove({remote_log_path!r})
if os.path.exists({remote_video_path!r}):
    os.remove({remote_video_path!r})

proc = subprocess.Popen(
    ['python3', '-u', {remote_script_path!r}],
    stdout=open({remote_log_path!r}, 'w'),
    stderr=subprocess.STDOUT
)
print('DETACHED_SPAWN_PID:', proc.pid)
"""
        spawn_res = await internal_colab_client.execute_code(spawn_code, keep_cell=False)
        out = spawn_res.get("output", "")
        m = re.search(r"DETACHED_SPAWN_PID:\s*(\d+)", out)
        if not m:
            raise RuntimeError(f"Failed spawning detached process: {out}")
        pid = m.group(1)
        job["logs"].append(f"GPU generation running detached on A100 (PID {pid}). Monitoring progress...")

        # Step 4: Decoupled polling loop
        start_time = time.time()
        timeout_seconds = 1800  # 30 minutes max
        completed = False

        while time.time() - start_time < timeout_seconds:
            await asyncio.sleep(5)
            poll_code = f"""
import os
log_data = ""
if os.path.exists({remote_log_path!r}):
    with open({remote_log_path!r}) as f:
        log_data = f.read()

done = "=== JOB_SUCCESS:" in log_data
has_error = "Traceback (most recent call last):" in log_data

print("===POLL_START===")
print(f"DONE:{{done}}")
print(f"ERROR:{{has_error}}")
print("TAIL:")
print(log_data[-1200:])
print("===POLL_END===")
"""
            try:
                poll_res = await internal_colab_client.execute_code(poll_code, keep_cell=False)
                poll_out = poll_res.get("output", "")
                
                # Update latest log tail
                tail_match = re.search(r"TAIL:\n(.*?)===POLL_END===", poll_out, re.S)
                if tail_match:
                    recent_tail = tail_match.group(1).strip()
                    lines = [ln.strip() for ln in recent_tail.split("\n") if ln.strip()]
                    if lines:
                        latest_line = lines[-1]
                        if not job["logs"] or job["logs"][-1] != latest_line:
                            job["logs"].append(latest_line)

                if "DONE:True" in poll_out:
                    completed = True
                    break
                elif "ERROR:True" in poll_out:
                    job["status"] = "failed"
                    job["error"] = f"Script execution error on Colab: {poll_out[-600:]}"
                    return
            except Exception as poll_err:
                job["logs"].append(f"Polling note: {poll_err}")

        if not completed:
            job["status"] = "failed"
            job["error"] = "Job timed out after 30 minutes."
            return

        # Step 5: Download completed video from Colab
        job["logs"].append(f"Generation successful on A100! Transferring {output_filename} to studio...")
        target_path = VIDEOS_DIR / output_filename
        await internal_colab_client.download_file_from_colab(remote_video_path, str(target_path))

        # Generate poster image with ffmpeg
        poster_path = VIDEOS_DIR / f"{Path(output_filename).stem}_poster.jpg"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:00.2", "-i", str(target_path),
                "-vframes", "1", "-q:v", "2", str(poster_path)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Poster generation note: {e}")

        # Update metadata
        file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)
        metadata["size_mb"] = file_size_mb
        metadata_list = load_metadata()
        metadata_list = [item for item in metadata_list if item.get("filename") != output_filename]
        metadata_list.insert(0, metadata)
        save_metadata(metadata_list)

        job["status"] = "completed"
        job["video_url"] = f"/videos/{output_filename}"
        job["logs"].append(f"Job complete! Video ready: {output_filename} ({file_size_mb} MB)")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["logs"].append(f"Execution error: {e}")

@app.post("/api/video/setup-colab")
async def setup_colab():
    if not internal_colab_client or not internal_colab_client.is_connected:
        raise HTTPException(status_code=400, detail="Colab is not connected.")
    code = generate_colab_setup_code()
    res = await internal_colab_client.execute_code(code, keep_cell=False)
    return {"status": "success", "output": res.get("output", "")}

@app.post("/api/video/clean-cache")
async def clean_colab_cache():
    """Cleans old checkpoints from Colab to prevent the 80% disk full warning."""
    if not internal_colab_client or not internal_colab_client.is_connected:
        raise HTTPException(status_code=400, detail="Colab is not connected.")
    code = generate_cache_cleanup_code()
    res = await internal_colab_client.execute_code(code, keep_cell=False)
    return {"status": "success", "output": res.get("output", "")}

@app.post("/api/video/generate/t2v")
async def generate_t2v(req: T2VRequest, bg: BackgroundTasks):
    job_id = f"t2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"t2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"
    
    dimensions = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640)
    }
    width, height = dimensions.get(req.aspect_ratio, (832, 480))
    num_frames = max(49, req.duration * 16)
    
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" if req.model_size == "1.3B" else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    
    script_content = generate_t2v_script(
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
        "logs": [f"T2V Job queued: '{req.prompt}' ({req.model_size})"]
    }
    
    bg.add_task(run_detached_job, job_id, script_content, out_filename, metadata)
    return {"job_id": job_id, "status": "queued", "filename": out_filename}

@app.post("/api/video/generate/i2v")
async def generate_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("low quality, blurry, distorted, flickering, artifacts, bad anatomy"),
    aspect_ratio: str = Form("9:16"),
    duration: int = Form(5),
    steps: int = Form(25),
    seed: int = Form(42),
    bg: BackgroundTasks = BackgroundTasks()
):
    job_id = f"i2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"i2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"
    
    # Save input image locally
    img_bytes = await image.read()
    img_ext = Path(image.filename or "image.png").suffix or ".png"
    img_filename = f"input_{uuid.uuid4().hex[:8]}{img_ext}"
    local_img_path = UPLOADS_DIR / img_filename
    with open(local_img_path, "wb") as f:
        f.write(img_bytes)

    remote_img_path = f"/content/uploads/{img_filename}"
    
    dimensions = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640)
    }
    width, height = dimensions.get(aspect_ratio, (480, 832))
    num_frames = 17 if duration <= 2 else max(17, duration * 8)
    
    script_content = generate_i2v_script(
        remote_image_path=remote_img_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        seed=seed,
        model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        output_filename=out_filename
    )
    
    metadata = {
        "filename": out_filename,
        "input_image": f"/uploads/{img_filename}",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "type": "I2V-14B",
        "model_size": "14B",
        "created_at": time.time()
    }
    
    jobs[job_id] = {
        "id": job_id,
        "type": "I2V",
        "status": "queued",
        "created_at": time.time(),
        "prompt": prompt,
        "metadata": metadata,
        "logs": [f"I2V Job queued with reference image: '{image.filename}'"]
    }
    
    bg.add_task(
        run_detached_job,
        job_id,
        script_content,
        out_filename,
        metadata,
        str(local_img_path),
        remote_img_path
    )
    return {"job_id": job_id, "status": "queued", "filename": out_filename}

@app.post("/api/video/stitch")
async def stitch_videos(req: StitchRequest):
    if len(req.video_filenames) < 2:
        raise HTTPException(status_code=400, detail="At least 2 videos are required to stitch.")

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

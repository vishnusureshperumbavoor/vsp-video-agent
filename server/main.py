"""VSP Video Agent API — JarvisLabs-only backend.

Every GPU operation goes over SSH to a JarvisLabs instance through the vendored
`server.jarvis` harness. Drive this with `python cli.py` or the REST endpoints
below; there is no web UI.
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.gpu import REMOTE_UPLOADS, GPUError, internal_gpu_client
from server.wan_worker import (
    REMOTE_JOB_DIR,
    generate_cache_cleanup_code,
    generate_gpu_setup_code,
    generate_i2v_script,
    generate_t2v_script,
)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "public" / "videos"
UPLOADS_DIR = BASE_DIR / "public" / "uploads"
METADATA_FILE = VIDEOS_DIR / "metadata.json"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="VSP Video Agent API",
    description="Wan 2.1 video generation on a JarvisLabs A100",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


async def verify_a100() -> str:
    """Hardware guardrail: refuse to stage a job unless an A100 is attached."""
    try:
        info = await internal_gpu_client.probe()
    except GPUError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not info["connected"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "No running JarvisLabs GPU. Provision or adopt an A100 first; "
                "`python cli.py status` shows the current state."
            ),
        )

    gpu_name = info["gpu"]
    if "A100" not in gpu_name.upper():
        raise HTTPException(
            status_code=400,
            detail=(
                f"❌ STOPPING EXECUTION: NVIDIA A100 GPU is strictly required for video generation.\n"
                f"Currently detected: '{gpu_name}'. Switch the JarvisLabs instance to an A100."
            ),
        )
    return gpu_name


@app.get("/api/health")
async def health():
    try:
        info = await internal_gpu_client.probe()
    except GPUError:
        info = {"connected": False, "gpu": None, "instance_id": None}
    gpu = info.get("gpu")
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "transport": "jarvislabs-ssh",
        "gpu_connected": info.get("connected", False),
        "instance_id": info.get("instance_id"),
        "gpu": gpu,
        "a100_ready": bool(gpu and "A100" in gpu.upper()),
    }


@app.get("/api/gpu/status")
async def gpu_status():
    try:
        return await internal_gpu_client.probe()
    except GPUError as exc:
        return {"connected": False, "gpu": None, "error": str(exc)}


@app.post("/api/gpu/setup")
async def gpu_setup():
    await verify_a100()
    res = await internal_gpu_client.execute_code(generate_gpu_setup_code(), timeout=1800)
    if res.get("isError"):
        raise HTTPException(status_code=500, detail=res.get("output", "")[-2000:])
    return {"status": "success", "output": res.get("output", "")}


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
            "type": info.get("type", "T2V"),
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
    remote_upload_path: Optional[str] = None,
):
    """Stage → spawn detached → poll log → download.

    The remote process is started with `setsid`, so it survives the SSH session
    and the API can poll a log file instead of holding a connection open for the
    whole render.
    """
    job = jobs[job_id]
    job["status"] = "running"
    job["logs"].append("Staging job on the JarvisLabs A100...")

    remote_script = f"{REMOTE_JOB_DIR}/{job_id}.py"
    remote_log = f"{REMOTE_JOB_DIR}/{job_id}.log"
    remote_video = f"{REMOTE_JOB_DIR}/{output_filename}"

    try:
        if local_upload_path and remote_upload_path:
            job["logs"].append(f"Uploading reference image to {remote_upload_path}...")
            await internal_gpu_client.upload_file(local_upload_path, remote_upload_path)
            job["logs"].append("Reference image uploaded.")

        size = await internal_gpu_client.write_file(remote_script, script_content)
        job["logs"].append(f"Script staged ({size} bytes).")

        await internal_gpu_client.execute_code(
            f"import os\nfor p in ({remote_log!r}, {remote_video!r}):\n"
            f"    os.path.exists(p) and os.remove(p)\n"
        )

        job["logs"].append("Spawning detached GPU process...")
        pid = await internal_gpu_client.spawn(
            f"python3 -u {remote_script}",
            log_path=remote_log,
            workdir=REMOTE_JOB_DIR,
        )
        job["logs"].append(f"Generation running detached (PID {pid or 'unknown'}).")

        start_time = time.time()
        timeout_seconds = 1800
        completed = False

        while time.time() - start_time < timeout_seconds:
            await asyncio.sleep(5)
            log_data = await internal_gpu_client.read_log(remote_log, tail_bytes=4000)

            lines = [ln.strip() for ln in log_data.splitlines() if ln.strip()]
            if lines:
                latest = lines[-1]
                if not job["logs"] or job["logs"][-1] != latest:
                    job["logs"].append(latest)

            if "=== JOB_SUCCESS:" in log_data:
                completed = True
                break
            if "Traceback (most recent call last):" in log_data:
                job["status"] = "failed"
                job["error"] = f"Remote script failed:\n{log_data[-1200:]}"
                return

        if not completed:
            job["status"] = "failed"
            job["error"] = "Job timed out after 30 minutes."
            return

        job["logs"].append(f"Render complete. Downloading {output_filename}...")
        target_path = VIDEOS_DIR / output_filename
        ok = await internal_gpu_client.download_file(remote_video, str(target_path))
        if not ok or not target_path.exists():
            job["status"] = "failed"
            job["error"] = "Download failed; the video may still exist on the remote host."
            return

        poster_path = VIDEOS_DIR / f"{Path(output_filename).stem}_poster.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", "00:00:00.2", "-i", str(target_path),
                    "-vframes", "1", "-q:v", "2", str(poster_path),
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"Poster generation note: {exc}")

        file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)
        metadata["size_mb"] = file_size_mb
        metadata_list = [i for i in load_metadata() if i.get("filename") != output_filename]
        metadata_list.insert(0, metadata)
        save_metadata(metadata_list)

        job["status"] = "completed"
        job["video_url"] = f"/videos/{output_filename}"
        job["logs"].append(f"Done: {output_filename} ({file_size_mb} MB)")

    except GPUError as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["logs"].append(f"GPU error: {exc}")
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["logs"].append(f"Execution error: {exc}")


@app.post("/api/video/clean-cache")
async def clean_gpu_cache():
    """Drop unused checkpoints so the instance stays under its disk threshold."""
    await verify_a100()
    res = await internal_gpu_client.execute_code(generate_cache_cleanup_code(), timeout=600)
    if res.get("isError"):
        raise HTTPException(status_code=500, detail=res.get("output", "")[-2000:])
    return {"status": "success", "output": res.get("output", "")}


@app.post("/api/video/generate/t2v")
async def generate_t2v(req: T2VRequest, bg: BackgroundTasks):
    await verify_a100()

    job_id = f"t2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"t2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"

    dimensions = {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)}
    width, height = dimensions.get(req.aspect_ratio, (832, 480))
    num_frames = max(49, req.duration * 16)

    model_id = (
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        if req.model_size == "1.3B"
        else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )

    script_content = generate_t2v_script(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=req.steps,
        seed=req.seed,
        model_id=model_id,
        output_filename=out_filename,
        output_dir=REMOTE_JOB_DIR,
    )

    metadata = {
        "filename": out_filename,
        "prompt": req.prompt,
        "aspect_ratio": req.aspect_ratio,
        "duration": req.duration,
        "model_size": req.model_size,
        "type": "T2V",
        "created_at": time.time(),
    }

    jobs[job_id] = {
        "id": job_id,
        "type": "T2V",
        "status": "queued",
        "created_at": time.time(),
        "prompt": req.prompt,
        "metadata": metadata,
        "logs": [f"T2V job queued: '{req.prompt}' ({req.model_size})"],
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
    bg: BackgroundTasks = BackgroundTasks(),
):
    await verify_a100()

    job_id = f"i2v-{uuid.uuid4().hex[:6]}"
    out_filename = f"i2v_{int(time.time())}_{uuid.uuid4().hex[:4]}.mp4"

    img_bytes = await image.read()
    img_ext = Path(image.filename or "image.png").suffix or ".png"
    img_filename = f"input_{uuid.uuid4().hex[:8]}{img_ext}"
    local_img_path = UPLOADS_DIR / img_filename
    with open(local_img_path, "wb") as f:
        f.write(img_bytes)

    remote_img_path = f"{REMOTE_UPLOADS}/{img_filename}"

    dimensions = {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)}
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
        output_filename=out_filename,
        output_dir=REMOTE_JOB_DIR,
    )

    metadata = {
        "filename": out_filename,
        "input_image": f"/uploads/{img_filename}",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "type": "I2V-14B",
        "model_size": "14B",
        "created_at": time.time(),
    }

    jobs[job_id] = {
        "id": job_id,
        "type": "I2V",
        "status": "queued",
        "created_at": time.time(),
        "prompt": prompt,
        "metadata": metadata,
        "logs": [f"I2V job queued with reference image: '{image.filename}'"],
    }

    bg.add_task(
        run_detached_job,
        job_id,
        script_content,
        out_filename,
        metadata,
        str(local_img_path),
        remote_img_path,
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
        str(target_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        if list_file_path.exists():
            list_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"FFmpeg stitching failed: {exc}")

    if list_file_path.exists():
        list_file_path.unlink()

    meta_list = load_metadata()
    meta_list.insert(0, {
        "filename": final_output,
        "prompt": f"Stitched Movie ({len(req.video_filenames)} scenes)",
        "duration": len(req.video_filenames) * 5,
        "type": "MOVIE",
        "created_at": time.time(),
    })
    save_metadata(meta_list)

    return {
        "status": "success",
        "filename": final_output,
        "url": f"/videos/{final_output}",
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)

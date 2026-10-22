# 🤖 AGENTS.md — Vibe Video Studio Instruction Manual

This document provides autonomous coding agents (Antigravity, Cursor, Claude Code, terminal bots) with instructions to generate and assemble AI videos using Vibe Video Studio.

---

## 🎯 System Architecture Overview

* **Compute Worker:** Remote Google Colab instance running an **NVIDIA A100-SXM4 GPU (40 GB VRAM)**.
* **Bridge Harness:** `vspagent` Python SDK managing `colab-mcp` browser pairing and code execution.
* **Local Backend:** FastAPI server running on `http://localhost:8000`.
* **Local Frontend:** React Vite dashboard on `http://localhost:5174`.
* **Video Generation Engine:** **Wan 2.1** (`1.3B` for drafts, `14B` for final cinematic output, and `I2V-14B` for image animation).

---

## ⚡ Asynchronous Detached Execution Pattern (Fire & Poll)

When dispatching heavy diffusion jobs to Google Colab via `colab-mcp`, **never hold a synchronous blocking cell execution open for multi-minute generation**. Doing so triggers MCP timeouts and can deadlock if the browser DOM sleeps.

**The Production Pattern implemented in `server/main.py`:**
1. **Script Staging:** The backend generates a standalone script and writes it to `/content/<job_id>.py`.
2. **Safe Detached Spawn:** Colab executes `subprocess.Popen(['python3', '-u', '/content/<job_id>.py'], stdout=open('/content/<job_id>.log', 'w'))`. The cell returns in `< 0.5s`.
3. **Decoupled Polling:** The FastAPI background task polls `/content/<job_id>.log` every 5 seconds to stream progress into the UI job drawer.
4. **Chunked Staging & Download:** Large assets (images and videos) are streamed using chunked base64 slices to prevent pipe saturation.

---

## 🖥️ Colab Browser State Rules

Because `colab-mcp` bridges via Chrome extension to the Google Colab web tab:
1. **Active Tab Requirement:** The Google Colab tab in Chrome should stay in an open, un-minimized browser window so Chrome doesn't throttle background JavaScript execution.
2. **Dismiss System Modals:** Colab displays modal alerts (e.g. *"Disk is almost full"* or *"Runtime warning"*). When any modal backdrop is active in Colab, the web UI intercepts all clicks, which blocks extension automation. Dismissing modals immediately restores automation.
3. **Cache Maintenance:** Run `python cli.py clean-cache` or `POST /api/video/clean-cache` periodically to keep Colab disk usage under 55% and prevent disk alert modals entirely.

---

## 🛠️ Agent Command Line Interface (`cli.py`)

Agents should use `python cli.py <command>` for zero-overhead video generation:

### 1. Check GPU & System Health
```bash
python cli.py status
```
*Expected Output:* Confirms backend is `online` and Colab connection is `connected`.

### 2. Clean Colab Checkpoint Cache
```bash
python cli.py clean-cache
```
Frees ~30-50 GB of disk on Colab to maintain healthy disk thresholds.

### 3. Generate a Video from Text (T2V)
```bash
# Fast Draft Preview (1.3B model, ~30-60s)
python cli.py generate "A dramatic drone flyover of a futuristic neon city in rain" --model 1.3B --duration 3

# High-Fidelity Cinematic Master (14B model, ~2-4m)
python cli.py generate "Close-up cinematic shot of an astronaut looking at Earth, golden visor reflection, 8k" --model 14B --duration 5 --ratio 16:9
```
*Options:*
* `--model`: `1.3B` (fast draft) or `14B` (flagship cinematic)
* `--ratio`: `16:9` (landscape), `9:16` (reels/shorts), or `1:1` (square)
* `--duration`: `3`, `5`, or `10` seconds
* `--steps`: Inference steps (default: 25)
* `--no-wait`: Submits to queue and returns immediately without blocking

### 4. List Local Videos
```bash
python cli.py list
```
Returns filenames, prompts, file sizes, and download URLs.

### 5. Stitch Multiple Clips into a Movie
```bash
python cli.py stitch scene1.mp4 scene2.mp4 scene3.mp4 --output final_short_film.mp4
```
Uses local FFmpeg for lossless concatenation.

---

## 🌐 Direct REST API Endpoints

Agents with HTTP access can query `http://localhost:8000` directly:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns backend health and local video counts |
| `GET` | `/colab/status` | Returns live Colab connection status and GPU info |
| `POST` | `/api/video/clean-cache` | Clears unused Colab checkpoints to free disk |
| `POST` | `/api/video/generate/t2v` | Dispatches text-to-video payload to Colab A100 |
| `POST` | `/api/video/generate/i2v` | Dispatches image-to-video payload (multipart/form-data) |
| `GET` | `/api/video/jobs/{job_id}` | Polls job logs and status (`running`, `completed`, `failed`) |
| `GET` | `/api/video/gallery` | Returns metadata list of all locally saved MP4s |
| `POST` | `/api/video/stitch` | Concatenates list of video filenames with FFmpeg |

---

## ⚠️ Important Engineering Rules for Agents

1. **Temporal Dimension Rule:** Wan 2.1 requires `num_frames` to strictly follow **`4k + 1`** (e.g., `17`, `33`, `49`, `81`). `cli.py` and `server/wan_worker.py` enforce this automatically.
2. **`torchao` Incompatibility:** Do not install older versions of `torchao` on Colab. `server/wan_worker.py` automatically purges outdated `torchao` modules to prevent `FqnToConfig` import errors.
3. **`ftfy` Patch in Wan I2V:** Wan I2V calls `ftfy.fix_text()` in prompt normalization. Always include `diffusers.pipelines.wan.pipeline_wan_i2v.ftfy = ftfy`.
4. **Memory Management for 14B:** Always use `pipe.enable_model_cpu_offload()` for 14B models on the 40GB A100 to guarantee zero OOM risk.

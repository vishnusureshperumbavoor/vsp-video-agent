# 🤖 AGENTS.md — Vibe Video Studio Instruction Manual

This document provides autonomous coding agents (Antigravity, Cursor, Claude Code, terminal bots) with instructions to generate and assemble AI videos using Vibe Video Studio.

---

## 🎯 System Architecture Overview

* **Compute Worker:** Remote Google Colab instance running an **NVIDIA A100-SXM4 GPU (40 GB VRAM)**.
* **Bridge Harness:** `vspagent` Python SDK managing `colab-mcp` browser pairing and code execution.
* **Local Backend:** FastAPI server running on `http://localhost:8000`.
* **Local Frontend:** React Vite dashboard on `http://localhost:5174`.
* **Video Generation Engine:** **Wan 2.1** (`1.3B` for drafts, `14B` for final cinematic output).

---

## 🛠️ Agent Command Line Interface (`cli.py`)

Agents should use `python cli.py <command>` for zero-overhead video generation:

### 1. Check GPU & System Health
```bash
python cli.py status
```
*Expected Output:* Confirms backend is `online` and Colab connection is `connected`.

### 2. Generate a Video from Text (T2V)
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

### 3. List Local Videos
```bash
python cli.py list
```
Returns filenames, prompts, file sizes, and download URLs.

### 4. Stitch Multiple Clips into a Movie
```bash
python cli.py stitch scene1.mp4 scene2.mp4 scene3.mp4 --output final_short_film.mp4
```
Uses local FFmpeg for lossless concatenation.

---

## 🌐 Direct REST API Endpoints (For Programmatic Agent Calls)

Agents with HTTP access can query `http://localhost:8000` directly:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns backend health and local video counts |
| `GET` | `/colab/status` | Returns live Colab connection status and GPU info |
| `POST` | `/api/video/generate/t2v` | Dispatches text-to-video payload to Colab A100 |
| `POST` | `/api/video/generate/i2v` | Dispatches image-to-video payload |
| `GET` | `/api/video/jobs/{job_id}` | Polls job logs and status (`running`, `completed`, `failed`) |
| `GET` | `/api/video/gallery` | Returns metadata list of all locally saved MP4s |
| `POST` | `/api/video/stitch` | Concatenates list of video filenames with FFmpeg |

---

## ⚠️ Important Engineering Rules for Agents

1. **Temporal Dimension Rule:** Wan 2.1 requires `num_frames` to strictly follow **`4k + 1`** (e.g., `17`, `33`, `49`, `81`). `cli.py` and `server/wan_worker.py` enforce this automatically.
2. **torchao Incompatibility:** Do not install older versions of `torchao` on Colab. `server/wan_worker.py` automatically purges outdated `torchao` modules to prevent `FqnToConfig` import errors.
3. **Model Selection:**
   * Use **`1.3B`** for testing, debugging pipelines, and validating prompt concepts.
   * Use **`14B`** for final user-facing videos and cinematic fidelity.

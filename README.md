# 🎬 Vibe Video Studio

**AI video generation on JarvisLabs A100 GPUs. FastAPI + Wan 2.1/2.2, driven by
CLI and REST — no UI, no Colab.**

![Model](https://img.shields.io/badge/Model-Wan_2.1_%2F_2.2-indigo.svg)
![GPU](https://img.shields.io/badge/GPU-A100_SXM4-emerald.svg)
![Transport](https://img.shields.io/badge/Transport-JarvisLabs_SSH-blue.svg)

---

## 🌟 Features

- **Text-to-Video (T2V):** Generate 5s–10s clips with **Wan 2.1 (1.3B Fast / 14B Pro)**.
- **Image-to-Video (I2V):** Animate a reference photo with Wan 2.1 I2V 14B.
- **Wan Animate (character replacement):** Drive a character image with a
  performance clip via **Wan2.2-Animate-14B** on ComfyUI.
- **Stitching:** Concatenate clips into a film with local FFmpeg.
- **JarvisLabs transport:** plain SSH to an on-demand A100. No Colab, no MCP
  browser bridge, no notebook kernel.
- **CLI-first:** `python cli.py` for the API, `scripts/wan_animate_runner.py`
  for provisioning and detached renders.

> The React dashboard was removed. This project is operated from the command
> line and the REST API.

---

## 🚀 Quickstart

### 1. Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env    # add JARVIS_LABS_API_TOKEN
```

The JarvisLabs harness is **vendored** in `server/jarvis/`. There is no
`vspagent` dependency and no sibling checkout to keep in sync.

### 2. Start the backend

```bash
./backend_start.sh    # FastAPI on http://localhost:8000
```

### 3. Check the GPU

```bash
python cli.py status
```

This reports backend health, the attached instance and whether an A100 is
verified. Video generation refuses to start on anything else.

### 4. Generate

```bash
# Text to video
python cli.py generate "A drone flyover of a neon city in rain" --model 1.3B --duration 3

# List results
python cli.py list

# Stitch clips
python cli.py stitch a.mp4 b.mp4 -o final.mp4
```

---

## Wan Animate on JarvisLabs (A100)

The runner provisions (or adopts) a JarvisLabs **ComfyUI** instance and runs
**Wan2.2-Animate-14B** in replacement mode.

- Inputs: `public/uploads/wan_animate/character.png` and `driving.mp4`
- Workflow: `scripts/wan_animate_workflow.json`
  (replacement mode, relighting LoRA, 6-step lightx2v)
- Runner: `scripts/wan_animate_runner.py`
- Required model: `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors`
  plus `WanAnimate_relight_lora_fp16.safetensors`
- Required nodes: ComfyUI core Wan Animate nodes,
  `comfyui_controlnet_aux`, `ComfyUI-KJNodes`, `ComfyUI-Impact-Pack`,
  and `ComfyUI-segment-anything-2`
- Output: `public/videos/wan_animate_*.mp4`

### Run

```bash
# Preview the cost/plan only
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.png \
  --dry-run

# Cheap draft: 640x368, capped at 154 frames, 40 GB A100
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.png \
  --gpu A100 --resolution draft --frames 154 --yes
```

The runner trims the driving clip to `--frames`, compresses it for upload,
removes the unused duplicate `SaveVideo` branch, and destroys the instance in
`finally`. Use `--keep-instance` / `--machine-id` to adopt and reuse an
instance. Use `--fs-id` to attach a persistent JarvisLabs filesystem so the
~20 GB of models do not need downloading every run.

The workflow processes two 77-frame segments, so it covers roughly the first
9.6 seconds of a driving clip. Add more `WanAnimateToVideo` segments to cover a
longer input.

---

## REST API

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend health, GPU connection and A100 readiness |
| `GET` | `/api/gpu/status` | Live JarvisLabs instance probe |
| `POST` | `/api/gpu/setup` | Install the Wan 2.1 runtime on the GPU box |
| `POST` | `/api/video/clean-cache` | Drop unused checkpoints on the GPU box |
| `POST` | `/api/video/generate/t2v` | Queue a text-to-video job |
| `POST` | `/api/video/generate/i2v` | Queue an image-to-video job (multipart) |
| `GET` | `/api/video/jobs` | List jobs |
| `GET` | `/api/video/jobs/{job_id}` | Poll logs and status |
| `GET` | `/api/video/gallery` | Local video metadata |
| `POST` | `/api/video/stitch` | Concatenate clips with FFmpeg |

---

## 📂 Project Structure

```
vsp-video-agent/
├── server/
│   ├── main.py              # FastAPI orchestrator
│   ├── gpu.py               # JarvisLabs SSH transport (async adapter)
│   ├── wan_worker.py        # Wan 2.1 execution payloads
│   └── jarvis/              # Vendored JarvisLabs harness (no Colab)
├── scripts/
│   ├── wan_animate_runner.py    # Provision → render → download → destroy
│   ├── wan_animate_workflow.json
│   └── wan_animate_remote.py
├── public/
│   ├── videos/              # Generated MP4s + metadata.json
│   └── uploads/             # Reference images and driving clips
├── cli.py                   # Agent/developer CLI
├── backend_start.sh
├── requirements.txt         # Backend
└── requirements-jarvis.txt  # Wan Animate runner
```

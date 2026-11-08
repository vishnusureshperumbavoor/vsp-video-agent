# 🎬 Vibe Video Studio

**Fullstack AI Video Generation Studio powered by React, FastAPI, and Google Colab A100 GPU (via `vspagent`).**

![Vibe Video Studio](https://img.shields.io/badge/Model-Wan_2.1-indigo.svg)
![GPU-Accelerated](https://img.shields.io/badge/GPU-A100_SXM4-emerald.svg)
![Runtime](https://img.shields.io/badge/Bridge-vspagent-purple.svg)

---

## 🌟 Features

- **Text-to-Video (T2V):** Generate 5s–10s cinematic video clips using **Wan 2.1 (1.3B Fast / 14B Pro)** at 720p/1080p.
- **Image-to-Video (I2V):** Upload reference photos (your face, character, or custom scenery) and animate camera and character movement.
- **Multi-Scene Storyboard:** Compose multi-shot sequences (Opening ➔ Close-up ➔ Climax) and stitch them into a continuous film using **FFmpeg**.
- **vspagent Colab A100 Harness:** Connects directly to Google Colab using MCP with zero local VRAM requirements.
- **Cinema Video Player:** Smooth playback, scrubbing, looping, custom speeds (0.5x, 1x, 1.5x, 2x), and direct downloads.
- **Live Colab Logs:** Real-time log streaming directly from the remote A100 GPU.

---

## 🚀 Quickstart

### 1. Start the Servers

```bash
# In vibe-video-studio directory:
./backend_start.sh    # Starts FastAPI backend on http://localhost:8000
./frontend_start.sh   # Starts React Vite frontend on http://localhost:5174
```

### 2. Connect Your Colab A100 Runtime

1. Open your Google Colab notebook in your browser and switch the runtime to **NVIDIA A100 GPU**.
2. Keep the Colab notebook tab open.
3. In Vibe Video Studio (**http://localhost:5174/**), click **Connect Colab**, select your Chrome profile, and click **Connect**.
4. Once connected, click **Setup A100** to install Wan 2.1 dependencies with a single click.

### 3. Generate Videos

- **Text to Video:** Enter your prompt (e.g. *"Cyberpunk car drifting on wet asphalt at night"*), choose aspect ratio (`16:9`, `9:16`, `1:1`), and click **Generate Video on A100**.
- **Image to Video:** Drag and drop your face or scene photo, enter a motion prompt, and click **Animate Photo on A100**.
- **Storyboard:** Assign generated clips to scenes and click **Stitch Movie** to produce a finished film!

---

## Wan Animate on JarvisLabs (A100)

The studio can also run **Wan2.2-Animate-14B** on an on-demand JarvisLabs
**ComfyUI** instance in replacement mode.

- Inputs: `public/uploads/wan_animate/character.jpg` and `driving.mp4`
- Workflow: `scripts/wan_animate_workflow.json`
  (replacement mode, relighting LoRA, 6-step lightx2v)
- Runner: `scripts/wan_animate_runner.py`
- Required model: `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors`
  plus `WanAnimate_relight_lora_fp16.safetensors`
- Required nodes: ComfyUI core Wan Animate nodes,
  `comfyui_controlnet_aux`, `ComfyUI-KJNodes`, `ComfyUI-Impact-Pack`,
  and `ComfyUI-segment-anything-2`
- Output: `public/videos/wan_animate_*.mp4`

### Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-jarvis.txt
uv pip install --python .venv/bin/python -e ../vspagent
cp .env.example .env    # add JARVIS_LABS_API_TOKEN
```

### Run

```bash
# Cheap draft: 640x368, capped at 154 frames, 40 GB A100
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.jpg \
  --gpu A100 --resolution draft --frames 154 --yes

# Preview the cost/plan only
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.jpg \
  --dry-run
```

The runner trims the driving clip to `--frames`, compresses it for the
Jupyter upload limit, removes the unused duplicate `SaveVideo` branch, and
destroys the instance in `finally`. Use `--keep-instance` / `--machine-id` to
adopt and reuse an instance. Use `--fs-id` to attach a persistent JarvisLabs
filesystem so the ~20 GB of models do not need downloading every run.

The workflow processes two 77-frame segments, so it covers roughly the first
9.6 seconds of a driving clip. Add more `WanAnimateToVideo` segments to cover a
longer input.

## 📂 Project Structure

```
vibe-video-studio/
├── server/
│   ├── main.py              # FastAPI orchestrator & vspagent router
│   └── wan_worker.py        # Wan 2.1 execution payloads for A100
├── src/
│   ├── components/
│   │   ├── Header.tsx       # A100 GPU status & connection triggers
│   │   ├── T2VPanel.tsx     # Text-to-Video prompt & settings
│   │   ├── I2VPanel.tsx     # Image-to-Video photo uploader
│   │   ├── StoryboardPanel.tsx # Multi-scene movie timeline & stitcher
│   │   ├── VideoPlayer.tsx  # Cinema video player & active job monitor
│   │   ├── VideoGallery.tsx # Local video library grid
│   │   └── ColabConnectionModal.tsx # vspagent pairing modal
│   ├── services/
│   │   └── api.ts           # REST API client
│   ├── App.tsx              # Main studio interface
│   └── main.tsx
├── public/
│   └── videos/              # Stored MP4 video files & metadata
├── backend_start.sh
└── frontend_start.sh
```

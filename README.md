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

# 🎬 VSP Video Agent

**AI video generation on JarvisLabs A100 GPUs. FastAPI + Wan 2.1/2.2, driven by
CLI and REST.**

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
- **JarvisLabs transport:** plain SSH to an on-demand A100.
- **CLI-first:** `python cli.py` for the API, `scripts/wan_animate_runner.py`
  for provisioning and detached renders.

> VSP Video Agent is operated from the command line and the REST API. There is
> no web UI.

---

## 🚀 Quickstart

### 1. Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env    # add JARVIS_LABS_API_TOKEN
```

The JarvisLabs harness is **vendored** in `server/jarvis/`, so the app is fully
self-contained: no external agent package and no sibling checkout to keep in
sync.

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

## MOSS-Image on JarvisLabs (A100)

`scripts/moss_image_runner.py` generates **still images** with
[`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
and an optional LoRA. It is deliberately **not** part of the video pipeline:
MOSS-Image is a text-to-image model, so nothing in `server/wan_worker.py` or the
FastAPI app can load it. Same rail as the Wan Animate runner — provision an A100,
run detached, fetch the PNGs, destroy in `finally`.

```bash
# Plan and cost only
.venv/bin/python scripts/moss_image_runner.py \
  --prompt "moss_xvstyle, a lighthouse at dusk" --count 2 --dry-run

# Generate, then auto-destroy
.venv/bin/python scripts/moss_image_runner.py \
  --prompt "moss_xvstyle, a lighthouse at dusk" \
  --count 4 --size 1024 --yes
```

| Flag | Meaning |
| :--- | :--- |
| `--count` | number of images (each gets `--seed + index`) |
| `--size` | square side; must be a multiple of 16 |
| `--lora-scale` / `--skip-lora` | adapter weight, or run the base model |
| `--machine-id` | adopt a running instance instead of provisioning |
| `--keep-instance` | leave it up for a second run |

Output lands in `public/images/` (git-ignored). Defaults come from the model
card: 50 steps, CFG 3.0, `uncond_skip_layers=[22]`, bf16 transformer and text
encoder, fp32 VAE. The runner keeps `enable_model_cpu_offload()` on by default
because the weights are ~27 GB and a 40 GB A100 OOMs without it.

Only the 70 MB LoRA file is downloaded, never the LoRA repo's bundled
`merged-model/` copy (~26 GB). The base model is fetched on the GPU box.

The LoRA in use is adult-content (`not-for-all-audiences`): 18+ subjects only,
and no real person's likeness without their consent.

### Looking at the results from a phone

`public/images/` is git-ignored and the app has no web UI, so a run that finished
on the GPU box leaves files a mobile client cannot display. Three text-first
options:

```bash
# What was generated, and whether any image came out blank or black
.venv/bin/python scripts/image_preview.py

# Rough ASCII proof, visible in the agent's log
.venv/bin/python scripts/image_preview.py --ascii --width 44

# Browse the real files from a phone on the same Wi-Fi
.venv/bin/python scripts/image_preview.py --serve --port 8080
```

`--serve` is a static file server for the operator, not part of the app. It binds
to `0.0.0.0` so your phone can reach it on the LAN, and it prints the exact URL.
Never port-forward it or put it behind a public tunnel.

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

## 🧪 Development

This repo is written mostly by coding agents, so the layout is enforced by CI
rather than by convention. A change is done when all three pass — they take about
two seconds and need no GPU:

```bash
.venv/bin/ruff check .
.venv/bin/python scripts/check_quality.py   # size ratchet + import boundaries
.venv/bin/python -m pytest
```

- **Size ratchet:** `quality/budgets.json` caps every file and function at its
  current size. Limits may go down, never up.
- **Frozen vendor:** `server/jarvis/` is checksummed. Re-vendor it, don't edit it.
- **Duplication:** `scripts/clone_detection.py` flags renamed copies (normalized
  token windows), so "copy, rename, tweak a constant" is caught.
- **Architecture:** `scripts/check_quality.py` enforces the import layers, so a
  router cannot reach into the transport and the domain stays pure Python.
- **Free tests:** `tests/fakes.py` provides `FakeGPU`, so the whole job lifecycle
  runs in-process for ₹0 instead of a real render.

`AGENTS.md` has the ownership map: which layer a new endpoint, engine or
transport method belongs to.

---

## 📂 Project Structure

```
vsp-video-agent/
├── server/
│   ├── main.py              # FastAPI orchestrator
│   ├── gpu.py               # JarvisLabs SSH transport (async adapter)
│   ├── wan_worker.py        # Wan 2.1 execution payloads
│   └── jarvis/              # Vendored JarvisLabs harness
├── scripts/
│   ├── wan_animate_runner.py    # Provision → render → download → destroy
│   ├── wan_animate_workflow.json
│   ├── wan_animate_remote.py
│   ├── moss_image_runner.py     # Provision → images → download → destroy
│   ├── moss_image_remote.py
│   ├── image_preview.py     # text listing, ASCII proof, LAN viewer for a phone
│   ├── check_quality.py     # size ratchet, import layers, frozen vendor
│   └── clone_detection.py   # near-duplicate (renamed copy) detector
├── public/
│   ├── videos/              # Generated MP4s + metadata.json
│   ├── images/              # Generated MOSS-Image PNGs
│   └── uploads/             # Reference images and driving clips
├── tests/                   # FakeGPU + payload, lifecycle, API, contract tests
├── quality/budgets.json     # Size ratchet and frozen-vendor checksums
├── cli.py                   # Agent/developer CLI
├── backend_start.sh
├── requirements.txt         # Backend
└── requirements-jarvis.txt  # Wan Animate runner
```

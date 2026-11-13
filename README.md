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

Point it at `public/` (not `public/images/`) to reach video too:

```bash
.venv/bin/python scripts/image_preview.py --serve --dir public
# http://<lan-ip>:8080/images/...   http://<lan-ip>:8080/videos/...
```

Video is served with **byte ranges** (`206 Partial Content`, via
`scripts/range_serve.py`), which is not optional: iOS Safari refuses to play a
video without it, and no browser can seek. `ffmpeg -movflags +faststart` is the
other half — the index has to come before the frames.

### When the phone is not on the same Wi-Fi

The LAN address only exists inside your network. Put both devices on a private
overlay instead of opening a port:

```bash
# once, on the laptop and on the phone: same tailnet, both logged in
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4                       # e.g. 100.101.102.103

# serve on that address only
.venv/bin/python scripts/image_preview.py --serve --dir public --bind 100.101.102.103
# then open http://100.101.102.103:8080/ on the phone, on any network
```

That is WireGuard: encrypted, authenticated per device, no open ports, and it
works on cellular. `tailscale serve` is fine too if you want TLS.

**Never** reach for `tailscale funnel`, `cloudflared`, `ngrok`, or a router
port-forward. All four publish an unauthenticated directory listing — of adult
content, once the MOSS runner has produced any.

Options that need no server at all:

| Approach | When |
| :--- | :--- |
| Syncthing, sharing `public/images/` | you want renders in the phone gallery |
| `--ascii` / `--json` | no usable network; text is enough to spot a failure |
| `scp` to a host the phone already reaches | corporate networks that block both |

---

## Girlfriend image app (Telegram beta)

The image product is a three-tier stack: **Telegram bot → FastAPI backend →
GPU image service**. The bot and the GPU are never public; the backend is the
only surface that holds the shared secrets.

```
Telegram user
   │  /imagine a rooftop at sunset
   ▼
scripts/telegram_bot.py        (long polling, no inbound port)
   │  X-Telegram-Id + X-Bot-Token
   ▼
FastAPI backend                (auth, quota, SQLite, queue, files)
   │  Bearer IMAGE_API_TOKEN
   ▼
scripts/image_server.py        (warm diffusers pipeline, :6006)
```

### Run it locally

```bash
cp .env.example .env
# fill BOT_INTERNAL_TOKEN, TELEGRAM_BOT_TOKEN and IMAGE_API_TOKEN
./backend_start.sh
python scripts/telegram_bot.py
```

Create a character in the bot, then generate:

```text
/character Ava | silver hair, green eyes, 25 years old, cinematic portrait
/imagine a rooftop at sunset
```

The backend stores jobs in `data/girlfriend.db`, writes PNGs to
`public/images/<telegram-id>/` and serves them at `/images/...`. The bot uploads
the PNG bytes to Telegram, so the backend does **not** need a public URL for
photos; set `PUBLIC_BASE_URL` only if a web client needs absolute image links.

### Public mode versus your private character

Prompt-only generation is the default: anyone who opens the bot can send
`/imagine <scene>` with no setup. Characters are optional and stay private to
the owner:

```bash
TELEGRAM_CHAT_ID=            # empty = anyone can use the bot
TELEGRAM_OWNER_ID=855227717  # only this user can /character, /characters, /select
DAILY_IMAGE_LIMIT=10         # per user
GLOBAL_DAILY_LIMIT=100       # hard cap across all users
```

Your own `/imagine` still uses your selected character; a public user has no
character, so they get plain prompt-to-image. Send `/prompt` to stop using your
character for a while, and `/select <id>` to bring it back. `/last` resends the
newest picture, which also recovers deliveries lost when the bot restarts. Each
photo caption includes the render time (`⏱ 63.4s`); `/status` shows it too.

Run the backend as **one process** (`uvicorn` default). The queue and the SQLite
writer are per-process; scaling to several API instances needs Postgres and a
shared queue first.

### Deploy the GPU service

The service is a normal JarvisLabs GPU container; there is no serverless path
for a diffusers pipeline.

```bash
jl filesystem create --name img-models --storage 200 --region IN2 --yes
jl scripts add scripts/image_startup.sh --name image-service
jl create --gpu A100 --template pytorch --name img-service \
  --storage 100 --fs-id <fs_id> --script-id <script_id> \
  --http-ports "6006" --yes --json
jl get <machine_id> --json     # endpoints[0] and ssh_command
jl upload <machine_id> scripts/image_server.py /home/app/image_server.py
# create /home/app/env (mode 600) with IMAGE_MODEL_ID, IMAGE_API_TOKEN, etc.
```

Point the backend at the endpoint:

```bash
IMAGE_API_URL=https://<endpoint>
IMAGE_API_TOKEN=<the same value as /home/app/env>
```

Pause the instance when idle (`jl pause <id>`) and resume it before a demo
(`jl resume <id>`). A failed job reports that the GPU is offline instead of
hanging, so the bot can tell the user to retry.

Model defaults are env-driven: `IMAGE_MODEL_ID` plus an optional
`IMAGE_CUSTOM_PIPELINE` (needed for MOSS-Image's `pipeline_moss_image`).
Leave `IMAGE_CUSTOM_PIPELINE` empty for a plain diffusers repo such as
`black-forest-labs/FLUX.1-schnell`. Set `IMAGE_DEFAULT_STEPS=50` for MOSS and
`4` for FLUX.1-schnell; the bot uses this value when it queues a job.

### Optional LoRA adapter

`scripts/image_server.py` can wrap the MOSS transformer with a LoRA adapter:

```bash
IMAGE_LORA_DIR=/home/jl_fs/lora/moss   # directory with pytorch_lora_weights.safetensors
IMAGE_LORA_SCALE=0.5                   # 0 disables, 1.0 is full strength
```

The MOSS pipeline does not expose `load_lora_weights`, so the server applies the
adapter directly to the transformer's linear layers. Only the adapter file is
needed (~70 MB); never download the repo's merged model. Any adapter used here
is adult content: 18+ only, and no real person's likeness without consent.

### Image endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/characters` | Create a character bible entry |
| `GET` | `/api/characters` | List this user's characters |
| `GET` | `/api/characters/active` | The selected character |
| `POST` | `/api/characters/{id}/select` | Choose who to draw |
| `POST` | `/api/images` | Queue a generation |
| `GET` | `/api/images` | Recent jobs for this user |
| `GET` | `/api/images/{job_id}` | Status, error and image URL |

All routes require `X-Telegram-Id` and `X-Bot-Token`; only the bot should hold
the token. The GPU endpoint requires its own `IMAGE_API_TOKEN`.

### Operator tools

```bash
# Backend, GPU service, active character, recent jobs
python scripts/image_admin.py doctor

# Characters and mode
python scripts/image_admin.py characters
python scripts/image_admin.py select <character_id>
python scripts/image_admin.py prompt

# Queue and wait, or check recent work
python scripts/image_admin.py imagine "a rooftop at sunset" --wait
python scripts/image_admin.py jobs --limit 10
python scripts/image_admin.py last

# Speed benchmark: calls the GPU directly, bypasses quota, bills GPU time
python scripts/image_bench.py --dry-run
python scripts/image_bench.py --sizes 1024,768 --steps 50,30
```

Both tools read `.env`, so `BOT_INTERNAL_TOKEN`, `TELEGRAM_OWNER_ID`,
`IMAGE_API_URL` and `IMAGE_API_TOKEN` come from the same place as the bot.

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
│   ├── main.py              # FastAPI wiring
│   ├── api/                 # characters, images, video routers
│   ├── domain/              # character bible + job state machine
│   ├── services/            # queue, quota, job lifecycle
│   ├── transport/           # HTTP adapter to the GPU image service
│   ├── store.py             # SQLite repository
│   ├── storage.py           # local image storage (R2/S3 seam)
│   ├── gpu.py               # JarvisLabs SSH transport (async adapter)
│   ├── wan_worker.py        # Wan 2.1 execution payloads
│   └── jarvis/              # Vendored JarvisLabs harness
├── scripts/
│   ├── image_server.py          # GPU-side diffusers HTTP service
│   ├── image_startup.sh         # JarvisLabs container startup script
│   ├── telegram_bot.py          # bot front door
│   ├── wan_animate_runner.py    # Provision → render → download → destroy
│   ├── wan_animate_workflow.json
│   ├── wan_animate_remote.py
│   ├── moss_image_runner.py     # Provision → images → download → destroy
│   ├── moss_image_remote.py
│   ├── image_preview.py     # text listing, ASCII proof, LAN viewer for a phone
│   ├── range_serve.py       # byte-range handler so mobile video can seek
│   ├── check_quality.py     # size ratchet, import layers, frozen vendor
│   └── clone_detection.py   # near-duplicate (renamed copy) detector
├── public/
│   ├── videos/              # Generated MP4s + metadata.json
│   ├── images/              # Generated images, served at /images
│   └── uploads/             # Reference images and driving clips
├── data/                    # SQLite database (git-ignored)
├── tests/                   # FakeGPU/FakeImageAPI + domain, service, API tests
├── quality/budgets.json     # Size ratchet and frozen-vendor checksums
├── cli.py                   # Agent/developer CLI
├── backend_start.sh
├── requirements.txt         # Backend
└── requirements-jarvis.txt  # Wan Animate runner
```

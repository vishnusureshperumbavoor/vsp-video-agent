# 🤖 AGENTS.md — VSP Video Agent Instruction Manual

Instructions for autonomous coding agents (Antigravity, Cursor, Claude Code,
terminal bots) working on this repo.

**This project is JarvisLabs-only and CLI-first.** Every GPU operation is an SSH
command against an on-demand JarvisLabs instance, and the app has no web UI.

---

## ⛔ STRICT HARDWARE MANDATE: NVIDIA A100 GPU ONLY

1. **Never generate video without an NVIDIA A100.** Local CPU or any other GPU
   must NEVER be used. Video diffusion needs the memory bandwidth and bfloat16
   tensor cores of the A100.
2. **Pre-flight enforcement.** `/api/video/generate/t2v`, `/api/video/generate/i2v`
   and `python cli.py generate` all call `verify_a100()` before staging a job.
3. **Automatic rejection.** If no instance is running, or the attached GPU is not
   an A100, execution MUST terminate with:
   `"❌ STOPPING EXECUTION: NVIDIA A100 GPU is strictly required for video generation."`

---

## 🎯 System Architecture

* **Compute worker:** JarvisLabs GPU instance (A100 40 GB) running ComfyUI or the
  Wan diffusers pipeline.
* **Transport:** SSH only, via the vendored harness in `server/jarvis/`
  (`JarvisLabsClient`). No `vspagent` dependency, no sibling checkout.
* **Backend:** FastAPI on `http://localhost:8000` (`server/main.py`).
* **GPU adapter:** `server/gpu.py` — async wrapper used by the API jobs.
* **Engines:** Wan 2.1 (`1.3B` draft, `14B` cinematic, `I2V-14B`), and
  Wan2.2-Animate-14B for character replacement.
* **CLI:** `cli.py` (API client + direct `gpu` commands) and
  `scripts/wan_animate_runner.py` (provision → render → download → destroy).

---

## ⚡ Asynchronous Detached Execution (Fire & Poll)

Never hold a blocking SSH call open for a multi-minute render.

1. **Stage:** write the generated script to `/home/vsp_jobs/<job_id>.py`.
2. **Spawn detached:** `cd /home/vsp_jobs; setsid nohup python3 -u <job_id>.py > <job_id>.log 2>&1 < /dev/null &`
   — the call returns immediately.
3. **Poll:** the FastAPI background task tails `<job_id>.log` every 5 s.
4. **Download:** fetch the finished MP4 over SCP.

`server/gpu.py` implements this as `write_file` → `spawn` → `read_log` →
`download_file`.

---

## 🛠️ Command Line Interface

### Check the GPU

```bash
python cli.py status          # backend + attached instance + A100 readiness
python cli.py gpu doctor      # pre-flight without the backend
python cli.py gpu instances   # list JarvisLabs instances
```

### Generate

```bash
python cli.py generate "A drone flyover of a neon city in rain" --model 1.3B --duration 3
python cli.py generate "Astronaut watching Earth, golden visor, 8k" --model 14B --duration 5 --ratio 16:9
```

Options: `--model 1.3B|14B`, `--ratio 16:9|9:16|1:1`, `--duration 3|5|10`,
`--steps`, `--no-wait`.

### Inspect, clean, stitch

```bash
python cli.py jobs            # queued/running/completed jobs and last log line
python cli.py list            # local gallery
python cli.py clean-cache     # drop unused checkpoints on the GPU box
python cli.py stitch a.mp4 b.mp4 -o final.mp4
```

### Pre-flight helpers (no backend required)

```bash
python cli.py gpu audit /home/ComfyUI/models/vae/wan_2.1_vae.safetensors \
  --min-size /home/ComfyUI/models/vae/wan_2.1_vae.safetensors=200000000
python cli.py gpu wait 8188 --http --path /object_info
```

---

## 🌐 REST API

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend health, GPU connection, A100 readiness |
| `GET` | `/api/gpu/status` | Live instance probe |
| `POST` | `/api/gpu/setup` | Install the Wan 2.1 runtime on the GPU box |
| `POST` | `/api/video/clean-cache` | Drop unused checkpoints |
| `POST` | `/api/video/generate/t2v` | Queue text-to-video |
| `POST` | `/api/video/generate/i2v` | Queue image-to-video (multipart) |
| `GET` | `/api/video/jobs` | List jobs |
| `GET` | `/api/video/jobs/{job_id}` | Poll logs and status |
| `GET` | `/api/video/gallery` | Local video metadata |
| `POST` | `/api/video/stitch` | Concatenate clips with FFmpeg |

---

## 🎭 Wan Animate on JarvisLabs

Use `scripts/wan_animate_runner.py`, never ad-hoc shell commands.

```bash
# Always inspect the plan/cost first
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.png \
  --gpu A100 --resolution draft --frames 154 --dry-run

# Then run with auto-destroy
.venv/bin/python scripts/wan_animate_runner.py \
  --video public/uploads/wan_animate/driving.mp4 \
  --image public/uploads/wan_animate/character.png \
  --gpu A100 --resolution draft --frames 154 --yes
```

Rules:

1. **Run the pre-flight before provisioning.** `python cli.py gpu doctor` must
   pass, and the model audit must be clean:

   ```bash
   python cli.py gpu audit \
     /home/ComfyUI/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors \
     --min-size /home/ComfyUI/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors=18000000000
   ```

   **Never size-check a template model with plain `stat -c %s`.** Template
   weights are symlinks into the read-only `/home/.sd` mount, so plain `stat`
   reports the ~37-byte link length, the model looks missing, and the
   re-download dies with `Read-only file system`. Use `stat -Lc %s`.
2. **Never kill and start ComfyUI in one shell line.** `pkill -f` matches the
   shell running the compound command and kills itself before the server
   launches. Kill by port in one call, spawn with `setsid` in a second call,
   then `python cli.py gpu wait 8188 --http --path /object_info`.
3. **Background jobs must use `setsid`** and be polled from the log file.
   `spawn()` emits `cd dir; cmd &` — **not** `cd dir && cmd &`, which makes bash
   background the whole AND-list and block until the job finishes.
4. **SSH must not forward stdin** (`ssh -n`). A forwarded stdin keeps the
   channel open, so a successfully spawned daemon looks like a hang.
5. **Always cap frames** to the workflow budget (154) before uploading. Never
   preprocess the whole source clip when only the first segments are used.
6. **Draft at `--resolution draft`** (640×368). Use `720p` only for a final.
7. Prefer **`--gpu A100` (40 GB)** or a spot GPU; `A100-80GB` is only for 720p.
8. Use `--fs-id` with a persistent JarvisLabs filesystem to cache the ~20 GB of
   models across runs. The runner warns when it provisions without one.
9. Use `--dry-run` to show the estimated INR cost before provisioning.
10. The runner destroys created instances in `finally`. If it is interrupted,
    verify with `jl list` and destroy any remaining instance.
11. Never expose ComfyUI or the JarvisLabs instance publicly. Use SSH only.
12. Only use character images and driving videos you have rights to use.

---

## ⚠️ Engineering Rules

1. **Temporal dimension rule:** Wan requires `num_frames` to follow **`4k + 1`**
   (`17`, `33`, `49`, `81`). `server/wan_worker.py` enforces this automatically.
2. **`torchao` incompatibility:** do not install older `torchao` versions. The
   generated scripts purge stale `torchao` modules to avoid `FqnToConfig` errors.
3. **`ftfy` patch in Wan I2V:** Wan I2V calls `ftfy.fix_text()`. The generated
   I2V script sets `diffusers.pipelines.wan.pipeline_wan_i2v.ftfy = ftfy`.
4. **Memory management for 14B:** always use `pipe.enable_model_cpu_offload()`
   for 14B models on the 40 GB A100 to guarantee zero OOM risk.
5. **Transport is SSH only.** Do not introduce notebook kernels or
   browser-automation paths.
6. **`server/jarvis/` is vendored.** If you change it, update the upstream
   `vspagent` project too, and keep `CLIENT_API_VERSION` in sync.

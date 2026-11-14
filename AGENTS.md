# 🤖 AGENTS.md — VSP Video Agent Instruction Manual

Instructions for autonomous coding agents (Antigravity, Cursor, Claude Code,
terminal bots) working on this repo.

**This project is JarvisLabs-only and CLI-first.** Every GPU operation is an SSH
command against an on-demand JarvisLabs instance, and the app has no web UI.

Read **How This Codebase Stays Small** before adding any code.

---

## 📏 How This Codebase Stays Small (read before writing code)

This repository is maintained mostly by coding agents. Agents append to whatever
file the task names, so the layout is enforced mechanically rather than by good
intentions. Three commands run in CI on every push.

### Definition of done

A change is not finished until all three pass:

```bash
.venv/bin/ruff check .
.venv/bin/python scripts/check_quality.py
.venv/bin/python -m pytest
```

They take about two seconds and need no GPU. If a test needs a GPU, it is testing
at the wrong layer — use `tests/fakes.py` instead.

### Where code goes

| You are adding… | Put it in | Never in |
| :--- | :--- | :--- |
| An HTTP endpoint | `server/api/<feature>.py` (parse → call → serialize) | `server/main.py` |
| An engine or payload builder | `server/domain/` | `server/main.py` |
| A transport method | `server/gpu.py` **and** `tests/fakes.py` | a router |
| A CLI command | `cli.py` | `server/` |
| Character/prompt rules | `server/domain/character.py` | a router |
| Image job lifecycle | `server/domain/image_job.py` | the service |
| An image endpoint | `server/api/images.py` | `server/main.py` |
| A GPU image adapter | `server/transport/image_api.py` **and** `tests/fakes.py` | a router |
| The Telegram front door | `scripts/telegram_bot.py` | `server/` |
| GPU-side model serving | `scripts/image_server.py` | `server/` |
| Image operator CLI | `scripts/image_admin.py` | `server/` |
| Speed benchmark | `scripts/image_bench.py` | the backend |
| Reference-edit CLI | `scripts/reference_edit.py` | the Telegram flow |
| Character bootstrap CLI | `scripts/character_cli.py` + `scripts/character/` | `server/` |
| Video operator CLI | `scripts/video_admin.py` | `server/` |
| Video HTTP client | `scripts/video_client.py` | a router |
| Video runbook | `docs/video-generation.md` | a model class |
| Micro-story CLI | `scripts/story_video.py` | `server/` |
| Story planning/rendering | `scripts/story/` | a router |
| Provisioning/render logic | a package under `scripts/`, with a thin entrypoint | one giant `main()` |

`server/main.py` is **wiring only**. It is already at its cap: if you need a new
line there, extract a module first. Same for `scripts/wan_animate_runner.py`,
which still mixes provisioning, ffmpeg, upload, ComfyUI control and teardown —
split it when you next touch it.

### Hard limits — `quality/budgets.json`

| Rule | Limit |
| :--- | :--- |
| New file | 500 lines (warn at 300) |
| Function | 120 lines (warn at 80) |
| `server/jarvis/` | frozen, verified by sha256 |
| Near-duplicate code | 41 duplicated lines total, ratchet to 0 |

The budget file is a **ratchet**: limits may go down, never up. Raising one is a
deliberate edit that appears in review as a change to that file.

### Duplication

`scripts/clone_detection.py` normalizes the token stream (identifiers → `N`,
numbers → `0`, strings → `S`) and hashes every 50-token window. A copied
function with renamed variables still matches. The 41 lines currently flagged
are the shared scaffolding of `generate_t2v_script()` and
`generate_i2v_script()`; remove them when you split that file. Do not raise the
cap.

### Frozen vendored code

`server/jarvis/` is third-party. Do not edit, reformat or "improve" it — the gate
fails on a checksum mismatch. To upgrade it, re-vendor the files and update the
hashes in `quality/budgets.json` in the same commit.

### Test pyramid

| File | Covers |
| :--- | :--- |
| `tests/test_payloads.py` | golden strings for the generated GPU scripts (4k+1 frames, offload, ftfy, torchao) |
| `tests/test_job_lifecycle.py` | queue → stage → spawn → poll → download, on `FakeGPU` |
| `tests/test_api.py` | status codes and the A100 guardrail |
| `tests/test_contracts.py` | route snapshot and forbidden transport vocabulary |
| `tests/test_clone_detection.py` | the near-duplicate detector |
| `tests/test_quality_gate.py` | the gate itself |
| `tests/test_moss_runner.py` | MOSS-Image command quoting, cost estimate, remote arg validation |
| `tests/test_image_preview.py` | blank-render detection, ASCII grid, phone listing |
| `tests/test_range_serve.py` | byte ranges (`206`), so mobile video plays and seeks |
| `tests/test_character_domain.py` | prompt composition, size/step rules, job states |
| `tests/test_image_service.py` | queue, quota, storage, GPU failure, owner isolation |
| `tests/test_image_api.py` | bot auth, character/image routes, gallery |
| `tests/test_telegram_bot.py` | command parsing and user-facing formatting |
| `tests/test_image_tools.py` | admin formatting and benchmark configs |
| `tests/test_character_tools.py` | bootstrap scenes, captions, curation math, train command |
| `tests/test_video_tools.py` | video payloads, CLI parsing, job formatting |
| `tests/test_story_tools.py` | story parsing, prompts, SRT timing |

Add the regression test in the cheapest layer that can catch it.

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
  (`JarvisLabsClient`). Frozen by checksum; never edited in place.
* **Backend:** FastAPI on `http://localhost:8000`. `server/main.py` is wiring;
  feature routers belong in `server/api/`.
* **GPU adapter:** `server/gpu.py` — async wrapper used by the API jobs.
* **Domain:** pure Python (no FastAPI, no transport) in `server/domain/` — job
  state machine, payload builders, frame/resolution rules.
* **Engines:** Wan 2.1 (`1.3B` draft, `14B` cinematic, `I2V-14B`), and
  Wan2.2-Animate-14B for character replacement.
* **CLI:** `cli.py` (HTTP client + direct `gpu` commands) and
  `scripts/wan_animate_runner.py` (provision → render → download → destroy).

`server/api/` owns the HTTP routers (video plus the Telegram-facing character
and image product). `server/domain/` is pure Python: the character bible and the
image job state machine. `server/services/` owns the queue and quota. Do not
grow `server/main.py`; add a router and register it from `server/api/__init__.py`.

## 🎞️ Video generation

Any agent can drive the video pipeline through `scripts/video_admin.py`; the
full pre-flight, model choices, timings and pitfalls are in
`docs/video-generation.md`. No OpenCode-specific skill is required.

```bash
python scripts/video_admin.py doctor
python scripts/video_admin.py t2v "a beach at sunset"
python scripts/video_admin.py i2v face.png "she smiles at the camera"
```

1.3B is the draft model. Use I2V from an approved character still for identity,
or Wan Animate for motion transfer. Never expose SSH or ComfyUI publicly.

## 🎬 Micro-story videos (Level 1)

`scripts/story_video.py` turns a JSON storyboard + one character reference into
a 9:16 MP4 with Piper narration, an ambient bed and burned captions. The
agent-agnostic runbook is `docs/microstory-video.md`; the renderer is never run
in CI, so always start with `--dry-run`.

```bash
python scripts/story_video.py plan --script story.json
python scripts/story_video.py render --script story.json --reference face.png --dry-run
```

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

## 🖼️ MOSS-Image on JarvisLabs

`scripts/moss_image_runner.py` generates still images with
`OpenMOSS-Team/MOSS-Image-SFT` plus the `axiomofmind/Moss-image-nsfw-uncensored-lora`
adapter (on by default at scale 0.8; `--skip-lora` runs the base model). It is a
**separate model family** — text-to-image, not Wan — and is intentionally not
wired into the FastAPI app or `server/wan_worker.py`.

Rules:

1. **Never `git clone` the LoRA repo.** `axiomofmind/Moss-image-nsfw-uncensored-lora`
   bundles a `merged-model/` copy (~26 GB); the adapter itself is 70 MB. The
   runner downloads only `pytorch_lora_weights.safetensors` via
   `hf_hub_download`. Locally, use
   `GIT_LFS_SKIP_SMUDGE=1 git clone ...` then
   `git lfs pull --include="pytorch_lora_weights.safetensors"`.
2. **Keep `enable_model_cpu_offload()` on.** The weights are ~27 GB; a 40 GB A100
   OOMs at 1024² without it. `--no-offload` is for bigger cards only.
3. **Sizes must be multiples of 16.** The remote helper rejects anything else
   before loading the model.
4. **Do not drop `uncond_skip_layers=[22]`** for the default sampler; it is part
   of the model card's recipe. Passing `--negative-prompt` sets it to `[]`.
5. **Output goes to `public/images/`**, which is git-ignored. Never commit
   generated images.
6. The default LoRA is adult-content (`not-for-all-audiences`): 18+ subjects
   only, and no real person's likeness without consent.
7. Same teardown rules as Wan Animate: `--dry-run` first, detached `spawn`,
   destroy in `finally`, verify with `jl list` after an interruption.
8. **Results are verified as text, not pixels.** The operator may be on a phone:
   `python scripts/image_preview.py` lists dimensions and flags blank renders,
   `--ascii` prints a proof, `--serve` exposes the folder on the private network.
   That server is an operator tool — never bind it to a public address and never
   put it behind `tailscale funnel`, `cloudflared`, `ngrok` or a port-forward.
   Off-LAN access is a tailnet plus `--bind <tailscale-ip>`, nothing else.

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

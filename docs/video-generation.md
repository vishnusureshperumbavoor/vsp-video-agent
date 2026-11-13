# Video generation runbook

Agent-agnostic. Any agent or terminal can follow this; it does not depend on
OpenCode, Cursor or a specific UI.

The pipeline is:

```
Telegram / CLI / HTTP
        │
        ▼
FastAPI backend ── SSH ──> JarvisLabs A100
   /api/video/*              diffusers / Wan 2.1
        │
        ▼
public/videos/*.mp4
```

## Pre-flight

1. The backend must see an A100:

   ```bash
   python scripts/video_admin.py doctor
   ```

   Expected: `backend: healthy` and a `NVIDIA A100...` GPU line.

2. If it says `connected: false` / no instance, register the running instance
   with the vendored client:

   ```python
   from server.jarvis import JarvisLabsClient

   client = JarvisLabsClient()
   client.register_instance(
       host="<public-ip>",
       machine_id="<machine-id>",
       name="video-a100",
       gpu_type="A100",
       status="Running",
   )
   ```

   Get the values from `jl list --json`. Do not expose SSH or ComfyUI publicly.

3. The GPU box needs `accelerate`, `imageio[ffmpeg]` and `sentencepiece`. Without
   `accelerate`, diffusers silently forces `low_cpu_mem_usage=False` and Wan
   fails with `keep_in_fp32_modules` — the payload already passes
   `low_cpu_mem_usage=True`.

4. Wan requires `num_frames == 4k + 1` (`17`, `33`, `49`, `81`). The payload
   snaps invalid values.

## CLI

```bash
# Text to video
python scripts/video_admin.py t2v "a beach at sunset"

# Better quality, slower
python scripts/video_admin.py t2v "a beach at sunset" --model 14B --duration 5

# Image to video (character still)
python scripts/video_admin.py i2v face.png "she smiles and turns to the camera"

# Inspect / resend / clean
python scripts/video_admin.py jobs
python scripts/video_admin.py status <job_id>
python scripts/video_admin.py send <filename>
python scripts/video_admin.py clean-cache
```

Options: `--model 1.3B|14B`, `--duration`, `--steps`, `--ratio`,
`--seed`, `--no-wait`, `--json`. `t2v` defaults to 3s / 25 steps / 16:9;
`i2v` defaults to 5s / 25 steps / 9:16.

## API

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend + GPU |
| `GET` | `/api/gpu/status` | Live A100 probe |
| `POST` | `/api/video/generate/t2v` | Text to video |
| `POST` | `/api/video/generate/i2v` | Image to video (multipart) |
| `GET` | `/api/video/jobs` | List jobs |
| `GET` | `/api/video/jobs/{id}` | Status + logs |
| `POST` | `/api/video/clean-cache` | Drop unused checkpoints |

## Models and measured timings

| Model | Use | Measured |
| :--- | :--- | :--- |
| Wan 2.1 T2V 1.3B | drafts | 3s / 480p / 25 steps: **58.7s GPU, 80s total** |
| Wan 2.1 T2V 14B | final quality | several minutes, CPU offload |
| Wan 2.1 I2V 14B | animate a still | minutes, CPU offload |
| Wan2.2-Animate-14B | motion transfer | ComfyUI setup, 20GB+ models |

The 1.3B model does **not** preserve a character's identity. Use I2V from an
approved still, or Wan Animate for dance/motion transfer.

## Telegram

`/video <prompt>` uses the backend T2V endpoint and sends the MP4 with GPU and
total render time. Defaults are read from `.env`:

```bash
VIDEO_MODEL_SIZE=1.3B
VIDEO_DURATION=3
VIDEO_STEPS=25
VIDEO_DAILY_LIMIT=3
```

The daily limit is in-process; a bot restart resets it.

## Pitfalls

- **Paused instance:** the endpoint returns 404/offline. Resume it and retry.
- **Resume can change the machine ID:** `jl resume` may return a new
  `machine_id` and public IP. Re-register the running instance with the
  vendored client before expecting the backend to find it.
- **Missing accelerate:** the `low_cpu_mem_usage` error above.
- **Frames:** never pass a value that is not `4k + 1`.
- **SSH:** use `setsid nohup` for detached jobs, `ssh -n`, and never
  `pkill -f` the pattern in the same shell line that starts the service.
- **Cleanup:** verify `jl list` after an interrupted run; destroy anything left
  behind.
- **Disk:** the 14B checkpoints are large; check `/home` before downloading.

## Verify

```bash
python scripts/video_admin.py status <job_id>
ffprobe -v error -show_entries format=duration -of csv=p=0 public/videos/<file>.mp4
```

A completed job must show `=== JOB_SUCCESS:` in its logs and a non-zero MP4.

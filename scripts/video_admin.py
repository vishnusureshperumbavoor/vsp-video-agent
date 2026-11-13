#!/usr/bin/env python3
"""Operator CLI for the video pipeline.

Works with any agent or terminal; it only speaks HTTP to the FastAPI backend.

    python scripts/video_admin.py doctor
    python scripts/video_admin.py t2v "a beach at sunset"
    python scripts/video_admin.py i2v face.png "she smiles at the camera"
    python scripts/video_admin.py jobs
    python scripts/video_admin.py status <job_id>
    python scripts/video_admin.py send <filename>
    python scripts/video_admin.py clean-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telegram_bot import Telegram, load_dotenv  # noqa: E402
from video_client import VideoClient, VideoError  # noqa: E402
from video_jobs import extract_video_seconds  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "public" / "videos"


def format_seconds(value) -> str:
    return f"{value:.1f}s" if value is not None else "—"


def format_job(job: dict) -> str:
    seconds = extract_video_seconds(job.get("logs"))
    prompt = (job.get("prompt") or "")[:60]
    return f"{job.get('status', '?')}: {prompt} ({format_seconds(seconds)})"


def build_t2v_payload(
    prompt: str,
    model: str = "1.3B",
    duration: int = 3,
    steps: int = 25,
    ratio: str = "16:9",
    seed: int | None = None,
) -> dict:
    payload = {
        "prompt": prompt,
        "model_size": model,
        "duration": duration,
        "steps": steps,
        "aspect_ratio": ratio,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def build_i2v_payload(
    prompt: str,
    duration: int = 5,
    steps: int = 25,
    ratio: str = "9:16",
    seed: int | None = None,
) -> dict:
    payload = {
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "aspect_ratio": ratio,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def build_client(args) -> VideoClient:
    load_dotenv(str(ROOT / ".env"))
    return VideoClient(args.url or os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"))


def emit(args, payload, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for line in lines:
            print(line)


def wait_for_job(client: VideoClient, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        job = client.get_job(job_id)
        logs = job.get("logs", [])
        if logs and logs[-1] != last:
            last = logs[-1]
            print(f"[{job.get('status')}] {last}", flush=True)
        if job.get("status") in ("completed", "failed"):
            return job
        time.sleep(5)
    raise VideoError("Timed out waiting for the video job.")


def cmd_doctor(args) -> int:
    client = build_client(args)
    health = client.health()
    gpu = client.gpu_status()
    jobs = client.list_jobs()[:3]
    payload = {"backend": health, "gpu": gpu, "jobs": jobs}
    lines = [
        f"backend: {health.get('status')} ({health.get('transport')})",
        f"gpu:     {gpu.get('gpu') or 'none'} a100={gpu.get('connected')}",
        *[f"  {format_job(job)}" for job in jobs],
    ]
    emit(args, payload, lines)
    return 0 if health.get("status") == "healthy" and gpu.get("connected") else 1


def finish_job(args, client: VideoClient, job: dict, started: float) -> int:
    print(f"queued: {job.get('job_id')}")
    if args.no_wait:
        emit(args, job, [json.dumps(job)])
        return 0
    job = wait_for_job(client, job["job_id"], args.timeout)
    gpu_seconds = extract_video_seconds(job.get("logs"))
    lines = [
        f"status: {job.get('status')}",
        f"gpu:    {format_seconds(gpu_seconds)}",
        f"total:  {format_seconds(time.time() - started)}",
        f"url:    {job.get('video_url') or '—'}",
        f"error:  {job.get('error') or '—'}",
    ]
    emit(args, job, lines)
    return 0 if job.get("status") == "completed" else 1


def cmd_t2v(args) -> int:
    client = build_client(args)
    started = time.time()
    job = client.create_t2v(
        build_t2v_payload(
            args.prompt, args.model, args.duration, args.steps, args.ratio, args.seed
        )
    )
    return finish_job(args, client, job, started)


def cmd_i2v(args) -> int:
    client = build_client(args)
    started = time.time()
    job = client.create_i2v(
        args.image,
        build_i2v_payload(args.prompt, args.duration, args.steps, args.ratio, args.seed),
    )
    return finish_job(args, client, job, started)


def cmd_jobs(args) -> int:
    client = build_client(args)
    jobs = client.list_jobs()[: args.limit]
    emit(args, jobs, [format_job(job) for job in jobs] or ["no jobs"])
    return 0


def cmd_status(args) -> int:
    client = build_client(args)
    job = client.get_job(args.job_id)
    emit(args, job, [format_job(job), f"error: {job.get('error') or '—'}"])
    return 0


def cmd_send(args) -> int:
    load_dotenv(str(ROOT / ".env"))
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    owner = int(
        os.environ.get("TELEGRAM_OWNER_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or 0
    )
    if not token or not owner:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID first")
    path = VIDEOS_DIR / args.filename
    if not path.exists():
        raise SystemExit(f"video not found: {path}")
    caption = args.caption or args.filename
    Telegram(token).send_video(owner, path.read_bytes(), caption)
    print(f"sent: {args.filename}")
    return 0


def cmd_clean(args) -> int:
    client = build_client(args)
    result = client.clean_cache()
    emit(args, result, [f"clean-cache: {result.get('status')}"])
    return 0


def add_options(parser: argparse.ArgumentParser, options: list[tuple[str, dict]]) -> None:
    for flags, kwargs in options:
        parser.add_argument(flags, **kwargs)


def add_generation_options(
    parser: argparse.ArgumentParser, duration: int, steps: int, ratio: str
) -> None:
    parser.add_argument("--duration", type=int, default=duration)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--ratio", default=ratio, choices=["16:9", "9:16", "1:1"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--no-wait", action="store_true")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video pipeline operator CLI")
    parser.add_argument("--url", help="Backend URL (default: BACKEND_URL)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Backend, GPU and recent jobs").set_defaults(func=cmd_doctor)

    t2v = sub.add_parser("t2v", help="Text to video")
    add_options(
        t2v,
        [
            ("prompt", {}),
            ("--model", {"default": "1.3B", "choices": ["1.3B", "14B"]}),
        ],
    )
    add_generation_options(t2v, 3, 25, "16:9")
    t2v.set_defaults(func=cmd_t2v)

    i2v = sub.add_parser("i2v", help="Image to video")
    add_options(
        i2v,
        [
            ("image", {}),
            ("prompt", {}),
        ],
    )
    add_generation_options(i2v, 5, 25, "9:16")
    i2v.set_defaults(func=cmd_i2v)

    jobs = sub.add_parser("jobs", help="List video jobs")
    jobs.add_argument("--limit", type=int, default=10)
    jobs.set_defaults(func=cmd_jobs)

    status = sub.add_parser("status", help="One video job")
    status.add_argument("job_id")
    status.set_defaults(func=cmd_status)

    send = sub.add_parser("send", help="Send a local video to Telegram")
    send.add_argument("filename")
    send.add_argument("--caption")
    send.set_defaults(func=cmd_send)

    sub.add_parser("clean-cache", help="Drop unused GPU checkpoints").set_defaults(func=cmd_clean)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VideoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

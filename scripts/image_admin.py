#!/usr/bin/env python3
"""Operator CLI for the girlfriend image app.

Talks to the FastAPI backend with the bot token, so it never touches the GPU
directly and never needs the Telegram app.

    python scripts/image_admin.py doctor
    python scripts/image_admin.py characters
    python scripts/image_admin.py imagine "a rooftop at sunset" --wait
    python scripts/image_admin.py jobs --limit 10
    python scripts/image_admin.py prompt
    python scripts/image_admin.py select <character_id>
    python scripts/image_admin.py last
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from telegram_bot import Backend, BotError, load_dotenv, wait_for_image

ROOT = Path(__file__).resolve().parent.parent


def format_seconds(value) -> str:
    return f"{value:.1f}s" if value is not None else "—"


def format_job(job: dict) -> str:
    prompt = (job.get("prompt") or "")[:60]
    return f"{job['status']}: {prompt} ({format_seconds(job.get('generation_seconds'))})"


def emit(args, payload, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for line in lines:
            print(line)


def build_backend(args) -> tuple[Backend, int]:
    token = os.environ.get("BOT_INTERNAL_TOKEN", "")
    if not token:
        raise SystemExit("BOT_INTERNAL_TOKEN is not set in .env")
    owner = args.owner or int(
        os.environ.get("TELEGRAM_OWNER_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or 0
    )
    if not owner:
        raise SystemExit("set TELEGRAM_OWNER_ID or pass --owner")
    url = args.url or os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
    return Backend(url, token), owner


def gpu_health() -> dict:
    url = os.environ.get("IMAGE_API_URL", "")
    if not url:
        return {"ok": False, "error": "IMAGE_API_URL is not set"}
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": type(exc).__name__}


def cmd_doctor(args) -> int:
    backend, owner = build_backend(args)
    try:
        health = backend.health()
    except BotError as exc:
        print(f"backend: unreachable ({exc})")
        return 1
    gpu = gpu_health()
    try:
        active = backend.active_character(owner)
    except BotError:
        active = None
    jobs = backend.list_images(owner)[:3]
    payload = {
        "backend": health,
        "gpu": gpu,
        "owner": owner,
        "active_character": active,
        "jobs": jobs,
    }
    lines = [
        f"backend:  {health.get('status')} ({health.get('transport')})",
        f"gpu:      {gpu.get('ok')} {gpu.get('model', gpu.get('error', ''))}",
        f"owner:    {owner}",
        f"character:{' ' + active['name'] if active else ' prompt-only'}",
        *[f"  {format_job(job)}" for job in jobs],
    ]
    emit(args, payload, lines)
    return 0 if health.get("status") == "healthy" and gpu.get("ok") else 1


def cmd_characters(args) -> int:
    backend, owner = build_backend(args)
    characters = backend.list_characters(owner)
    lines = [
        f"{item['id']}  {item['name']}  seed={item['seed']}"
        for item in characters
    ] or ["no characters"]
    emit(args, characters, lines)
    return 0


def cmd_select(args) -> int:
    backend, owner = build_backend(args)
    character = backend.select_character(owner, args.character_id)
    emit(args, character, [f"selected: {character['name']}"])
    return 0


def cmd_prompt(args) -> int:
    backend, owner = build_backend(args)
    payload = backend.clear_character(owner)
    emit(args, payload, ["prompt mode: /imagine no longer uses a character"])
    return 0


def cmd_imagine(args) -> int:
    backend, owner = build_backend(args)
    job = backend.create_image(
        owner,
        args.prompt,
        character_id=args.character,
        size=args.size,
        steps=args.steps,
        seed=args.seed,
    )
    if args.wait:
        job = wait_for_image(backend, owner, job["id"])
    lines = [
        f"job:    {job['id']}",
        f"status: {job['status']}",
        f"prompt: {job['prompt']}",
        f"time:   {format_seconds(job.get('generation_seconds'))}",
        f"url:    {job.get('image_url') or '—'}",
    ]
    emit(args, job, lines)
    return 0 if job["status"] != "failed" else 1


def cmd_jobs(args) -> int:
    backend, owner = build_backend(args)
    jobs = backend.list_images(owner)[: args.limit]
    lines = [format_job(job) for job in jobs] or ["no jobs"]
    emit(args, jobs, lines)
    return 0


def cmd_status(args) -> int:
    backend, owner = build_backend(args)
    job = backend.get_image(owner, args.job_id)
    emit(args, job, [format_job(job), f"error: {job.get('error') or '—'}"])
    return 0


def cmd_last(args) -> int:
    backend, owner = build_backend(args)
    completed = next(
        (job for job in backend.list_images(owner) if job.get("status") == "completed"),
        None,
    )
    if completed is None:
        emit(args, None, ["no completed pictures yet"])
        return 1
    emit(args, completed, [completed["image_url"] or "no url"])
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operate the girlfriend image app")
    parser.add_argument("--url", help="Backend URL (default: BACKEND_URL or localhost:8000)")
    parser.add_argument("--owner", type=int, help="Telegram user id (default: TELEGRAM_OWNER_ID)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Backend, GPU service, character and recent jobs").set_defaults(
        func=cmd_doctor
    )
    sub.add_parser("characters", help="List characters").set_defaults(func=cmd_characters)

    select = sub.add_parser("select", help="Select a character")
    select.add_argument("character_id")
    select.set_defaults(func=cmd_select)

    sub.add_parser("prompt", help="Clear the active character").set_defaults(func=cmd_prompt)

    imagine = sub.add_parser("imagine", help="Queue an image")
    imagine.add_argument("prompt")
    imagine.add_argument("--character")
    imagine.add_argument("--size", type=int, default=1024)
    imagine.add_argument("--steps", type=int)
    imagine.add_argument("--seed", type=int)
    imagine.add_argument("--wait", dest="wait", action="store_true", default=True)
    imagine.add_argument("--no-wait", dest="wait", action="store_false")
    imagine.set_defaults(func=cmd_imagine)

    jobs = sub.add_parser("jobs", help="Recent jobs")
    jobs.add_argument("--limit", type=int, default=10)
    jobs.set_defaults(func=cmd_jobs)

    status = sub.add_parser("status", help="One job")
    status.add_argument("job_id")
    status.set_defaults(func=cmd_status)

    sub.add_parser("last", help="Newest completed image URL").set_defaults(func=cmd_last)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(str(ROOT / ".env"))
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Telegram front door for the girlfriend image app.

The bot is a thin HTTP client: it verifies the Telegram user, forwards requests
to the FastAPI backend with a shared secret, and sends the finished picture back.
It never imports the server, never touches the GPU and never logs a token.

    TELEGRAM_BOT_TOKEN=... BOT_INTERNAL_TOKEN=... \
      python3 scripts/telegram_bot.py

Commands: /start /character /characters /select /imagine /status
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests
from video_jobs import VIDEO_POLL_SECONDS, allow_video, format_video_caption

MAX_INFLIGHT = 4
POLL_SECONDS = 600
HELP = (
    "Send me a scene and I will make a picture.\n\n"
    "/imagine a rooftop at sunset — generate\n"
    "/video a cinematic beach at sunset — 3s clip\n"
    "/last — resend the newest picture\n"
    "/status — recent pictures\n\n"
    "Optional character mode:\n"
    "/character Name | how she looks\n"
    "/characters — list your characters\n"
    "/select <id> — choose who to draw\n"
    "/prompt — stop using your character"
)
PUBLIC_HELP = (
    "Send /imagine <scene> and I will make a picture.\n\n"
    "Example: /imagine a lighthouse at dusk, 35mm photo\n"
    "/video a cinematic beach at sunset — 3s clip\n"
    "/last — resend the newest picture\n"
    "/status — recent pictures"
)


class BotError(RuntimeError):
    """A failure the user is allowed to see."""


def load_dotenv(path: str = ".env") -> None:
    """Load .env without overriding variables already in the environment."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def is_allowed(chat_id, owner_id, allowed_chat_id: int) -> bool:
    """An empty allowlist leaves the bot open; a set one locks it to that chat."""
    if not allowed_chat_id:
        return True
    return chat_id == allowed_chat_id or owner_id == allowed_chat_id


def character_commands_allowed(owner_id, character_owner: int) -> bool:
    """Characters are private when an owner is configured; prompts stay public."""
    return not character_owner or owner_id == character_owner


def http_json(
    label: str,
    url: str,
    payload: dict | None = None,
    method: str = "POST",
    headers: dict | None = None,
    timeout: int = 30,
):
    """One JSON request. The label keeps tokens out of every error message."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            detail = json.loads(body).get("detail", body)
        except (ValueError, AttributeError):
            detail = body
        message = str(detail).strip()[:300] or f"{label} failed (HTTP {exc.code})."
        raise BotError(message) from exc
    except OSError as exc:
        raise BotError(f"{label} is unreachable.") from exc


def parse_command(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text.startswith("/"):
        return "", text
    head, _, argument = text.partition(" ")
    command = head[1:].split("@", 1)[0].lower()
    return command, argument.strip()


def parse_character(argument: str) -> tuple[str, str]:
    name, separator, appearance = argument.partition("|")
    if not separator or not name.strip() or not appearance.strip():
        raise BotError("Use: /character Ava | silver hair, green eyes, 25 years old")
    return name.strip(), appearance.strip()


class Telegram:
    def __init__(self, token: str) -> None:
        self._base = f"https://api.telegram.org/bot{token}"

    def call(self, method: str, payload: dict | None = None) -> dict:
        return http_json("Telegram", f"{self._base}/{method}", payload, timeout=40)

    def updates(self, offset: int) -> list[dict]:
        result = self.call("getUpdates", {"offset": offset, "timeout": 30})
        return result.get("result", [])

    def send_message(self, chat_id: int, text: str) -> None:
        self.call("sendMessage", {"chat_id": chat_id, "text": text[:4000]})

    def send_photo_bytes(self, chat_id: int, data: bytes, caption: str) -> None:
        try:
            response = requests.post(
                f"{self._base}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption[:900]},
                files={"photo": ("image.png", data, "image/png")},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise BotError("Telegram is unreachable.") from exc
        if not response.ok:
            raise BotError(f"Telegram rejected the photo (HTTP {response.status_code}).")

    def send_video(self, chat_id: int, data: bytes, caption: str) -> None:
        try:
            response = requests.post(
                f"{self._base}/sendVideo",
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"video": ("clip.mp4", data, "video/mp4")},
                timeout=180,
            )
        except requests.RequestException as exc:
            raise BotError("Telegram is unreachable.") from exc
        if not response.ok:
            raise BotError(f"Telegram rejected the video (HTTP {response.status_code}).")


class Backend:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token

    def _request(self, method: str, owner_id: int, path: str, payload: dict | None = None):
        return http_json(
            "Backend",
            f"{self.base_url}{path}",
            payload,
            method=method,
            headers={"X-Telegram-Id": str(owner_id), "X-Bot-Token": self._token},
        )

    def create_character(self, owner_id: int, name: str, appearance: str) -> dict:
        payload = {"name": name, "appearance": appearance}
        return self._request("POST", owner_id, "/api/characters", payload)

    def list_characters(self, owner_id: int) -> list[dict]:
        return self._request("GET", owner_id, "/api/characters")

    def select_character(self, owner_id: int, character_id: str) -> dict:
        return self._request("POST", owner_id, f"/api/characters/{character_id}/select")

    def clear_character(self, owner_id: int) -> dict:
        return self._request("DELETE", owner_id, "/api/characters/active")

    def create_image(self, owner_id: int, prompt: str, **options) -> dict:
        payload = {"prompt": prompt}
        payload.update({key: value for key, value in options.items() if value is not None})
        return self._request("POST", owner_id, "/api/images", payload)

    def get_image(self, owner_id: int, job_id: str) -> dict:
        return self._request("GET", owner_id, f"/api/images/{job_id}")

    def list_images(self, owner_id: int) -> list[dict]:
        return self._request("GET", owner_id, "/api/images")

    def create_video(
        self,
        prompt: str,
        duration: int = 3,
        steps: int = 25,
        model_size: str = "1.3B",
        aspect_ratio: str = "16:9",
    ) -> dict:
        payload = {
            "prompt": prompt,
            "duration": duration,
            "steps": steps,
            "model_size": model_size,
            "aspect_ratio": aspect_ratio,
        }
        return http_json("Backend", f"{self.base_url}/api/video/generate/t2v", payload)

    def get_video(self, job_id: str) -> dict:
        return http_json(
            "Backend", f"{self.base_url}/api/video/jobs/{job_id}", method="GET"
        )

    def health(self) -> dict:
        return self._request("GET", 0, "/api/health")

    def active_character(self, owner_id: int) -> dict:
        return self._request("GET", owner_id, "/api/characters/active")


def describe_characters(characters: list[dict]) -> str:
    if not characters:
        return "No characters yet. Create one with /character Name | how she looks"
    lines = [f"• {item['name']} — {item['id']}" for item in characters]
    return "Your characters:\n" + "\n".join(lines)


def format_caption(job: dict, limit: int = 900) -> str:
    """Prompt plus the render time, for the photo caption."""
    prompt = (job.get("prompt") or "")[:limit]
    seconds = job.get("generation_seconds")
    if seconds is None:
        start = job.get("started_at") or job.get("created_at") or 0
        end = job.get("updated_at") or 0
        seconds = round(end - start, 1) if end and start else None
    if seconds is None:
        return prompt
    return f"{prompt}\n⏱ {seconds:.1f}s"


def _poll_job(fetch, job_id: str, timeout: int, timeout_message: str) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = fetch(job_id)
        if job.get("status") == "completed":
            return job
        if job.get("status") == "failed":
            raise BotError(job.get("error") or "Generation failed.")
        time.sleep(2)
    raise BotError(timeout_message)


def wait_for_image(backend: Backend, owner_id: int, job_id: str) -> dict:
    return _poll_job(
        lambda job: backend.get_image(owner_id, job),
        job_id,
        POLL_SECONDS,
        "Timed out waiting for the GPU. Try again once it is resumed.",
    )


def wait_for_video(backend: Backend, job_id: str) -> dict:
    return _poll_job(backend.get_video, job_id, VIDEO_POLL_SECONDS, "Timed out waiting for the video.")


def fetch_bytes(url: str, label: str) -> bytes:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise BotError(f"Could not fetch the generated {label}.") from exc


def fetch_image(url: str) -> bytes:
    return fetch_bytes(url, "image")


def fetch_video(url: str) -> bytes:
    return fetch_bytes(url, "video")


def deliver_video(bot: Telegram, backend: Backend, chat_id: int, owner_id: int, job_id: str) -> None:
    started = time.time()
    try:
        job = wait_for_video(backend, job_id)
        url = job.get("video_url") or ""
        if url.startswith("/"):
            url = f"{backend.base_url}{url}"
        caption = format_video_caption(job, time.time() - started)
        bot.send_video(chat_id, fetch_video(url), caption)
    except BotError as exc:
        bot.send_message(chat_id, str(exc))


def deliver(bot: Telegram, backend: Backend, chat_id: int, owner_id: int, job_id: str) -> None:
    try:
        job = wait_for_image(backend, owner_id, job_id)
        url = job.get("image_url") or ""
        if url.startswith("/"):
            url = f"{backend.base_url}{url}"
        bot.send_photo_bytes(chat_id, fetch_image(url), format_caption(job))
    except BotError as exc:
        bot.send_message(chat_id, str(exc))


def _reply(chat_id: int, text: str) -> tuple[int, str, None, None]:
    return chat_id, text, None, None


def handle_message(
    backend: Backend, update: dict, character_owner: int = 0
) -> tuple[int, str, str | None, str | None] | None:
    """Return (chat_id, reply, job_id, kind), where kind is image or video."""
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    owner_id = (message.get("from") or {}).get("id")
    text = message.get("text") or ""
    if not chat_id or not owner_id or not text:
        return None
    command, argument = parse_command(text)
    if command in ("", "start", "help"):
        help_text = HELP if character_commands_allowed(owner_id, character_owner) else PUBLIC_HELP
        return _reply(chat_id, help_text)
    if command in ("character", "characters", "select", "prompt") and not character_commands_allowed(
        owner_id, character_owner
    ):
        return _reply(chat_id, "Characters are private. Use /imagine <scene> for a picture.")
    if command in ("prompt", "nocharacter"):
        backend.clear_character(owner_id)
        return _reply(
            chat_id,
            "Prompt mode enabled. /imagine uses your text only. "
            "Use /select <id> to bring a character back.",
        )
    if command == "character":
        name, appearance = parse_character(argument)
        character = backend.create_character(owner_id, name, appearance)
        backend.select_character(owner_id, character["id"])
        return _reply(chat_id, f"Created {character['name']}. Send /imagine <scene>.")
    if command == "characters":
        return _reply(chat_id, describe_characters(backend.list_characters(owner_id)))
    if command == "select":
        if not argument:
            raise BotError("Use: /select <id>")
        character = backend.select_character(owner_id, argument)
        return _reply(chat_id, f"Now drawing {character['name']}.")
    if command == "imagine":
        if not argument:
            raise BotError("Use: /imagine a rooftop at sunset")
        job = backend.create_image(owner_id, argument)
        return chat_id, "Queued. I will send it when it is ready.", job["id"], "image"
    if command == "video":
        if not argument:
            raise BotError("Use: /video a cinematic beach at sunset")
        limit = int(os.environ.get("VIDEO_DAILY_LIMIT", "3"))
        if not allow_video(owner_id, limit):
            return _reply(chat_id, f"Daily video limit ({limit}) reached.")
        job = backend.create_video(
            argument,
            duration=int(os.environ.get("VIDEO_DURATION", "3")),
            steps=int(os.environ.get("VIDEO_STEPS", "25")),
            model_size=os.environ.get("VIDEO_MODEL_SIZE", "1.3B"),
        )
        return chat_id, "Queued. I will send the clip when it is ready.", job["job_id"], "video"
    if command == "last":
        completed = next(
            (job for job in backend.list_images(owner_id) if job.get("status") == "completed"),
            None,
        )
        if completed is None:
            return _reply(chat_id, "No finished pictures yet.")
        return chat_id, "Here is your latest picture.", completed["id"], "image"
    if command == "status":
        jobs = backend.list_images(owner_id)[:5]
        if not jobs:
            return _reply(chat_id, "No pictures yet.")
        lines = []
        for item in jobs:
            seconds = item.get("generation_seconds")
            timing = f" ({seconds:.0f}s)" if seconds is not None else ""
            lines.append(f"{item['status']}: {item['prompt'][:60]}{timing}")
        return _reply(chat_id, "\n".join(lines))
    return _reply(chat_id, PUBLIC_HELP)


def main() -> int:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    backend_token = os.environ.get("BOT_INTERNAL_TOKEN", "")
    if not token or not backend_token:
        print("Set TELEGRAM_BOT_TOKEN and BOT_INTERNAL_TOKEN first.", file=sys.stderr)
        return 2
    bot = Telegram(token)
    backend = Backend(os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"), backend_token)
    allowed_chat_id = int(os.environ.get("TELEGRAM_CHAT_ID", "0") or 0)
    character_owner = int(os.environ.get("TELEGRAM_OWNER_ID", "0") or 0)
    if not character_owner:
        character_owner = allowed_chat_id
    inflight = threading.Semaphore(MAX_INFLIGHT)
    offset = 0
    print("bot listening; press Ctrl-C to stop", flush=True)
    while True:
        try:
            updates = bot.updates(offset)
        except BotError as exc:
            print(exc, flush=True)
            time.sleep(2)
            continue
        for update in updates:
            offset = max(offset, int(update.get("update_id", 0)) + 1)
            message = update.get("message") or {}
            chat_id = (message.get("chat") or {}).get("id")
            owner_id = (message.get("from") or {}).get("id")
            if not is_allowed(chat_id, owner_id, allowed_chat_id):
                continue
            try:
                result = handle_message(backend, update, character_owner)
                if result is None:
                    continue
                result_chat, reply, job_id, kind = result
                if job_id and owner_id:
                    inflight.acquire()
                    target = (
                        _deliver_video_and_release
                        if kind == "video"
                        else _deliver_and_release
                    )
                    threading.Thread(
                        target=target,
                        args=(inflight, bot, backend, result_chat, owner_id, job_id),
                        daemon=True,
                    ).start()
                else:
                    bot.send_message(result_chat, reply)
            except BotError as exc:
                if chat_id:
                    try:
                        bot.send_message(chat_id, str(exc))
                    except BotError:
                        pass
            except Exception as exc:  # one bad update must not kill the bot
                print(f"update failed: {type(exc).__name__}", flush=True)
                if chat_id:
                    try:
                        bot.send_message(chat_id, "Something went wrong. Try again.")
                    except BotError:
                        pass


def _deliver_and_release(semaphore, bot, backend, chat_id, owner_id, job_id) -> None:
    try:
        deliver(bot, backend, chat_id, owner_id, job_id)
    finally:
        semaphore.release()


def _deliver_video_and_release(semaphore, bot, backend, chat_id, owner_id, job_id) -> None:
    try:
        deliver_video(bot, backend, chat_id, owner_id, job_id)
    finally:
        semaphore.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)

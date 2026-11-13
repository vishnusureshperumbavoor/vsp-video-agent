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

MAX_INFLIGHT = 4
POLL_SECONDS = 180
HELP = (
    "Send me a scene and I will make a picture of your character.\n\n"
    "/character Name | how she looks  — create her\n"
    "/characters — list your characters\n"
    "/select <id> — choose who to draw\n"
    "/imagine a rooftop at sunset — generate\n"
    "/status — recent pictures"
)


class BotError(RuntimeError):
    """A failure the user is allowed to see."""


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
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        raise BotError(f"{label} rejected the request (HTTP {exc.code}). {detail}") from exc
    except urllib.error.URLError as exc:
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

    def send_photo(self, chat_id: int, url: str, caption: str) -> None:
        self.call("sendPhoto", {"chat_id": chat_id, "photo": url, "caption": caption[:900]})


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

    def create_image(self, owner_id: int, prompt: str) -> dict:
        return self._request("POST", owner_id, "/api/images", {"prompt": prompt})

    def get_image(self, owner_id: int, job_id: str) -> dict:
        return self._request("GET", owner_id, f"/api/images/{job_id}")

    def list_images(self, owner_id: int) -> list[dict]:
        return self._request("GET", owner_id, "/api/images")


def describe_characters(characters: list[dict]) -> str:
    if not characters:
        return "No characters yet. Create one with /character Name | how she looks"
    lines = [f"• {item['name']} — {item['id']}" for item in characters]
    return "Your characters:\n" + "\n".join(lines)


def wait_for_image(backend: Backend, owner_id: int, job_id: str) -> dict:
    deadline = time.time() + POLL_SECONDS
    while time.time() < deadline:
        job = backend.get_image(owner_id, job_id)
        if job.get("status") == "completed":
            return job
        if job.get("status") == "failed":
            raise BotError(job.get("error") or "Generation failed.")
        time.sleep(2)
    raise BotError("Timed out waiting for the GPU. Try again once it is resumed.")


def deliver(bot: Telegram, backend: Backend, chat_id: int, owner_id: int, job_id: str) -> None:
    try:
        job = wait_for_image(backend, owner_id, job_id)
        url = job.get("image_url") or ""
        if url.startswith("/"):
            url = f"{backend.base_url}{url}"
        bot.send_photo(chat_id, url, job.get("prompt", "")[:900])
    except BotError as exc:
        bot.send_message(chat_id, str(exc))


def handle_message(backend: Backend, update: dict) -> tuple[int, str, str | None] | None:
    """Return (chat_id, reply, job_id). job_id is set only for /imagine."""
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    owner_id = (message.get("from") or {}).get("id")
    text = message.get("text") or ""
    if not chat_id or not owner_id or not text:
        return None
    command, argument = parse_command(text)
    if command in ("", "start", "help"):
        return chat_id, HELP, None
    if command == "character":
        name, appearance = parse_character(argument)
        character = backend.create_character(owner_id, name, appearance)
        backend.select_character(owner_id, character["id"])
        return chat_id, f"Created {character['name']}. Send /imagine <scene>.", None
    if command == "characters":
        return chat_id, describe_characters(backend.list_characters(owner_id)), None
    if command == "select":
        if not argument:
            raise BotError("Use: /select <id>")
        character = backend.select_character(owner_id, argument)
        return chat_id, f"Now drawing {character['name']}.", None
    if command == "imagine":
        if not argument:
            raise BotError("Use: /imagine a rooftop at sunset")
        job = backend.create_image(owner_id, argument)
        return chat_id, "Queued. I will send it when it is ready.", job["id"]
    if command == "status":
        jobs = backend.list_images(owner_id)[:5]
        if not jobs:
            return chat_id, "No pictures yet.", None
        lines = [f"{item['status']}: {item['prompt'][:70]}" for item in jobs]
        return chat_id, "\n".join(lines), None
    return chat_id, HELP, None


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    backend_token = os.environ.get("BOT_INTERNAL_TOKEN", "")
    if not token or not backend_token:
        print("Set TELEGRAM_BOT_TOKEN and BOT_INTERNAL_TOKEN first.", file=sys.stderr)
        return 2
    bot = Telegram(token)
    backend = Backend(os.environ.get("BACKEND_URL", "http://127.0.0.1:8000"), backend_token)
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
            try:
                result = handle_message(backend, update)
                if result is None:
                    continue
                result_chat, reply, job_id = result
                if job_id and owner_id:
                    inflight.acquire()
                    threading.Thread(
                        target=_deliver_and_release,
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


def _deliver_and_release(semaphore, bot, backend, chat_id, owner_id, job_id) -> None:
    try:
        deliver(bot, backend, chat_id, owner_id, job_id)
    finally:
        semaphore.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)

#!/usr/bin/env python3
"""Tiny ComfyUI control helper executed *inside* the JarvisLabs instance.

The runner uploads this file over Jupyter and calls it with one of the commands
below. It deliberately prints compact JSON so the Jupyter IOPub rate limit is
not hit by ComfyUI's full history payload.
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8188"


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=120) as response:
        return json.load(response)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wan_animate_remote.py submit|history|queue|object_info")
        return 2

    command = sys.argv[1]

    if command == "submit":
        payload = json.load(open("/home/wan_animate_job/workflow_prompt.json"))
        request = urllib.request.Request(
            BASE + "/prompt",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            print(json.dumps(json.load(response)))
        return 0

    if command == "queue":
        data = _get("/queue")
        running = data.get("queue_running", [])
        pending = data.get("queue_pending", [])
        print(
            json.dumps(
                {
                    "running": len(running),
                    "pending": len(pending),
                    "ids": [item[1] for item in running + pending if len(item) > 1],
                }
            )
        )
        return 0

    if command == "history":
        if len(sys.argv) < 3:
            print("{}")
            return 2
        prompt_id = sys.argv[2]
        data = _get("/history/" + prompt_id)
        item = data.get(prompt_id)
        if not item:
            print("{}")
            return 0
        print(
            json.dumps(
                {
                    "status": item.get("status"),
                    "outputs": item.get("outputs", {}),
                }
            )
        )
        return 0

    if command == "object_info":
        data = _get("/object_info")
        print(json.dumps({name: name in data for name in sys.argv[2:]}))
        return 0

    print(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

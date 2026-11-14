"""Runs on the GPU box: submit a graph to ComfyUI, wait, print saved paths.

The local CLI uploads this file plus a job JSON and tails the log. Output is
line-oriented so the caller never has to parse a progress bar.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def _request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> int:
    job = json.load(open(sys.argv[1]))
    server = job.get("server", "http://127.0.0.1:8188").rstrip("/")
    graph = job["graph"]

    try:
        prompt_id = _request(f"{server}/prompt", {"prompt": graph})["prompt_id"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:3000]
        print(f"=== ERROR: ComfyUI rejected the graph: {detail}", flush=True)
        return 1

    print(f"=== PROMPT {prompt_id}", flush=True)
    deadline = time.time() + float(job.get("timeout", 2400))
    ticks = 0
    while time.time() < deadline:
        try:
            entry = _request(f"{server}/history/{prompt_id}").get(prompt_id)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
            continue
        if entry:
            if entry.get("status", {}).get("status_str") == "error":
                print(f"=== ERROR {json.dumps(entry['status'])[:3000]}", flush=True)
                return 1
            outputs = entry.get("outputs") or {}
            if outputs:
                for payload in outputs.values():
                    for item in payload.get("images", []):
                        subfolder = item.get("subfolder", "")
                        path = f"{subfolder}/{item['filename']}".lstrip("/")
                        print(f"=== FILE {path}", flush=True)
                print("=== DONE", flush=True)
                return 0
        ticks += 1
        if ticks % 12 == 0:
            print(f"=== WAIT {int(time.time() - (deadline - float(job.get('timeout', 2400))))}s", flush=True)
        time.sleep(5)

    print("=== TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

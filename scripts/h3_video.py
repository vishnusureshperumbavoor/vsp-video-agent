#!/usr/bin/env python3
"""MiniMax-H3 text-to-video with native stereo audio, on a JarvisLabs A100.

H3 jointly denoises video and audio, so there is no separate TTS or foley
pass. The model is far too large for a 40 GB card in bf16; this runner uses the
Comfy-Org quantized repack (int8 transformer + nvfp4 text encoder, ~42 GB).

    # plan only, no SSH
    python scripts/h3_video.py "a saxophonist in the rain" --dry-run

    # first run on a fresh instance: install ComfyUI and pull the weights
    python scripts/h3_video.py "..." --machine-id 574386 --setup --yes

    # later runs reuse the warm server
    python scripts/h3_video.py "..." --machine-id 574386 --pause

The instance is never destroyed here; pass --pause to stop billing when done.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instance_adopt import adopt_instance  # noqa: E402
from minimax_h3.workflow import (  # noqa: E402
    A100_INR_PER_HOUR,
    DEFAULT_HEIGHT,
    DEFAULT_SECONDS,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    MODEL_FILES,
    MODEL_REPO,
    build_graph,
    estimate,
    snap_length,
)

from server.jarvis import JarvisLabsClient  # noqa: E402

COMFY_DIR = "/home/jl_fs/ComfyUI"
COMFY_PYTHON = "/home/venv/bin/python"
REMOTE_DIR = "/home/app"
PORT = 8188
PACKAGES = "-r requirements.txt"
WORKER = Path(__file__).resolve().parent / "minimax_h3" / "worker.py"


def log(message: str) -> None:
    print(f"[h3] {message}", flush=True)


def die(message: str) -> None:
    log(f"ERROR: {message}")
    raise SystemExit(1)


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError:
        die(f"bad --size {value!r}; expected WIDTHxHEIGHT, e.g. 736x416")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-H3 video + audio on JarvisLabs")
    parser.add_argument("prompt")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--size", default=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--machine-id", help="Adopt a running instance")
    parser.add_argument("--host", default="", help="SSH host, for when the instance list lags")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--setup", action="store_true", help="Install ComfyUI and weights")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan only")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--pause", action="store_true", help="Pause the instance when done")
    parser.add_argument("--out-dir", default=str(ROOT / "public" / "videos"))
    parser.add_argument("--timeout", type=int, default=40, help="Render timeout in minutes")
    return parser.parse_args()


def ensure_instance(
    client: JarvisLabsClient, machine_id: str, host: str = "", port: int = 22
) -> None:
    if adopt_instance(client, machine_id, "h3") is not None:
        return
    if host:
        client.register_instance(
            host=host,
            machine_id=str(machine_id),
            name="h3",
            port=port,
            gpu_type="A100",
            status="Running",
        )
        return
    die(
        f"instance {machine_id} is not in the API list (it lags behind a resume); "
        "rerun with --host from the ssh_command that `jl resume` printed"
    )


def remote(client: JarvisLabsClient, machine_id: str, command: str) -> tuple[int, str]:
    return client.exec(command, instance_id=machine_id, stream=False)


def healthy(client: JarvisLabsClient, machine_id: str) -> bool:
    code, _ = remote(
        client,
        machine_id,
        f"curl -sf -m 3 http://127.0.0.1:{PORT}/object_info -o /dev/null",
    )
    return code == 0


def start_server(client: JarvisLabsClient, machine_id: str) -> None:
    log("starting ComfyUI ...")
    remote(
        client,
        machine_id,
        f"cd {COMFY_DIR} && setsid nohup {COMFY_PYTHON} -u main.py "
        f"--listen 127.0.0.1 --port {PORT} > /home/app/comfyui.log 2>&1 < /dev/null &",
    )
    wait_healthy(client, machine_id)


def wait_healthy(client: JarvisLabsClient, machine_id: str) -> None:
    code, _ = remote(
        client,
        machine_id,
        f"for i in $(seq 1 40); do sleep 3; "
        f"curl -sf -m 3 http://127.0.0.1:{PORT}/object_info -o /dev/null && exit 0; done; exit 1",
    )
    if code != 0:
        die("ComfyUI did not answer on port 8188; see /home/app/comfyui.log")


def missing_models(client: JarvisLabsClient, machine_id: str) -> list[str]:
    missing = []
    for path in MODEL_FILES:
        code, _ = remote(client, machine_id, f"test -f {COMFY_DIR}/models/{path}")
        if code != 0:
            missing.append(path)
    return missing


def install_comfyui(client: JarvisLabsClient, machine_id: str) -> None:
    code, out = remote(client, machine_id, f"test -f {COMFY_DIR}/main.py")
    if code != 0:
        log("cloning ComfyUI ...")
        code, out = remote(
            client,
            machine_id,
            f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}",
        )
        if code != 0:
            die(f"clone failed:\n{out[-800:]}")
    log("installing ComfyUI requirements ...")
    code, out = remote(client, machine_id, f"cd {COMFY_DIR} && {COMFY_PYTHON} -m pip install -q {PACKAGES}")
    if code != 0:
        die(f"pip install failed:\n{out[-1200:]}")


def download_models(client: JarvisLabsClient, machine_id: str, missing: list[str]) -> None:
    total = sum(MODEL_FILES[path] for path in missing)
    log(f"downloading {len(missing)} files ({total:.1f} GB) ...")
    payload = base64.b64encode(
        (
            "import os\n"
            "from huggingface_hub import hf_hub_download\n"
            f"os.environ.setdefault('HF_HOME', '/home/jl_fs/hf')\n"
            f"for name in {missing!r}:\n"
            f"    print(hf_hub_download(repo_id={MODEL_REPO!r}, filename=name, "
            f"local_dir={COMFY_DIR + '/models'!r}), flush=True)\n"
        ).encode()
    ).decode("ascii")
    code, out = remote(
        client,
        machine_id,
        f"echo {payload} | base64 -d | {COMFY_PYTHON} -",
    )
    if code != 0:
        die(f"model download failed:\n{out[-1500:]}")
    log("weights ready")


def setup(client: JarvisLabsClient, machine_id: str) -> None:
    install_comfyui(client, machine_id)
    missing = missing_models(client, machine_id)
    if missing:
        download_models(client, machine_id, missing)
    if not healthy(client, machine_id):
        start_server(client, machine_id)


def put_text(client: JarvisLabsClient, machine_id: str, remote_path: str, text: str) -> None:
    """Write a small file over SSH.

    The vendored ``sync_up`` uses ``scp`` without ``BatchMode``/``-n``, so it can
    block forever on a passphrase prompt. Job files are a few KB; base64 over an
    already-working ``ssh`` channel is both faster and safe.
    """
    payload = base64.b64encode(text.encode()).decode("ascii")
    code, out = remote(client, machine_id, f"echo {payload} | base64 -d > {remote_path}")
    if code != 0:
        die(f"could not write {remote_path}:\n{out[-500:]}")


def get_bytes(client: JarvisLabsClient, machine_id: str, remote_path: str) -> bytes | None:
    code, out = remote(client, machine_id, f"base64 -w0 {remote_path}")
    if code != 0:
        return None
    try:
        return base64.b64decode(out.strip(), validate=True)
    except (ValueError, binascii.Error):
        return None


def submit(client: JarvisLabsClient, machine_id: str, graph: dict, timeout: int) -> list[str]:
    job = {"graph": graph, "server": f"http://127.0.0.1:{PORT}", "timeout": timeout * 60}

    log("uploading the job ...")
    remote(client, machine_id, f"mkdir -p {REMOTE_DIR}")
    put_text(client, machine_id, f"{REMOTE_DIR}/h3_job.json", json.dumps(job))
    put_text(client, machine_id, f"{REMOTE_DIR}/h3_worker.py", WORKER.read_text())

    log_path = f"{REMOTE_DIR}/h3_render.log"
    remote(client, machine_id, f"rm -f {log_path}")
    remote(
        client,
        machine_id,
        f"cd {REMOTE_DIR} && setsid nohup {COMFY_PYTHON} -u h3_worker.py h3_job.json "
        f"> {log_path} 2>&1 < /dev/null &",
    )
    log("waiting for ComfyUI ...")

    seen = 0
    files: list[str] = []
    deadline = time.time() + timeout * 60
    last_tick = 0.0
    while time.time() < deadline:
        code, out = remote(client, machine_id, f"tail -n +{seen + 1} {log_path} 2>/dev/null || true")
        lines = out.splitlines()
        seen += len(lines)
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("=== FILE "):
                files.append(line.split(" ", 2)[2].strip())
            elif line.startswith("=== ERROR") or line.startswith("=== TIMEOUT"):
                die(line)
            elif not line.startswith("=== WAIT"):
                log(line)
        if any("=== DONE" in line for line in lines):
            return files
        if time.time() - last_tick > 30:
            last_tick = time.time()
            log("still rendering ...")
        time.sleep(6)
    die("render timed out; the remote log is still on the instance")


def fetch(client: JarvisLabsClient, machine_id: str, paths: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for path in paths:
        target = out_dir / Path(path).name
        data = get_bytes(client, machine_id, f"{COMFY_DIR}/output/{path}")
        if data is None:
            log(f"WARNING: could not download {path}")
            continue
        target.write_bytes(data)
        saved.append(target)
    return saved


def main() -> int:
    args = parse_args()
    width, height = parse_size(args.size)
    length = snap_length(args.seconds)
    minutes, cost = estimate(args.seconds, width, height, args.steps)

    log("MiniMax-H3 plan")
    log(f"  size     {width}x{height}, {length} frames (~{length / 24:.1f}s), {args.steps} steps")
    log("  audio    native stereo, 32 kHz, generated in the same pass")
    log(f"  estimate ~{minutes:.1f} min, ~{cost:.0f} INR on an A100")
    log(f"  output   {args.out_dir}")

    graph = build_graph(
        args.prompt,
        width=width,
        height=height,
        length=length,
        steps=args.steps,
        seed=args.seed,
        lora_strength=args.lora_strength,
    )
    if args.dry_run:
        print(json.dumps(graph, indent=2)[:1200])
        return 0
    if not args.machine_id:
        die("--machine-id is required; `jl list` shows what is running")
    if not args.yes:
        answer = input("Render on the GPU? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return 1

    client = JarvisLabsClient()
    ensure_instance(client, args.machine_id, args.host, args.port)
    machine_id = str(args.machine_id)
    try:
        if args.setup:
            setup(client, machine_id)
        elif not healthy(client, machine_id):
            missing = missing_models(client, machine_id)
            if missing:
                die(f"ComfyUI is missing {len(missing)} model files; rerun with --setup")
            start_server(client, machine_id)

        started = time.time()
        files = submit(client, machine_id, graph, args.timeout)
        saved = fetch(client, machine_id, files, Path(args.out_dir))
        elapsed = (time.time() - started) / 60
        log(f"done in {elapsed:.1f} min (~{A100_INR_PER_HOUR * elapsed / 60:.0f} INR)")
        for path in saved:
            log(f"  {path}")
        return 0 if saved else 1
    finally:
        if args.pause:
            log(f"pausing {machine_id}")
            client.pause(machine_id)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Provision a JarvisLabs GPU, generate images with MOSS-Image, then destroy it.

This is a standalone runner for a *separate* model family: MOSS-Image is a
text-to-image diffusion model, so its LoRAs cannot be loaded by the Wan video
pipeline. Nothing here is wired into the FastAPI app.

    # plan and cost only
    python scripts/moss_image_runner.py --prompt "moss_xvstyle, ..." --dry-run

    # run, then auto-destroy
    python scripts/moss_image_runner.py --prompt "moss_xvstyle, ..." --count 4 --yes

The instance is destroyed in `finally`. If the process is interrupted, check
`jl list` and destroy anything left behind.
"""

from __future__ import annotations

import argparse
import base64
import re
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instance_adopt import adopt_instance  # noqa: E402

from server.jarvis import JarvisLabsClient  # noqa: E402

REMOTE_DIR = "/home/vsp_jobs"
HELPER = ROOT / "scripts" / "moss_image_remote.py"
REMOTE_HELPER = f"{REMOTE_DIR}/{HELPER.name}"
MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"
LORA_REPO = "axiomofmind/Moss-image-nsfw-uncensored-lora"
LORA_FILE = "pytorch_lora_weights.safetensors"

GPU_PRICES_INR = {"A100": 84.24, "A100-80GB": 140.94, "A30": 38.88, "L4": 41.31}
# Base model is 27 GB, plus the LoRA, plus the diffusers wheel cache.
REQUIRED_DISK_GB = 45
# `peft` is required by `load_lora_weights`; the rest are pipeline imports.
PACKAGES = (
    "-U diffusers transformers accelerate peft huggingface_hub pillow sentencepiece"
)


def log(message: str) -> None:
    print(f"[moss] {message}", flush=True)


def die(message: str) -> None:
    log(f"ERROR: {message}")
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-Image generation on JarvisLabs")
    parser.add_argument("--prompt", required=True, help="Include the 'moss_xvstyle' trigger")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--count", type=int, default=1, help="Images to generate")
    parser.add_argument("--size", type=int, default=1024, help="Square size, multiple of 16")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--skip-lora", action="store_true", help="Run the base model only")
    parser.add_argument("--no-offload", action="store_true", help="Keep weights on the GPU")
    parser.add_argument("--gpu", default="A100", choices=sorted(GPU_PRICES_INR))
    parser.add_argument("--template", default="pytorch")
    parser.add_argument("--disk-gb", type=int, default=80, help="Instance scratch disk")
    parser.add_argument("--machine-id", help="Adopt an existing instance instead of creating one")
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--destroy-adopted", action="store_true")
    parser.add_argument("--skip-setup", action="store_true", help="Assume pip deps exist")
    parser.add_argument("--timeout", type=int, default=60, help="Job timeout in minutes")
    parser.add_argument("--boot-timeout", type=int, default=600)
    parser.add_argument("--out-dir", default=str(ROOT / "public" / "images"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    return parser.parse_args()


def estimate(args: argparse.Namespace) -> tuple[float, float]:
    """Rough (minutes, INR). Download and setup dominate a small run."""
    per_image = 1.4 if not args.no_offload else 0.9
    minutes = 4 + 12 + args.count * per_image
    return minutes, GPU_PRICES_INR[args.gpu] * minutes / 60


def ensure_instance(client: JarvisLabsClient, args: argparse.Namespace) -> tuple[str, bool]:
    if args.machine_id:
        wanted = str(args.machine_id)
        inst = adopt_instance(client, wanted, "moss")
        if inst is not None:
            log(f"adopting instance {wanted} ({inst.get('host')})")
            return wanted, False
        die(f"instance {wanted} not found; `jl list` shows what is running")

    log(f"creating {args.gpu} instance ({args.disk_gb} GB disk)...")
    info = client.create_instance(
        gpu_type=args.gpu,
        name=f"moss-{int(time.time())}",
        hdd=args.disk_gb,
        template=args.template,
    )
    return str(info["machine_id"]), True


def wait_ready(client: JarvisLabsClient, machine_id: str, args: argparse.Namespace) -> None:
    log("waiting for SSH...")
    if not client.wait_for_port(22, timeout=args.boot_timeout, instance_id=machine_id):
        die("instance never accepted SSH connections")
    code, out = client.exec(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1",
        instance_id=machine_id,
        stream=False,
    )
    gpu = out.strip()
    log(f"gpu: {gpu or 'unknown'}")
    if gpu and "A100" not in gpu.upper():
        log("WARNING: not an A100; the model is ~27 GB and may OOM")


def check_disk(client: JarvisLabsClient, machine_id: str) -> None:
    code, out = client.exec(
        "df -BG --output=avail /home | tail -1", instance_id=machine_id, stream=False
    )
    digits = re.sub(r"\D", "", out)
    available = int(digits) if digits else 0
    log(f"disk available on /home: {available} GB")
    if available < REQUIRED_DISK_GB:
        die(f"need at least {REQUIRED_DISK_GB} GB free; destroy and retry with --disk-gb")


def install_runtime(client: JarvisLabsClient, machine_id: str) -> None:
    log("installing diffusers/transformers (2-4 minutes)...")
    code, out = client.exec(
        f"python3 -m pip install -q {PACKAGES}", instance_id=machine_id, stream=False
    )
    if code != 0:
        die(f"pip install failed:\n{out[-800:]}")


def _run_python(client: JarvisLabsClient, machine_id: str, code: str) -> tuple[int, str]:
    """Run a Python snippet remotely without shell-quoting hazards."""
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return client.exec(
        f"echo {payload} | base64 -d | python3 -", instance_id=machine_id, stream=False
    )


def download_lora(client: JarvisLabsClient, machine_id: str) -> str:
    """Fetch only the 70 MB LoRA file, not the repo's 27 GB merged copy."""
    destination = f"{REMOTE_DIR}/lora/moss"
    code = (
        "from huggingface_hub import hf_hub_download\n"
        f"print(hf_hub_download(repo_id={LORA_REPO!r}, filename={LORA_FILE!r}, "
        f"local_dir={destination!r}))\n"
    )
    log("downloading LoRA (70 MB)...")
    status, out = _run_python(client, machine_id, code)
    if status != 0:
        die(f"LoRA download failed:\n{out[-800:]}")
    return destination


def upload_helper(client: JarvisLabsClient, machine_id: str) -> None:
    client.exec(f"mkdir -p {REMOTE_DIR}", instance_id=machine_id, stream=False)
    if not client.sync_up(str(HELPER), remote_path=REMOTE_HELPER, instance_id=machine_id):
        die("could not upload the remote helper")


def build_command(args: argparse.Namespace, lora_dir: str) -> str:
    parts = [
        f"python3 -u {shlex.quote(REMOTE_HELPER)}",
        "--prompt", shlex.quote(args.prompt),
        "--count", str(args.count),
        "--width", str(args.size),
        "--height", str(args.size),
        "--steps", str(args.steps),
        "--cfg", str(args.cfg),
        "--seed", str(args.seed),
        "--model-id", shlex.quote(MODEL_ID),
        "--out-dir", f"{REMOTE_DIR}/moss_out",
    ]
    if args.negative_prompt:
        parts += ["--negative-prompt", shlex.quote(args.negative_prompt)]
    if lora_dir:
        parts += ["--lora-dir", shlex.quote(lora_dir), "--lora-scale", str(args.lora_scale)]
    if args.no_offload:
        parts.append("--no-offload")
    return " ".join(parts)


def follow(
    client: JarvisLabsClient, machine_id: str, log_path: str, deadline: float
) -> tuple[list[str], bool]:
    """Tail the detached job's log, printing new lines and collecting outputs."""
    seen = 0
    saved: list[str] = []
    while time.time() < deadline:
        code, out = client.exec(
            f"tail -n +{seen + 1} {shlex.quote(log_path)} 2>/dev/null || true",
            instance_id=machine_id,
            stream=False,
        )
        # `tail -n +N` counts physical lines, so the offset must too: a blank
        # line would otherwise make the next poll re-read or skip output.
        raw = out.splitlines()
        seen += len(raw)
        lines = [line for line in raw if line.strip()]
        for line in lines:
            log(f"  {line}")
            if line.startswith("=== IMAGE_SAVED:"):
                saved.append(line.split(":", 1)[1].strip())

        if any("=== MOSS_SUCCESS" in line for line in lines):
            return saved, True
        if any("Traceback (most recent call last)" in line for line in lines):
            return saved, False
        time.sleep(10)
    return saved, False


def fetch(client: JarvisLabsClient, machine_id: str, paths: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    local: list[Path] = []
    for remote in paths:
        target = out_dir / Path(remote).name
        if client.sync_down(remote, local_path=str(target), instance_id=machine_id):
            local.append(target)
        else:
            log(f"WARNING: could not download {remote}")
    return local


def main() -> int:
    args = parse_args()
    minutes, cost = estimate(args)

    log("MOSS-Image plan:")
    log(f"  GPU:      {args.gpu} ({GPU_PRICES_INR[args.gpu]:.2f} INR/hour)")
    log(f"  Size:     {args.size}x{args.size}, {args.steps} steps, cfg {args.cfg}")
    log(f"  Images:   {args.count}")
    log(f"  LoRA:     {'skipped' if args.skip_lora else f'scale {args.lora_scale}'}")
    log(f"  Estimate: ~{minutes:.0f} min, ~{cost:.0f} INR")
    log(f"  Output:   {args.out_dir}")

    if args.dry_run:
        return 0
    if not args.yes:
        answer = input("Proceed and provision a GPU? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return 1

    client = JarvisLabsClient()
    if not client.token:
        die("no JarvisLabs token; set JARVIS_LABS_API_TOKEN or run `jl setup`")

    machine_id = ""
    created = False
    started = time.time()
    try:
        machine_id, created = ensure_instance(client, args)
        wait_ready(client, machine_id, args)
        check_disk(client, machine_id)
        if not args.skip_setup:
            install_runtime(client, machine_id)
        upload_helper(client, machine_id)
        lora_dir = "" if args.skip_lora else download_lora(client, machine_id)

        log_path = f"{REMOTE_DIR}/moss_{int(started)}.log"
        client.spawn(
            build_command(args, lora_dir),
            log_path=log_path,
            instance_id=machine_id,
            workdir=REMOTE_DIR,
        )
        log(f"job running detached; log: {log_path}")

        saved, ok = follow(client, machine_id, log_path, time.time() + args.timeout * 60)
        if not ok:
            log(f"job did not finish cleanly; the remote log is {log_path}")
            return 1

        files = fetch(client, machine_id, saved, Path(args.out_dir))
        elapsed = (time.time() - started) / 60
        log(f"done: {len(files)} image(s) in {elapsed:.1f} min (~{GPU_PRICES_INR[args.gpu] * elapsed / 60:.0f} INR)")
        for path in files:
            log(f"  {path}")
        return 0
    finally:
        should_destroy = created and not args.keep_instance
        if args.machine_id and args.destroy_adopted:
            should_destroy = True
        if should_destroy and machine_id:
            log(f"destroying {machine_id}")
            client.destroy(machine_id)
        elif machine_id:
            log(f"instance {machine_id} left running — destroy it with `jl destroy {machine_id}`")


if __name__ == "__main__":
    sys.exit(main())

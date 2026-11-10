#!/usr/bin/env python3
"""
VSP Video Agent — Command Line Interface for Agents & Developers.
Zero external dependencies (uses stdlib argparse and urllib).
"""

import sys
import os
import argparse
import urllib.request
import urllib.error
import json
import time
from pathlib import Path

BASE_URL = os.environ.get("VIBE_VIDEO_API_URL", "http://localhost:8000")


def _jarvis_client():
    """Load the vendored JarvisLabs harness (no HTTP backend needed)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from server.jarvis import JarvisLabsClient

    return JarvisLabsClient()


def _row(status, name, detail):
    icon = {"ok": "[ok]", "warn": "[warn]", "fail": "[FAIL]"}.get(status, "[?]")
    print(f"  {icon:<7} {name:<16}  {detail}")


def cmd_gpu_doctor(args):
    """Pre-flight: token, SDK, ssh, instance, persistent filesystem, live GPU."""
    import shutil as _shutil

    client = _jarvis_client()
    failed = 0

    token = getattr(client, "token", None)
    if token:
        _row("ok", "api token", f"found (len {len(token)})")
    else:
        failed += 1
        _row("fail", "api token", "none — set JARVIS_LABS_API_TOKEN or run `jl setup`")

    for binary in ("ssh", "scp", "curl", "jl"):
        found = _shutil.which(binary)
        _row("ok" if found else "warn", f"bin:{binary}", found or "not on PATH")

    inst = client.get_active_instance()
    if not inst:
        _row("warn", "active instance", "none — create/adopt an A100 first")
    else:
        mid = str(inst.get("machine_id"))
        status = str(inst.get("status", "Unknown"))
        _row(
            "ok" if status.lower() == "running" else "warn",
            "active instance",
            f"{mid} {inst.get('gpu_type', 'GPU')} [{status}] {inst.get('host')}",
        )
        if status.lower() == "running":
            code, out = client.exec(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1",
                instance_id=mid,
                stream=False,
            )
            gpu = out.strip().splitlines()[0] if out.strip() else "no response"
            ok = code == 0 and "no response" not in gpu
            if not ok:
                failed += 1
            _row("ok" if ok else "fail", "ssh transport", gpu)
            if ok and "A100" not in gpu.upper():
                _row("warn", "gpu model", "not an A100 — video generation will refuse to start")
            code, out = client.exec(
                "df -h /home | tail -1 | awk '{print $5\" used, \"$4\" free\"}'",
                instance_id=mid,
                stream=False,
            )
            _row("ok", "disk /home", out.strip() or "unknown")

    print()
    if failed:
        print(f"✖ {failed} check(s) failed — fix these before provisioning a GPU.")
        sys.exit(1)
    print("✔ All critical checks passed.")


def cmd_gpu_audit(args):
    """Verify remote files exist (following symlinks) and are writable."""
    import shlex as _shlex

    client = _jarvis_client()
    paths = list(args.path)
    script = r"""
for p in "$@"; do
  if [ -L "$p" ]; then kind=link
  elif [ -e "$p" ]; then kind=file
  else kind=missing; fi
  size=$(stat -Lc %s "$p" 2>/dev/null || echo 0)
  raw=$(stat -c %s "$p" 2>/dev/null || echo 0)
  parent=$(dirname "$p")
  if touch "$parent/.vsp_write_probe" 2>/dev/null; then w=rw; rm -f "$parent/.vsp_write_probe"; else w=ro; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$size" "$raw" "$w" "$p"
done
"""
    quoted = " ".join(_shlex.quote(p) for p in paths)
    code, out = client.exec(
        f"bash -s -- {quoted} <<'VSP_AUDIT'\n{script}\nVSP_AUDIT",
        stream=False,
    )
    if code != 0:
        print("✖ Remote audit failed; is the instance reachable?")
        sys.exit(1)

    mins = {}
    for spec in args.min_size or []:
        key, _, val = spec.rpartition("=")
        try:
            mins[key] = int(val)
        except ValueError:
            print(f"✖ Invalid --min-size: {spec}")
            sys.exit(2)

    bad = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        kind, size_s, raw_s, writable, path = parts
        size, raw = int(size_s or 0), int(raw_s or 0)
        notes, status = [], "ok"
        if kind == "missing":
            status, _ = "fail", notes.append("MISSING")
        elif size == 0:
            status, _ = "fail", notes.append("broken symlink or empty file")
        if kind == "link" and size != raw:
            notes.append(f"symlink (link len {raw} vs target {size})")
        need = mins.get(path)
        if need is not None and size < need:
            status = "fail"
            notes.append(f"below required {need} bytes")
        if writable == "ro":
            status = "fail" if kind == "missing" else "warn"
            notes.append("parent dir is READ-ONLY")
        if status == "fail":
            bad += 1
        gb = size / (1024 ** 3)
        size_txt = f"{gb:.2f} GB" if gb >= 1 else f"{size / (1024 ** 2):.1f} MB" if size else "0 B"
        _row(status, size_txt, f"{path}  {'; '.join(notes)}")

    print()
    if bad:
        print(f"✖ {bad} path(s) are unusable.")
        sys.exit(1)
    print(f"✔ All {len(paths)} path(s) verified.")


def cmd_gpu_wait(args):
    """Block until a remote port or HTTP endpoint is ready."""
    client = _jarvis_client()
    if args.http:
        ok = client.wait_for_http(
            args.port, args.path, timeout=args.timeout, interval=args.interval, instance_id=args.id
        )
    else:
        ok = client.wait_for_port(
            args.port, timeout=args.timeout, interval=args.interval, instance_id=args.id
        )
    if ok:
        print(f"✔ Port {args.port} is ready.")
    else:
        print(f"✖ Port {args.port} did not become ready within {args.timeout}s.")
        sys.exit(1)


def cmd_gpu_instances(args):
    """List JarvisLabs instances straight from the SDK."""
    client = _jarvis_client()
    instances = client.list_instances()
    if not instances:
        print("No JarvisLabs instances found.")
        return
    for inst in instances:
        print(
            f"• {inst.get('machine_id')}  {inst.get('name')}  "
            f"{inst.get('gpu_type')}  [{inst.get('status')}]  {inst.get('host')}"
        )

def api_get(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "VibeVideoCLI/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error connecting to backend at {url}: {e}")
        sys.exit(1)

def api_post(endpoint: str, data: dict):
    url = f"{BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VibeVideoCLI/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error calling {url}: {e}")
        sys.exit(1)

def cmd_status(args):
    print("=== Checking VSP Video Agent & JarvisLabs GPU Status ===")
    health = api_get("/api/health")
    gpu = api_get("/api/gpu/status")

    print(f"Backend Service:  {health.get('status')}")
    print(f"Transport:        {health.get('transport')}")
    print(f"GPU Connected:    {health.get('gpu_connected')}")
    print(f"Instance:         {health.get('instance_id') or 'none'}")
    print(f"Detected GPU:     {health.get('gpu') or 'None / CPU'}")
    print(f"A100 Ready:       {'✅ YES (A100 Verified)' if health.get('a100_ready') else '❌ NO (A100 Required)'}")
    print("-" * 50)
    print(f"GPU Status:       {gpu.get('connected')}")
    print(f"GPU Error:        {gpu.get('error') or 'none'}")

def cmd_list(args):
    videos = api_get("/api/video/gallery")
    print(f"=== Local Video Gallery ({len(videos)} videos) ===")
    if not videos:
        print("No videos generated yet.")
        return

    for v in videos:
        print(f"• {v['filename']} | {v['size_mb']} MB | {v['duration']}s | {v['type']}")
        print(f"  Prompt: \"{v['prompt']}\"")
        print(f"  URL:    {BASE_URL}{v['url']}")
        print()

def cmd_generate(args):
    print("=== Checking Hardware Pre-flight: NVIDIA A100 GPU ===")
    health = api_get("/api/health")
    if not health.get("gpu_connected"):
        print("❌ STOPPING EXECUTION: no running JarvisLabs GPU instance.")
        print("👉 Provision or adopt an A100, then re-run this command.")
        sys.exit(1)

    if not health.get("a100_ready"):
        gpu = health.get("gpu") or "Unknown / CPU"
        print(f"❌ STOPPING EXECUTION: NVIDIA A100 GPU is strictly required for video generation.")
        print(f"   Currently detected runtime: {gpu}")
        print("👉 Switch the JarvisLabs instance to an A100.")
        sys.exit(1)

    print(f"✅ Verified Hardware: {health.get('gpu')}")
    print(f"=== Dispatching Video Generation to JarvisLabs A100 ===")
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Model: Wan 2.1 {args.model}")
    print(f"Aspect Ratio: {args.ratio}, Duration: {args.duration}s, Steps: {args.steps}")

    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative or "low quality, blurry, distorted, artifacts",
        "aspect_ratio": args.ratio,
        "duration": args.duration,
        "steps": args.steps,
        "model_size": args.model,
        "seed": args.seed or int(time.time()) % 100000
    }

    res = api_post("/api/video/generate/t2v", payload)
    job_id = res.get("job_id")
    filename = res.get("filename")
    print(f"Job queued successfully: {job_id}")
    print(f"Target filename: {filename}")

    if not args.wait:
        print("Job submitted to queue. Use `python cli.py jobs` to monitor status.")
        return

    print("Waiting for generation on A100 GPU...")
    while True:
        time.sleep(5)
        job = api_get(f"/api/video/jobs/{job_id}")
        status = job.get("status")
        logs = job.get("logs", [])
        last_log = logs[-1] if logs else ""
        print(f"[{status.upper()}] {last_log}")

        if status == "completed":
            print("\n🎉 Generation finished successfully!")
            print(f"Video URL: {BASE_URL}{job.get('video_url')}")
            break
        elif status == "failed":
            print(f"\n❌ Generation failed: {job.get('error')}")
            sys.exit(1)

def cmd_jobs(args):
    jobs = api_get("/api/video/jobs")
    print(f"=== Active & Recent Jobs ({len(jobs)}) ===")
    for j in jobs:
        print(f"Job [{j['id']}] - Status: {j['status'].upper()} | Type: {j['type']}")
        if j.get("prompt"):
            print(f"  Prompt: \"{j['prompt']}\"")
        if j.get("logs"):
            print(f"  Last Log: {j['logs'][-1]}")
        print()

def cmd_stitch(args):
    if len(args.videos) < 2:
        print("Error: At least 2 video filenames are required to stitch.")
        sys.exit(1)

    print(f"=== Stitching {len(args.videos)} Videos with FFmpeg ===")
    for v in args.videos:
        print(f"  + {v}")

    payload = {
        "video_filenames": args.videos,
        "output_filename": args.output or f"movie_{int(time.time())}.mp4"
    }

    res = api_post("/api/video/stitch", payload)
    print("\n🎉 Stitching Complete!")
    print(f"Movie saved: {res.get('filename')} ({res.get('size_mb')} MB)")
    print(f"URL: {BASE_URL}{res.get('url')}")

def cmd_clean_cache(args):
    print("=== Cleaning GPU Checkpoint Cache to Free Disk Space ===")
    res = api_post("/api/video/clean-cache", {})
    print("Clean result:", res.get("status"))
    if res.get("output"):
        print(res.get("output"))

def main():
    parser = argparse.ArgumentParser(
        description="VSP Video Agent CLI — Manage AI video generation on A100 GPU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # clean-cache
    p_clean = subparsers.add_parser("clean-cache", help="Clean unused checkpoints on the GPU box to prevent disk full warnings")
    p_clean.set_defaults(func=cmd_clean_cache)

    # status
    p_status = subparsers.add_parser("status", help="Check backend and JarvisLabs A100 GPU health")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = subparsers.add_parser("list", help="List all generated videos in local gallery")
    p_list.set_defaults(func=cmd_list)

    # jobs
    p_jobs = subparsers.add_parser("jobs", help="List status of generation jobs")
    p_jobs.set_defaults(func=cmd_jobs)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate a video from a text prompt on a JarvisLabs A100")
    p_gen.add_argument("prompt", type=str, help="Prompt describing scene and motion")
    p_gen.add_argument("--model", type=str, default="1.3B", choices=["1.3B", "14B"], help="Wan 2.1 model size (1.3B or 14B)")
    p_gen.add_argument("--ratio", type=str, default="16:9", choices=["16:9", "9:16", "1:1"], help="Aspect ratio")
    p_gen.add_argument("--duration", type=int, default=5, choices=[3, 5, 10], help="Duration in seconds")
    p_gen.add_argument("--steps", type=int, default=25, help="Diffusion inference steps")
    p_gen.add_argument("--seed", type=int, default=None, help="Random seed")
    p_gen.add_argument("--negative", type=str, default=None, help="Negative prompt")
    p_gen.add_argument("--wait", action="store_true", default=True, help="Wait for completion and stream logs")
    p_gen.add_argument("--no-wait", dest="wait", action="store_false", help="Return immediately after queuing")
    p_gen.set_defaults(func=cmd_generate)

    # stitch
    p_stitch = subparsers.add_parser("stitch", help="Concatenate multiple video clips into a single movie")
    p_stitch.add_argument("videos", nargs="+", help="Video filenames to stitch in order")
    p_stitch.add_argument("-o", "--output", type=str, default=None, help="Output movie filename")
    p_stitch.set_defaults(func=cmd_stitch)

    # gpu — direct JarvisLabs control, no HTTP backend required
    p_gpu = subparsers.add_parser("gpu", help="JarvisLabs pre-flight and direct instance control")
    gpu_subs = p_gpu.add_subparsers(dest="gpu_command", required=True)

    g_inst = gpu_subs.add_parser("instances", help="List JarvisLabs instances")
    g_inst.set_defaults(func=cmd_gpu_instances)

    g_doc = gpu_subs.add_parser("doctor", help="Pre-flight: token, ssh, instance, live GPU")
    g_doc.set_defaults(func=cmd_gpu_doctor)

    g_aud = gpu_subs.add_parser("audit", help="Verify remote files follow symlinks and are writable")
    g_aud.add_argument("path", nargs="+", help="Remote paths to verify")
    g_aud.add_argument("--min-size", action="append", metavar="PATH=BYTES", help="Fail if smaller than BYTES")
    g_aud.set_defaults(func=cmd_gpu_audit)

    g_wait = gpu_subs.add_parser("wait", help="Wait until a remote port or HTTP endpoint is ready")
    g_wait.add_argument("port", type=int, help="Remote TCP port (e.g. 8188 for ComfyUI)")
    g_wait.add_argument("--http", action="store_true", help="Probe HTTP instead of a raw socket")
    g_wait.add_argument("--path", default="/", help="HTTP path to probe (default: /)")
    g_wait.add_argument("--timeout", type=int, default=600, help="Seconds to wait (default: 600)")
    g_wait.add_argument("--interval", type=int, default=5, help="Seconds between probes (default: 5)")
    g_wait.add_argument("--id", help="Instance ID to target")
    g_wait.set_defaults(func=cmd_gpu_wait)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

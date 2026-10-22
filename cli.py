#!/usr/bin/env python3
"""
Vibe Video Studio — Command Line Interface for Agents & Developers.
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
    print("=== Checking Vibe Video Studio & Colab GPU Status ===")
    health = api_get("/api/health")
    colab = api_get("/colab/status")

    print(f"Backend Service:  {health.get('status')}")
    print(f"vspagent Harness: {health.get('vspagent_available')}")
    print(f"Colab Connected:  {health.get('colab_connected')}")
    print(f"Detected GPU:     {health.get('colab_gpu') or 'None / CPU'}")
    print(f"A100 Ready:       {'✅ YES (A100 Verified)' if health.get('a100_ready') else '❌ NO (A100 Required)'}")
    print("-" * 50)
    print(f"Colab Connection: {colab.get('status')}")
    print(f"Colab Message:    {colab.get('message')}")
    print(f"Colab GPU Info:   {colab.get('gpu_info') or 'GPU detected & active'}")

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
    if not health.get("colab_connected"):
        print("❌ STOPPING EXECUTION: Google Colab is not connected.")
        print("👉 Please connect to your Google Colab instance in Chrome first.")
        sys.exit(1)

    if not health.get("a100_ready"):
        gpu = health.get("colab_gpu") or "Unknown / CPU"
        print(f"❌ STOPPING EXECUTION: NVIDIA A100 GPU is strictly required for video generation.")
        print(f"   Currently detected runtime: {gpu}")
        print("👉 Please switch to an A100 GPU in Colab:")
        print("   Colab -> Runtime -> Change runtime type -> Hardware accelerator: GPU -> A100 -> Save.")
        sys.exit(1)

    print(f"✅ Verified Hardware: {health.get('colab_gpu')}")
    print(f"=== Dispatching Video Generation to Colab A100 ===")
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
    print("=== Cleaning Colab Checkpoint Cache to Free Disk Space ===")
    res = api_post("/api/video/clean-cache", {})
    print("Clean result:", res.get("status"))
    if res.get("output"):
        print(res.get("output"))

def main():
    parser = argparse.ArgumentParser(
        description="Vibe Video Studio CLI — Manage AI video generation on A100 GPU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # clean-cache
    p_clean = subparsers.add_parser("clean-cache", help="Clean unused checkpoints in Colab cache to prevent disk full warnings")
    p_clean.set_defaults(func=cmd_clean_cache)

    # status
    p_status = subparsers.add_parser("status", help="Check backend and Colab A100 GPU health")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = subparsers.add_parser("list", help="List all generated videos in local gallery")
    p_list.set_defaults(func=cmd_list)

    # jobs
    p_jobs = subparsers.add_parser("jobs", help="List status of generation jobs")
    p_jobs.set_defaults(func=cmd_jobs)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate a video from a text prompt on Colab A100")
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

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

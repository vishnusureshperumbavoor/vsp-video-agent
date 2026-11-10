#!/usr/bin/env python3
"""Runs on the JarvisLabs GPU box. Generates images with MOSS-Image + a LoRA.

This file is uploaded by scripts/moss_image_runner.py and executed detached, so
it must stay dependency-light and print parseable markers:

    === MOSS_MODEL_READY
    === IMAGE_SAVED: /path/to/image.png
    === MOSS_SUCCESS: 3 images in 91.4s

Sampling defaults follow the OpenMOSS model card: flow-matching Euler, 50 steps,
CFG 3.0, negative-branch layer skip at index 22.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-dir", default="")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--model-id", default="OpenMOSS-Team/MOSS-Image-SFT")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--out-dir", default="/home/vsp_jobs/moss_out")
    parser.add_argument("--skip-layers", default="22")
    parser.add_argument("--no-offload", action="store_true")
    args = parser.parse_args()

    if args.width % 16 or args.height % 16:
        raise SystemExit("width and height must be positive multiples of 16")
    args.skip_layers = [int(v) for v in args.skip_layers.split(",") if v.strip()]
    return args


def build_pipeline(args):
    import torch
    from diffusers import DiffusionPipeline

    source = args.model_dir or args.model_id
    print(f"=== MOSS_LOADING: {source}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        source,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)

    if args.lora_dir:
        print(f"=== MOSS_LORA: {args.lora_dir} @ {args.lora_scale}", flush=True)
        try:
            pipe.load_lora_weights(
                args.lora_dir, weight_name="pytorch_lora_weights.safetensors"
            )
        except ImportError as exc:  # peft missing on the box
            raise SystemExit(f"LoRA needs peft installed: {exc}") from exc
        if args.lora_scale != 1.0:
            adapters = list(pipe.get_active_adapters())
            pipe.set_adapters(adapters, adapter_weights=[args.lora_scale] * len(adapters))

    if args.no_offload:
        pipe.to("cuda")
    else:
        # ~27 GB of weights: offload keeps a 40 GB A100 clear of OOM.
        pipe.enable_model_cpu_offload()
    return pipe, torch


def main() -> int:
    args = parse_args()
    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    pipe, torch = build_pipeline(args)
    print("=== MOSS_MODEL_READY", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    for index in range(args.count):
        kwargs = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.cfg,
            "generator": torch.Generator("cpu").manual_seed(args.seed + index),
        }
        if args.negative_prompt:
            kwargs["negative_prompt"] = args.negative_prompt
            kwargs["uncond_skip_layers"] = []
        elif args.skip_layers:
            kwargs["uncond_skip_layers"] = args.skip_layers

        print(f"=== MOSS_IMAGE_START: {index + 1}/{args.count}", flush=True)
        image = pipe(**kwargs).images[0]
        path = os.path.join(args.out_dir, f"moss_{stamp}_{index:02d}.png")
        image.save(path)
        saved += 1
        print(f"=== IMAGE_SAVED: {path}", flush=True)

    elapsed = time.time() - started
    print(f"=== MOSS_SUCCESS: {saved} images in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

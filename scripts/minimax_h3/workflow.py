"""Pure builders for MiniMax-H3 text-to-video+audio jobs.

MiniMax H3 denoises video and stereo audio in one pass. The GPU side is
ComfyUI; nothing in this module imports torch, requests or the SSH client.
Keep it that way so the graph can be tested on a laptop.
"""

from __future__ import annotations

import math

import realism

MODEL_REPO = "Comfy-Org/MiniMax-H3"

# Quantized Comfy-Org repack. int8_convrot needs torch with CUDA 13.
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# repo path -> size in GB (for the disk check and the plan output)
MODEL_FILES = {
    f"diffusion_models/{UNET}": 21.0,
    f"text_encoders/{TEXT_ENCODER}": 15.7,
    f"vae/{VIDEO_VAE}": 2.8,
    f"vae/{AUDIO_VAE}": 0.6,
    f"loras/{TURBO_LORA}": 2.0,
}

DEFAULT_WIDTH = 736
DEFAULT_HEIGHT = 416
DEFAULT_STEPS = 8
DEFAULT_SECONDS = 5.0
DEFAULT_SEED = 20260803
FRAME_GRID = 17
FPS = 24
A100_INR_PER_HOUR = 84.24


def snap_length(seconds: float) -> int:
    """Snap a duration to the model's frame grid (17k+5 frames at 24 fps)."""
    frames = max(5, round(seconds * FPS))
    return frames + (5 - frames % FRAME_GRID) % FRAME_GRID


def validate_size(width: int, height: int) -> tuple[int, int]:
    for value in (width, height):
        if value < 32 or value % 32:
            raise ValueError(f"{value} is not a positive multiple of 32")
    return width, height


def compose_prompt(
    scene: str,
    dialogue_line: str = "",
    language: str = "English",
    raw: bool = False,
) -> str:
    """Build an H3 prompt: scene, optional spoken line, optional film cues."""
    prompt = scene.strip()
    if not prompt:
        raise ValueError("prompt is empty")
    if dialogue_line:
        prompt = f"{prompt} {realism.dialogue(dialogue_line, language)}"
    if raw:
        prompt = realism.enhance(prompt, "video")
    return prompt


def resolve_settings(
    dialogue_line: str = "",
    steps: int | None = None,
    lora_strength: float | None = None,
) -> tuple[int, float]:
    """Dialogue needs more steps and no turbo adapter; motion defaults are fine."""
    resolved_steps = steps if steps is not None else (20 if dialogue_line else DEFAULT_STEPS)
    resolved_lora = (
        lora_strength if lora_strength is not None else (0.0 if dialogue_line else 1.0)
    )
    return resolved_steps, resolved_lora


def build_graph(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = 124,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    lora_strength: float = 1.0,
    prefix: str = "minimax_h3/video",
    first_frame: str = "",
) -> dict:
    """Return a ComfyUI API graph (node id -> class_type + inputs)."""
    if not prompt.strip():
        raise ValueError("prompt is empty")
    if length < 5:
        raise ValueError("length must be at least 5 frames")
    if steps < 1:
        raise ValueError("steps must be positive")
    validate_size(width, height)
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": TURBO_LORA,
                "strength_model": lora_strength,
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax"},
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "7": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["2", 0], "conditioning": ["6", 0]},
        },
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["2", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["6", 1],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
        },
        "13": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["11", 0], "vae": ["5", 0]},
        },
        "14": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["12", 0], "fps": FPS, "audio": ["13", 0]},
        },
        "15": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["14", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if first_frame:
        graph["16"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        graph["6"]["inputs"]["first_frame"] = ["16", 0]
    return graph


def estimate(seconds: float, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
             steps: int = DEFAULT_STEPS) -> tuple[float, float]:
    """Rough (minutes, INR) for a render on a 40 GB A100.

    Calibrated at 736x416 / 8 steps / 5.2 s: 21 s of sampling and 23 s of
    model load, VAE decode and encode. Sampling scales with pixels, frames and
    steps; overhead is constant.
    """
    length = snap_length(seconds)
    pixels = (width * height) / (DEFAULT_WIDTH * DEFAULT_HEIGHT)
    sampling = 2.7 * steps * pixels * (length / 124)
    minutes = (sampling + 30) / 60 + 2.0  # + model load on the first run
    return math.ceil(minutes * 10) / 10, A100_INR_PER_HOUR * minutes / 60

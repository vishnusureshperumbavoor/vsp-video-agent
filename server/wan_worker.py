"""
Wan 2.1 video generation payloads for a JarvisLabs A100.

Emits standalone Python scripts that are staged on the GPU box and run detached
via `setsid`. The scripts are transport-agnostic: they only need a working
diffusers install and a writable output directory.
"""

# Remote working directory for staged jobs, logs and outputs.
REMOTE_JOB_DIR = "/home/vsp_jobs"


def generate_cache_cleanup_code() -> str:
    """Remove dangling blobs and unused checkpoints to keep disk usage healthy."""
    return """
import os, shutil, subprocess

hf_hub = "/root/.cache/huggingface/hub"
print("=== GPU Disk Preflight Check ===")
subprocess.run(["df", "-h", "/home"], check=False)

if os.path.exists(hf_hub):
    # Free the older T2V cache when both T2V and I2V checkpoints exist.
    t2v_14b = os.path.join(hf_hub, "models--Wan-AI--Wan2.1-T2V-14B-Diffusers")
    if os.path.exists(t2v_14b):
        print("Clearing older T2V 14B checkpoint to keep disk healthy...")
        shutil.rmtree(t2v_14b, ignore_errors=True)

print("=== GPU Disk After Cleanup ===")
subprocess.run(["df", "-h", "/home"], check=False)
"""


def generate_gpu_setup_code() -> str:
    """Install/refresh the Wan 2.1 runtime on the GPU box."""
    return """
import sys, subprocess

print("=== Checking GPU Environment ===")
subprocess.run(["nvidia-smi"], check=True)

# Remove incompatible torchao if present (FqnToConfig import errors).
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

packages = [
    "git+https://github.com/huggingface/diffusers.git",
    "transformers>=4.48.0",
    "accelerate>=0.34.0",
    "sentencepiece",
    "imageio[ffmpeg]",
    "torchvision",
    "ftfy",
]

print("Installing / updating required packages for Wan 2.1 video generation...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
print("=== GPU Environment Ready for Video Generation ===")
"""


def generate_t2v_script(
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, flickering, artifacts",
    width: int = 832,
    height: int = 480,
    num_frames: int = 49,
    num_inference_steps: int = 25,
    guidance_scale: float = 6.0,
    seed: int = 42,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    output_filename: str = "generated_t2v.mp4",
    output_dir: str = REMOTE_JOB_DIR,
) -> str:
    # Wan requires num_frames to follow 4k + 1.
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        if num_frames < 17:
            num_frames = 17

    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')
    use_offload = "14B" in model_id
    output_path = f"{output_dir.rstrip('/')}/{output_filename}"

    return f"""import sys, os, gc, time
import torch

for k in list(sys.modules.keys()):
    if "torchao" in k:
        del sys.modules[k]

from diffusers import WanPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Text-to-Video Generation on A100 ===", flush=True)
print("Prompt: {prompt_escaped}", flush=True)
print("Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}", flush=True)

t0 = time.time()
gc.collect()
torch.cuda.empty_cache()

model_id = "{model_id}"
print(f"Loading {{model_id}} in bfloat16...", flush=True)

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
{"pipe.enable_model_cpu_offload()" if use_offload else 'pipe.to("cuda")'}

generator = torch.Generator(device="cuda").manual_seed({seed})

print("Generating video frames on A100 GPU...", flush=True)
output = pipe(
    prompt="{prompt_escaped}",
    negative_prompt="{neg_escaped}",
    height={height},
    width={width},
    num_frames={num_frames},
    num_inference_steps={num_inference_steps},
    guidance_scale={guidance_scale},
    generator=generator
)

video_frames = output.frames[0]
output_path = "{output_path}"
export_to_video(video_frames, output_path, fps=16)

file_size = os.path.getsize(output_path)
elapsed = time.time() - t0
print(f"=== JOB_SUCCESS: {{output_path}} generated in {{elapsed:.1f}}s ({{file_size}} bytes) ===", flush=True)
"""


def generate_i2v_script(
    remote_image_path: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, flickering, artifacts, bad anatomy",
    width: int = 480,
    height: int = 832,
    num_frames: int = 17,
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    seed: int = 42,
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    output_filename: str = "generated_i2v.mp4",
    output_dir: str = REMOTE_JOB_DIR,
) -> str:
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        if num_frames < 17:
            num_frames = 17

    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')
    output_path = f"{output_dir.rstrip('/')}/{output_filename}"

    return f"""import sys, os, gc, time
from PIL import Image

for k in list(sys.modules.keys()):
    if "torchao" in k:
        del sys.modules[k]

import ftfy
import torch
import diffusers
import diffusers.pipelines.wan.pipeline_wan_i2v as w_i2v
w_i2v.ftfy = ftfy

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Image-to-Video Generation on A100 ===", flush=True)
print("Motion Prompt: {prompt_escaped}", flush=True)
print("Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}", flush=True)

t0 = time.time()
gc.collect()
torch.cuda.empty_cache()

# Load source image
init_image = Image.open("{remote_image_path}").convert("RGB")
init_image = init_image.resize(({width}, {height}), Image.Resampling.LANCZOS)
print(f"Loaded source image, resized to: {{init_image.size}}", flush=True)

model_id = "{model_id}"
print(f"Loading cached {{model_id}} in bfloat16 with CPU offload...", flush=True)

pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed({seed})

print("Running Wan 2.1 I2V diffusion on A100 GPU...", flush=True)
output = pipe(
    image=init_image,
    prompt="{prompt_escaped}",
    negative_prompt="{neg_escaped}",
    height={height},
    width={width},
    num_frames={num_frames},
    num_inference_steps={num_inference_steps},
    guidance_scale={guidance_scale},
    generator=generator
)

video_frames = output.frames[0]
output_path = "{output_path}"
export_to_video(video_frames, output_path, fps=16)

file_size = os.path.getsize(output_path)
elapsed = time.time() - t0
print(f"=== JOB_SUCCESS: {{output_path}} generated in {{elapsed:.1f}}s ({{file_size}} bytes) ===", flush=True)
"""

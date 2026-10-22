"""
Wan 2.1 Video Generation Payloads for Colab A100.
Optimized Diffusers pipeline for NVIDIA A100 SXM4 40GB with detached asynchronous execution.
"""

def generate_cache_cleanup_code() -> str:
    """Removes dangling blobs and unused large checkpoints to maintain healthy Colab disk space."""
    return """
import os, shutil, subprocess

hf_hub = "/root/.cache/huggingface/hub"
print("=== Colab Disk Preflight Check ===")
subprocess.run(["df", "-h", "/content"], check=False)

if os.path.exists(hf_hub):
    # Free old T2V cache if both T2V and I2V exist to stay under 80% disk threshold
    t2v_14b = os.path.join(hf_hub, "models--Wan-AI--Wan2.1-T2V-14B-Diffusers")
    if os.path.exists(t2v_14b):
        print("Clearing older T2V 14B checkpoint to keep Colab disk healthy...")
        shutil.rmtree(t2v_14b, ignore_errors=True)

print("=== Colab Disk After Cleanup ===")
subprocess.run(["df", "-h", "/content"], check=False)
"""

def generate_colab_setup_code() -> str:
    return """
import sys, subprocess, os

print("=== Checking Colab GPU Environment ===")
subprocess.run(["nvidia-smi"], check=True)

# Remove incompatible torchao if present
subprocess.run(["pip", "uninstall", "-y", "torchao"], check=False)

packages = [
    "git+https://github.com/huggingface/diffusers.git",
    "transformers>=4.48.0",
    "accelerate>=0.34.0",
    "sentencepiece",
    "imageio[ffmpeg]",
    "torchvision",
    "ftfy"
]

print("Installing / updating required packages for Wan 2.1 video generation...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
print("=== Colab Environment Ready for Video Generation ===")
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
    output_filename: str = "generated_t2v.mp4"
) -> str:
    # Ensure num_frames is 4k + 1
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        if num_frames < 17:
            num_frames = 17

    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')
    use_offload = "14B" in model_id

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

pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
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
output_path = "/content/{output_filename}"
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
    output_filename: str = "generated_i2v.mp4"
) -> str:
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        if num_frames < 17:
            num_frames = 17

    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')

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

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
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
output_path = "/content/{output_filename}"
export_to_video(video_frames, output_path, fps=16)

file_size = os.path.getsize(output_path)
elapsed = time.time() - t0
print(f"=== JOB_SUCCESS: {{output_path}} generated in {{elapsed:.1f}}s ({{file_size}} bytes) ===", flush=True)
"""

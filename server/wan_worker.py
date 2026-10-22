"""
Wan 2.1 Video Generation Payloads for Colab A100.
Self-healing, highly-optimized Diffusers pipeline for NVIDIA A100 SXM4.
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
    "torchvision"
]

print("Installing / updating required packages for Wan 2.1 video generation...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
print("=== Colab Environment Ready for Video Generation ===")
"""

def generate_t2v_execution_code(
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

    return f"""
import sys, os, gc, base64

# Clean any cached torchao imports
for k in list(sys.modules.keys()):
    if "torchao" in k or "diffusers" in k:
        del sys.modules[k]

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Text-to-Video Generation on A100 ===")
print("Prompt: {prompt_escaped}")
print("Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")

gc.collect()
torch.cuda.empty_cache()

model_id = "{model_id}"
print(f"Loading {{model_id}} in bfloat16 onto NVIDIA A100...")

pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed({seed})

print("Generating video frames on A100 GPU...")
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
print(f"Saved video to {{output_path}} with {{len(video_frames)}} frames.")

file_size = os.path.getsize(output_path)
print(f"FILE_SIZE: {{file_size}} bytes")
with open(output_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

print("__VSP_VIDEO_OUTPUT_START__")
print(b64_data)
print("__VSP_VIDEO_OUTPUT_END__")
print("=== Generation Complete! ===")
"""

def generate_i2v_execution_code(
    image_b64: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, flickering, artifacts",
    width: int = 832,
    height: int = 480,
    num_frames: int = 49,
    num_inference_steps: int = 25,
    guidance_scale: float = 6.0,
    seed: int = 42,
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    output_filename: str = "generated_i2v.mp4"
) -> str:
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        if num_frames < 17:
            num_frames = 17

    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')

    return f"""
import sys, os, gc, base64, io
from PIL import Image

for k in list(sys.modules.keys()):
    if "torchao" in k or "diffusers" in k:
        del sys.modules[k]

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Image-to-Video Generation on A100 ===")
print("Motion Prompt: {prompt_escaped}")

gc.collect()
torch.cuda.empty_cache()

image_data = base64.b64decode("{image_b64}")
init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
init_image = init_image.resize(({width}, {height}))
print(f"Loaded reference image with size {{init_image.size}}")

model_id = "{model_id}"
print(f"Loading I2V pipeline: {{model_id}} in bfloat16...")

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed({seed})

print("Generating I2V video frames on A100...")
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
print(f"FILE_SIZE: {{file_size}} bytes")
with open(output_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

print("__VSP_VIDEO_OUTPUT_START__")
print(b64_data)
print("__VSP_VIDEO_OUTPUT_END__")
print("=== I2V Generation Complete! ===")
"""

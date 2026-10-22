"""
Wan 2.1 Video Generation Payloads for Colab A100.
Generates self-contained, robust Python execution scripts dispatched to Colab via vspagent.
"""

def generate_colab_setup_code() -> str:
    return """
import sys, subprocess, os

print("=== Checking Colab GPU Environment ===")
subprocess.run(["nvidia-smi"], check=True)

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
    num_frames: int = 81,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 42,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    output_filename: str = "generated_t2v.mp4"
) -> str:
    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')
    
    return f"""
import torch
import gc
import os
import base64
from diffusers import WanPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Text-to-Video Generation on A100 ===")
print("Prompt: {prompt_escaped}")
print("Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")

# Free any stray memory
gc.collect()
torch.cuda.empty_cache()

model_id = "{model_id}"
print(f"Loading model: {{model_id}} in bfloat16...")

pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Enable memory optimizations if needed
try:
    pipe.enable_model_cpu_offload()
    print("Enabled model CPU offload for optimal VRAM usage.")
except Exception as e:
    print(f"Offload notice: {{e}}")

generator = torch.Generator(device="cuda").manual_seed({seed})

print("Generating video frames (this takes ~1-3 minutes on A100)...")
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

# Output base64 block for vspagent transfer
file_size = os.path.getsize(output_path)
print(f"FILE_SIZE: {{file_size}} bytes")
with open(output_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

print(f"__VSP_VIDEO_OUTPUT_START__")
print(b64_data)
print(f"__VSP_VIDEO_OUTPUT_END__")
print("=== Generation Complete! ===")
"""

def generate_i2v_execution_code(
    image_b64: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, flickering, artifacts",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 42,
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    output_filename: str = "generated_i2v.mp4"
) -> str:
    prompt_escaped = prompt.replace('"', '\\"').replace('\n', ' ')
    neg_escaped = negative_prompt.replace('"', '\\"').replace('\n', ' ')

    return f"""
import torch
import gc
import os
import base64
from PIL import Image
import io
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Image-to-Video Generation on A100 ===")
print("Motion Prompt: {prompt_escaped}")

# Free memory
gc.collect()
torch.cuda.empty_cache()

# Decode reference image
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

try:
    pipe.enable_model_cpu_offload()
except Exception as e:
    pass

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

print(f"__VSP_VIDEO_OUTPUT_START__")
print(b64_data)
print(f"__VSP_VIDEO_OUTPUT_END__")
print("=== I2V Generation Complete! ===")
"""

import json
import urllib.request
import time
import base64
from pathlib import Path

code = """
import sys, os, gc, base64

for k in list(sys.modules.keys()):
    if "torchao" in k or "diffusers" in k:
        del sys.modules[k]

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

print("=== Starting Wan 2.1 Flagship 14B Generation on A100 ===")
print("Allocating memory...")
gc.collect()
torch.cuda.empty_cache()

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
print(f"Loading {model_id} in bfloat16 with CPU offload...")

pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
# Enable CPU offload to cleanly handle the 28GB 14B transformer + 10GB T5 encoder
pipe.enable_model_cpu_offload()

print("Generating 17 frames at 25 steps with the 14B model...")
gen = torch.Generator("cuda").manual_seed(66299)
prompt = "A futuristic cyberpunk supercar driving down a rainy Tokyo neon street at night, cinematic reflections, ultra-realistic, 4k"

output = pipe(
    prompt=prompt,
    negative_prompt="low quality, blurry, distorted, flickering, artifacts, bad anatomy",
    height=480,
    width=832,
    num_frames=17,
    num_inference_steps=25,
    guidance_scale=6.0,
    generator=gen
)

video_frames = output.frames[0]
output_path = "/content/cyberpunk_supercar_14b.mp4"
export_to_video(video_frames, output_path, fps=16)

file_size = os.path.getsize(output_path)
print(f"SUCCESS: 14B Video exported to {output_path}! Size: {file_size} bytes")

with open(output_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

print("__VSP_VIDEO_OUTPUT_START__")
print(b64_data)
print("__VSP_VIDEO_OUTPUT_END__")
print("=== 14B Generation Complete ===")
"""

print("Dispatching 14B model test to Colab A100 via FastAPI /colab/execute...")
req = urllib.request.Request(
    "http://localhost:8000/colab/execute",
    data=json.dumps({"code": code, "keep_cell": True, "timeout": 1800.0}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1800) as response:
        res = json.loads(response.read().decode("utf-8"))
        print(f"14B Execution finished in {time.time()-t0:.2f}s!")
        output_text = res.get("output", "")
        print("Output snippet:\n", output_text[:600])

        if "__VSP_VIDEO_OUTPUT_START__" in output_text and "__VSP_VIDEO_OUTPUT_END__" in output_text:
            b64_part = output_text.split("__VSP_VIDEO_OUTPUT_START__")[1].split("__VSP_VIDEO_OUTPUT_END__")[0].strip()
            video_bytes = base64.b64decode(b64_part)

            target = Path("/home/vishnusureshperumbavoor/Desktop/test/vibe-video-studio/public/videos/cyberpunk_supercar_14b.mp4")
            with open(target, "wb") as f:
                f.write(video_bytes)
            print(f"SAVED LOCALLY TO: {target} ({len(video_bytes)/(1024*1024):.2f} MB)")

            # Update metadata
            meta_path = Path("/home/vishnusureshperumbavoor/Desktop/test/vibe-video-studio/public/videos/metadata.json")
            meta = []
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                except Exception:
                    pass
            meta.insert(0, {
                "filename": "cyberpunk_supercar_14b.mp4",
                "prompt": "A futuristic cyberpunk supercar driving down a rainy Tokyo neon street at night, cinematic reflections, ultra-realistic, 4k [Wan 2.1 14B PRO]",
                "duration": 2,
                "size_mb": round(len(video_bytes)/(1024*1024), 2),
                "type": "T2V-14B",
                "created_at": time.time()
            })
            with open(meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)
            print("Metadata updated successfully!")
        else:
            print("Full output:", output_text)
except Exception as e:
    print("Error:", e)

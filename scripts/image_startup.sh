#!/usr/bin/env bash
# Runs on every JarvisLabs container start.
#
# The container keeps only /home across a pause, so the venv lives there and the
# weights live on the attached filesystem at /home/jl_fs. Edit /home/app/env
# (mode 600) with:
#
#   IMAGE_MODEL_ID=OpenMOSS-Team/MOSS-Image-SFT
#   IMAGE_CUSTOM_PIPELINE=pipeline_moss_image
#   IMAGE_FP32_VAE=1
#   IMAGE_API_TOKEN=<same value as the backend's IMAGE_API_TOKEN>
#   HF_TOKEN=<optional, for gated weights>
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/jl_fs/hf}"
mkdir -p "$HF_HOME" /home/app
if [ -f /home/app/env ]; then
  set -a
  . /home/app/env
  set +a
fi

if [ ! -x /home/venv/bin/uvicorn ]; then
  python3 -m venv --system-site-packages /home/venv
  /home/venv/bin/pip install -q diffusers transformers accelerate \
    safetensors peft "imageio[ffmpeg]" sentencepiece fastapi uvicorn pillow
fi
if ! /home/venv/bin/python -c "import accelerate, imageio, sentencepiece" 2>/dev/null; then
  /home/venv/bin/pip install -q accelerate "imageio[ffmpeg]" sentencepiece
fi

if ! curl -fsS http://127.0.0.1:6006/health >/dev/null 2>&1; then
  cd /home/app
  setsid nohup /home/venv/bin/uvicorn image_server:app \
    --host 0.0.0.0 --port 6006 >> /home/app/server.log 2>&1 < /dev/null &
fi

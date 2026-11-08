#!/usr/bin/env python3
"""Run Wan2.2-Animate on a JarvisLabs ComfyUI instance.

This is the reusable, cost-aware runner for the workflow in
``scripts/wan_animate_workflow.json``. It:

* estimates the GPU cost before starting
* creates (or adopts) a JarvisLabs GPU instance
* installs the SAM2 custom node if needed
* downloads the Animate model / relight LoRA when missing
* trims and compresses the driving clip to the frame budget
* submits the ComfyUI workflow over the instance's local API
* downloads the result into ``public/videos``
* destroys the instance in ``finally`` unless asked to keep it

Typical use::

    python scripts/wan_animate_runner.py \
      --video public/uploads/wan_animate/driving.mp4 \
      --image public/uploads/wan_animate/character.jpg \
      --gpu A100 --resolution draft --frames 154 --yes

The instance must use the JarvisLabs ``comfyui`` template. A persistent
filesystem can be attached with ``--fs-id`` so the ~20 GB of models survive
instance destruction.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_FILE = ROOT / "scripts" / "wan_animate_workflow.json"
REMOTE_HELPER = ROOT / "scripts" / "wan_animate_remote.py"

# The sibling vspagent checkout supplies the Jupyter/SSH harness and the
# existing JarvisLabs token discovery logic.
VSPAGENT_DIR = ROOT.parent / "vspagent"
if VSPAGENT_DIR.exists() and str(VSPAGENT_DIR) not in sys.path:
    sys.path.insert(0, str(VSPAGENT_DIR))

try:
    import requests
except ImportError as exc:  # pragma: no cover - installation hint
    raise SystemExit("Missing dependency: requests. Run: uv pip install -r requirements-jarvis.txt") from exc

try:
    from vspagent.jarvis import JarvisLabsClient
except Exception as exc:  # pragma: no cover - installation hint
    raise SystemExit(
        "Missing vspagent. Install it or keep the sibling ../vspagent checkout. "
        f"Import error: {exc}"
    ) from exc

try:
    import jarvislabs
except ImportError as exc:  # pragma: no cover - installation hint
    raise SystemExit("Missing dependency: jarvislabs. Run: uv pip install -r requirements-jarvis.txt") from exc


REGION_URLS = {
    "india-noida-01": "https://backendn.jarvislabs.net",
    "india-chennai-01": "https://backendc.jarvislabs.net",
    "europe-01": "https://backendeu.jarvislabs.net",
}

GPU_PRICES_INR = {
    "A100-80GB": 140.94,
    "A100": 84.24,
    "A30": 38.88,
    "L4": 41.31,
}

RESOLUTIONS = {
    # Width x height must both be multiples of 16 for WanAnimateToVideo.
    "draft": (640, 368),
    "480p": (832, 480),
    "640": (640, 640),
    "720p": (1280, 720),
}

COMFY_ROOT = Path("/home/ComfyUI")
REMOTE_ROOT = Path("/home/wan_animate_job")
REFERENCE_REMOTE_NAME = "video_wan2_2_14B_animate_reference_image.png"
VIDEO_REMOTE_NAME = "video_wan2_2_14B_animate_original_video.mp4"

# (remote path, download URL, minimum plausible size)
MODEL_SPECS = [
    (
        COMFY_ROOT / "models" / "diffusion_models" / "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/"
        "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        18_000_000_000,
    ),
    (
        COMFY_ROOT / "models" / "loras" / "WanAnimate_relight_lora_fp16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/"
        "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
        1_400_000_000,
    ),
    (
        COMFY_ROOT / "models" / "vae" / "wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/"
        "split_files/vae/wan_2.1_vae.safetensors",
        250_000_000,
    ),
    (
        COMFY_ROOT / "models" / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/"
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        6_000_000_000,
    ),
    (
        COMFY_ROOT / "models" / "clip_vision" / "clip_vision_h.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/"
        "split_files/clip_vision/clip_vision_h.safetensors",
        1_000_000_000,
    ),
    (
        COMFY_ROOT / "models" / "loras" / "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/"
        "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        700_000_000,
    ),
]

SAM2_REPO = "https://github.com/kijai/ComfyUI-segment-anything-2.git"


def log(message: str) -> None:
    print(message, flush=True)


def die(message: str) -> "None":
    raise SystemExit(message)


def _env_token() -> str | None:
    for key in ("JARVIS_LABS_API_TOKEN", "JL_API_KEY", "JARVISLABS_API_KEY"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()

    candidates = [
        ROOT / ".env",
        ROOT.parent / "vspagent" / ".env",
        ROOT.parent / "vspmedagent" / ".env",
        Path.home() / ".vspagent" / ".env",
        Path.home() / ".vspagent" / "jarvislabs.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                token = data.get("token") or data.get("jwt") or data.get("api_key")
                if token:
                    return str(token).strip()
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() in ("JARVIS_LABS_API_TOKEN", "JL_API_KEY", "JARVISLABS_API_KEY"):
                    value = value.strip().strip("'\"")
                    if value:
                        return value
        except Exception:
            continue
    return None


def _raw_instances(token: str, region: str) -> list[dict]:
    base = REGION_URLS.get(region, REGION_URLS["india-noida-01"])
    response = requests.get(
        base + "/users/fetch",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("instances", []) or []


def _raw_instance(token: str, machine_id: str, region: str) -> dict | None:
    for item in _raw_instances(token, region):
        if str(item.get("machine_id")) == str(machine_id):
            return item
    return None


def _wait_for_raw_instance(token: str, machine_id: str, region: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        item = _raw_instance(token, machine_id, region)
        if item:
            return item
        time.sleep(5)
    die(f"JarvisLabs instance {machine_id} did not appear in the account listing.")


def _jupyter_session(url: str, password: str):
    base = url.rstrip("/").removesuffix("/lab")
    session = requests.Session()
    session.get(base + "/login", timeout=30)
    xsrf = session.cookies.get("_xsrf", "")
    session.post(
        base + "/login",
        data={"password": password, "_xsrf": xsrf},
        headers={"Referer": base + "/login"},
        timeout=30,
    )
    xsrf = session.cookies.get("_xsrf", "")
    return session, xsrf, base


def _contents_url(base: str, remote_path: str) -> str:
    relative = remote_path.lstrip("/")
    if relative.startswith("home/"):
        relative = relative[len("home/") :]
    return base + "/api/contents/" + urllib.parse.quote(relative)


def _jupyter_put(session, xsrf: str, base: str, local_path: Path, remote_path: str) -> None:
    data = local_path.read_bytes()
    if len(data) > 9 * 1024 * 1024:
        die(
            f"{local_path.name} is {len(data) / 1024 / 1024:.1f} MB; "
            "Jupyter's upload limit is about 10 MB. Compress or trim it first."
        )
    response = session.put(
        _contents_url(base, remote_path),
        json={
            "content": base64.b64encode(data).decode("ascii"),
            "format": "base64",
            "type": "file",
        },
        headers={"X-XSRFToken": xsrf},
        timeout=180,
    )
    if response.status_code not in (200, 201):
        die(f"Jupyter upload failed for {remote_path}: {response.status_code} {response.text[:200]}")


def _jupyter_get(session, base: str, remote_path: str, local_path: Path) -> None:
    response = session.get(_contents_url(base, remote_path) + "?content=1", timeout=180)
    if not response.ok:
        die(f"Jupyter download failed for {remote_path}: {response.status_code} {response.text[:200]}")
    payload = response.json()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(base64.b64decode(payload["content"]))


class Remote:
    def __init__(self, client: JarvisLabsClient, machine_id: str) -> None:
        self.client = client
        self.machine_id = str(machine_id)

    def exec(self, command: str, stream: bool = False) -> tuple[int, str]:
        wrapped = "bash -lc " + shlex.quote(command)
        return self.client.exec_jupyter(wrapped, instance_id=self.machine_id, stream=stream)

    def require(self, command: str, label: str, stream: bool = False) -> str:
        code, output = self.exec(command, stream=stream)
        if code != 0:
            die(f"{label} failed (exit {code}):\n{output[-3000:]}")
        return output

    def helper(self, *args: str) -> dict:
        command = "python3 /home/wan_animate_job/wan_animate_remote.py " + " ".join(
            shlex.quote(arg) for arg in args
        )
        code, output = self.exec(command)
        if code != 0:
            die(f"ComfyUI helper failed: {output[-2000:]}")
        try:
            return json.loads(output.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise SystemExit(f"Could not parse ComfyUI helper output: {output[-2000:]}") from exc


def _find_ffmpeg() -> str:
    for candidate in (shutil.which("ffmpeg"), "/home/vsp/bin/ffmpeg"):
        if candidate and Path(candidate).is_file():
            return candidate
    die("ffmpeg is required to trim/compress the driving video and create a poster.")


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        die("ffmpeg failed:\n" + (result.stderr or result.stdout)[-3000:])


def _prepare_inputs(video: Path, image: Path, frames: int) -> tuple[Path, Path]:
    if not video.is_file():
        die(f"Driving video not found: {video}")
    if not image.is_file():
        die(f"Character image not found: {image}")

    ffmpeg = _find_ffmpeg()
    (ROOT / "public" / "jobs").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wan_animate_", dir=str(ROOT / "public" / "jobs")))
    image_png = work / REFERENCE_REMOTE_NAME
    video_mp4 = work / VIDEO_REMOTE_NAME

    image_command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(image), "-frames:v", "1", str(image_png)]
    _run_ffmpeg(image_command)

    crf = 28
    while True:
        video_command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(video_mp4),
        ]
        _run_ffmpeg(video_command)
        size_mb = video_mp4.stat().st_size / 1024 / 1024
        if size_mb <= 8.5 or crf >= 34:
            break
        crf += 2

    if video_mp4.stat().st_size > 9 * 1024 * 1024:
        die(
            f"Trimmed driving clip is still {video_mp4.stat().st_size / 1024 / 1024:.1f} MB. "
            "Lower --frames or use a shorter clip."
        )
    return image_png, video_mp4


def _build_workflow(args: argparse.Namespace, width: int, height: int) -> dict:
    workflow = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))

    # The published workflow encodes the first segment twice (nodes 19/232:*).
    # The final combined SaveVideo (243) is the one we need.
    for node_id in ("19", "232:15", "232:230"):
        workflow.pop(node_id, None)

    workflow["159"]["inputs"]["value"] = width
    workflow["160"]["inputs"]["value"] = height
    workflow["212"]["inputs"]["width"] = ["159", 0]
    workflow["212"]["inputs"]["height"] = ["160", 0]
    workflow["229"]["inputs"].update(
        {
            "coordinates": json.dumps([{"x": width // 2, "y": height // 2}]),
            "points_store": json.dumps(
                {"positive": [{"x": width // 2, "y": height // 2}], "negative": []}
            ),
            "neg_coordinates": "[]",
            "width": width,
            "height": height,
        }
    )
    workflow["10"]["inputs"]["image"] = REFERENCE_REMOTE_NAME
    workflow["145"]["inputs"]["file"] = VIDEO_REMOTE_NAME
    workflow["21"]["inputs"]["text"] = args.prompt
    workflow["1"]["inputs"]["text"] = args.negative
    # Published typo: the second segment used width for both dimensions.
    workflow["242:90"]["inputs"]["width"] = ["159", 0]
    workflow["242:90"]["inputs"]["height"] = ["160", 0]
    for node_id in ("232:63", "242:91"):
        workflow[node_id]["inputs"]["steps"] = args.steps
        workflow[node_id]["inputs"]["cfg"] = 1.0
        workflow[node_id]["inputs"]["sampler_name"] = "euler"
        workflow[node_id]["inputs"]["scheduler"] = "simple"
    return workflow


def _estimate(args: argparse.Namespace) -> tuple[int, int, float]:
    width, height = RESOLUTIONS[args.resolution]
    price = GPU_PRICES_INR.get(args.gpu)
    if price is None:
        die(f"Unknown GPU price for {args.gpu}; add it to GPU_PRICES_INR.")
    # Rough model: setup + per-frame preprocessing + sampling + encode.
    megapixels = (width * height) / (1280 * 720)
    preprocess_min = args.frames * 0.04
    sampling_min = max(2.0, 6.0 * megapixels) * (args.steps / 6.0)
    total_min = 5.0 + preprocess_min + sampling_min + 2.0
    cost = price * total_min / 60.0
    return width, height, cost


def _ensure_comfy(remote: Remote) -> None:
    remote.require("test -f /home/ComfyUI/main.py", "ComfyUI template check")
    code, _ = remote.exec("test -d /home/ComfyUI/custom_nodes/ComfyUI-segment-anything-2")
    if code != 0:
        log("Installing ComfyUI-segment-anything-2...")
        remote.require(
            "cd /home/ComfyUI/custom_nodes && "
            f"git clone --depth 1 {shlex.quote(SAM2_REPO)}",
            "SAM2 node install",
            stream=True,
        )
        _restart_comfy(remote)

    state = remote.helper("object_info", "DownloadAndLoadSAM2Model", "WanAnimateToVideo")
    if not state.get("DownloadAndLoadSAM2Model") or not state.get("WanAnimateToVideo"):
        _restart_comfy(remote)
        state = remote.helper("object_info", "DownloadAndLoadSAM2Model", "WanAnimateToVideo")
    missing = [name for name, present in state.items() if not present]
    if missing:
        die(f"ComfyUI is missing required nodes: {missing}")


def _restart_comfy(remote: Remote) -> None:
    log("Restarting ComfyUI to load custom nodes...")
    remote.exec(
        "pkill -f '[C]omfyUI/main.py' || true; sleep 2; "
        "cd /home/ComfyUI; "
        "nohup python main.py --port 8188 --listen 127.0.0.1 "
        "--enable-cors-header https://cloud.jarvislabs.ai "
        "> /home/ComfyUI/comfy_runner.log 2>&1 &"
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        code, output = remote.exec(
            "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8188/object_info"
        )
        if output.strip().endswith("200"):
            log("ComfyUI is ready.")
            return
        time.sleep(5)
    code, output = remote.exec("tail -30 /home/ComfyUI/comfy_runner.log")
    die(f"ComfyUI did not become ready.\n{output[-3000:]}")


def _model_size(remote: Remote, path: Path) -> int:
    code, output = remote.exec(f"stat -c %s {shlex.quote(str(path))} 2>/dev/null || true")
    try:
        return int(output.strip() or "0")
    except ValueError:
        return 0


def _ensure_models(remote: Remote) -> None:
    remote.require(f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}", "remote job directory")
    for path, url, minimum in MODEL_SPECS:
        size = _model_size(remote, path)
        if size >= minimum:
            continue
        log(f"Downloading {path.name} ({minimum / 1_000_000_000:.1f} GB minimum)...")
        remote.require(f"mkdir -p {shlex.quote(str(path.parent))}", "model directory")
        log_name = str(REMOTE_ROOT / (path.name + ".download.log"))
        remote.exec(
            f"cd {shlex.quote(str(path.parent))}; "
            f"nohup wget -c -q -O {shlex.quote(path.name)} {shlex.quote(url)} "
            f"> {shlex.quote(log_name)} 2>&1 &"
        )
        deadline = time.time() + 1800
        last = size
        stable = 0
        while time.time() < deadline:
            time.sleep(15)
            size = _model_size(remote, path)
            if size >= minimum:
                break
            if size > last:
                stable = 0
                last = size
                log(f"  {path.name}: {size / 1_000_000_000:.2f} GB")
            else:
                stable += 1
                if stable >= 4:
                    code, output = remote.exec(f"tail -20 {shlex.quote(log_name)}")
                    die(f"Download stalled for {path.name}:\n{output[-2000:]}")
        if size < minimum:
            die(f"Timed out downloading {path.name} ({size} bytes).")


def _upload_inputs(
    remote: Remote,
    session,
    xsrf: str,
    base: str,
    image: Path,
    video: Path,
    workflow: dict,
) -> None:
    remote.require(f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}", "remote job directory")
    _jupyter_put(session, xsrf, base, REMOTE_HELPER, str(REMOTE_ROOT / REMOTE_HELPER.name))
    _jupyter_put(
        session,
        xsrf,
        base,
        image,
        "/home/ComfyUI/input/" + REFERENCE_REMOTE_NAME,
    )
    _jupyter_put(
        session,
        xsrf,
        base,
        video,
        "/home/ComfyUI/input/" + VIDEO_REMOTE_NAME,
    )

    prompt_path = REMOTE_ROOT / "workflow_prompt.json"
    fd, prompt_name = tempfile.mkstemp(prefix="wan_prompt_", suffix=".json")
    os.close(fd)
    local_prompt = Path(prompt_name)
    local_prompt.write_text(
        json.dumps(
            {
                "prompt": workflow,
                "client_id": "vsp-wan-animate-runner",
            }
        ),
        encoding="utf-8",
    )
    _jupyter_put(session, xsrf, base, local_prompt, str(prompt_path))
    local_prompt.unlink(missing_ok=True)


def _run_workflow(remote: Remote, timeout_minutes: int) -> dict:
    submitted = remote.helper("submit")
    prompt_id = submitted.get("prompt_id")
    if not prompt_id:
        die(f"ComfyUI rejected the prompt: {submitted}")
    log(f"ComfyUI prompt queued: {prompt_id}")

    deadline = time.time() + timeout_minutes * 60
    last_queue = ""
    while time.time() < deadline:
        history = remote.helper("history", prompt_id)
        if history:
            status = history.get("status") or {}
            if status.get("completed") or history.get("outputs"):
                return history
            if status.get("status_str") == "error":
                die(f"ComfyUI job failed: {json.dumps(status)[:3000]}")
        queue = remote.helper("queue")
        summary = f"running={queue.get('running')} pending={queue.get('pending')}"
        if summary != last_queue:
            log("  " + summary)
            last_queue = summary
        time.sleep(10)
    die("Timed out waiting for ComfyUI to finish.")


def _find_output(history: dict) -> tuple[str, str]:
    outputs = history.get("outputs") or {}
    preferred = outputs.get("243") or {}
    for bucket in ("videos", "gifs", "images"):
        for item in preferred.get(bucket, []) or []:
            if item.get("filename"):
                return item.get("subfolder", ""), item["filename"]
    for node in outputs.values():
        for bucket in ("videos", "gifs", "images"):
            for item in node.get(bucket, []) or []:
                if item.get("filename"):
                    return item.get("subfolder", ""), item["filename"]
    die("ComfyUI finished but did not report an output video.")


def _update_metadata(filename: str, prompt: str, duration: float, size_mb: float) -> None:
    path = ROOT / "public" / "videos" / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            data = loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            data = []
    data = [item for item in data if item.get("filename") != filename]
    data.insert(
        0,
        {
            "filename": filename,
            "prompt": prompt,
            "duration": round(duration, 2),
            "type": "WAN-ANIMATE",
            "model_size": "14B",
            "aspect_ratio": "16:9",
            "size_mb": round(size_mb, 2),
            "created_at": time.time(),
        },
    )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or "/home/vsp/bin/ffprobe"
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _poster(video: Path) -> None:
    ffmpeg = _find_ffmpeg()
    poster = video.with_name(video.stem + "_poster.jpg")
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            "1",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(poster),
        ]
    )


def _cleanup_state(machine_id: str) -> None:
    path = Path.home() / ".vspagent" / "jarvislabs_instances.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop(str(machine_id), None)
        if data.get("active_id") == str(machine_id):
            data.pop("active_id", None)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="Driving video (you moving).")
    parser.add_argument("--image", type=Path, required=True, help="Character reference image.")
    parser.add_argument("--output", type=Path, help="Local output path (default: public/videos/wan_animate_<ts>.mp4).")
    parser.add_argument("--gpu", default="A100", help="JarvisLabs GPU type (default: A100, 40 GB).")
    parser.add_argument("--region", default="india-noida-01", choices=sorted(REGION_URLS))
    parser.add_argument("--storage", type=int, default=100, help="Instance storage in GB.")
    parser.add_argument("--template", default="comfyui", help="JarvisLabs template (must include ComfyUI).")
    parser.add_argument("--fs-id", type=int, help="Optional persistent filesystem ID for model/cache reuse.")
    parser.add_argument("--resolution", default="draft", choices=sorted(RESOLUTIONS))
    parser.add_argument("--frames", type=int, default=154, help="Frames to keep (workflow supports up to 154).")
    parser.add_argument("--steps", type=int, default=6, choices=range(4, 9), metavar="4-8")
    parser.add_argument("--prompt", default="A blonde woman with glasses is moving naturally and speaking.")
    parser.add_argument(
        "--negative",
        default="blurry, low quality, distorted, flickering, artifacts, bad anatomy, "
        "extra fingers, deformed face, text, watermark",
    )
    parser.add_argument("--timeout", type=int, default=45, help="Workflow timeout in minutes.")
    parser.add_argument("--machine-id", help="Adopt an existing running instance instead of creating one.")
    parser.add_argument("--keep-instance", action="store_true", help="Do not destroy the instance.")
    parser.add_argument("--destroy-adopted", action="store_true", help="Destroy an adopted instance too.")
    parser.add_argument("--skip-setup", action="store_true", help="Skip node/model installation checks.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost and print the plan without creating anything.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Friendly alias for the 40 GB A100. JarvisLabs names it simply "A100".
    if args.gpu == "A100-40GB":
        args.gpu = "A100"

    if args.frames > 154:
        log("Workflow supports two 77-frame segments; clamping --frames to 154.")
        args.frames = 154
    if args.frames < 77:
        die("--frames must be at least 77 for this workflow (recommended: 154).")

    width, height, estimate = _estimate(args)
    log("Wan Animate plan:")
    log(f"  GPU:        {args.gpu} ({GPU_PRICES_INR.get(args.gpu, 0):.2f} INR/hour)")
    log(f"  Resolution: {width}x{height} ({args.resolution})")
    log(f"  Frames:     {args.frames}")
    log(f"  Steps:      {args.steps}")
    log(f"  Estimated:  ~{estimate:.0f} INR")
    log(f"  Output:     {args.output or ROOT / 'public' / 'videos'}")
    if args.resolution == "720p" and args.gpu == "A100":
        log("WARNING: 720p on a 40 GB A100 is tight; use --gpu A100-80GB for safety.")

    if args.dry_run:
        return 0

    token = _env_token()
    if not token:
        die("No JarvisLabs token found. Set JARVIS_LABS_API_TOKEN or add it to a .env file.")
    os.environ["JL_API_KEY"] = token

    if not args.yes:
        answer = input("Proceed and create/adopt the GPU instance? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return 1

    image_png, video_mp4 = _prepare_inputs(args.video, args.image, args.frames)
    workflow = _build_workflow(args, width, height)

    sdk = jarvislabs.Client(api_key=token)
    created = False
    machine_id: str | None = None
    raw: dict | None = None
    client: JarvisLabsClient | None = None
    session = None
    xsrf = ""
    base = ""
    output_path: Path | None = None

    try:
        if args.machine_id:
            machine_id = str(args.machine_id)
            raw = _raw_instance(token, machine_id, args.region)
            if not raw:
                die(f"Instance {machine_id} was not found in region {args.region}.")
            if str(raw.get("status", "")).lower() != "running":
                die(f"Instance {machine_id} is not running (status: {raw.get('status')}).")
            log(f"Adopting instance {machine_id} at {raw.get('public_ip')}")
        else:
            log(f"Creating {args.gpu} {args.template} instance...")
            instance = sdk.instances.create(
                gpu_type=args.gpu,
                name="vsp-wan-animate",
                template=args.template,
                storage=args.storage,
                region=args.region,
                fs_id=args.fs_id,
            )
            created = True
            machine_id = str(instance.machine_id)
            log(f"Created instance {machine_id}")
            raw = _wait_for_raw_instance(token, machine_id, args.region)

        password = raw.get("password")
        jupyter_url = raw.get("url")
        if not password or not jupyter_url:
            die("JarvisLabs did not return a Jupyter URL/password for this instance.")

        client = JarvisLabsClient(token=token)
        client.register_instance(
            host=raw.get("public_ip", ""),
            machine_id=machine_id,
            name=raw.get("instance_name", "vsp-wan-animate"),
            gpu_type=raw.get("gpu_type", args.gpu),
            jupyter_url=jupyter_url,
            password=password,
        )
        remote = Remote(client, machine_id)

        session, xsrf, base = _jupyter_session(jupyter_url, password)
        remote.require(f"mkdir -p {shlex.quote(str(REMOTE_ROOT))}", "remote job directory")
        # The helper is needed for the ComfyUI readiness checks, so upload it
        # before the setup phase (it is re-uploaded with the inputs later).
        _jupyter_put(
            session,
            xsrf,
            base,
            REMOTE_HELPER,
            str(REMOTE_ROOT / REMOTE_HELPER.name),
        )

        if not args.skip_setup:
            _ensure_comfy(remote)
            _ensure_models(remote)

        _upload_inputs(remote, session, xsrf, base, image_png, video_mp4, workflow)
        history = _run_workflow(remote, args.timeout)

        subfolder, filename = _find_output(history)
        remote_output = f"/home/ComfyUI/output/{subfolder}/{filename}" if subfolder else f"/home/ComfyUI/output/{filename}"
        output_path = args.output or (
            ROOT / "public" / "videos" / f"wan_animate_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        _jupyter_get(session, base, str(remote_output), output_path)
        _poster(output_path)
        duration = _probe_duration(output_path)
        size_mb = output_path.stat().st_size / 1024 / 1024
        _update_metadata(output_path.name, args.prompt, duration, size_mb)
        log(f"Saved: {output_path} ({size_mb:.1f} MB, {duration:.1f}s)")
        return 0

    finally:
        try:
            if client and machine_id:
                _cleanup_state(machine_id)
        finally:
            should_destroy = created and not args.keep_instance
            if args.machine_id and args.destroy_adopted:
                should_destroy = True
            if should_destroy and machine_id:
                try:
                    cost = sdk.instances.get(int(machine_id)).cost
                    log(f"Session cost before destroy: {cost:.2f} INR")
                except Exception:
                    pass
                try:
                    sdk.instances.destroy(int(machine_id))
                    log(f"Destroyed instance {machine_id}")
                except Exception as exc:
                    log(f"WARNING: could not destroy instance {machine_id}: {exc}")
            elif machine_id:
                log(f"Kept instance {machine_id}. Remember to destroy it later.")


if __name__ == "__main__":
    raise SystemExit(main())

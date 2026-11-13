"""Build the FLUX.2 klein LoRA training command and helper script."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

TRAINER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
)


@dataclass(frozen=True, slots=True)
class TrainSettings:
    dataset: str = "/home/jl_fs/character/dataset"
    output: str = "/home/jl_fs/character/lora"
    trigger: str = "gbrla"
    model: str = "black-forest-labs/FLUX.2-klein-4B"
    steps: int = 800
    rank: int = 16
    resolution: int = 1024
    batch: int = 1
    grad_accum: int = 4
    learning_rate: float = 1e-4
    seed: int = 0
    trainer: str = "train_dreambooth_lora_flux2_klein.py"


def build_command(settings: TrainSettings) -> list[str]:
    prompt = (
        f"{settings.trigger}, a photorealistic blonde Mexican woman in her mid-20s, "
        "natural makeup, Instagram lifestyle photo"
    )
    return [
        "accelerate",
        "launch",
        settings.trainer,
        f"--pretrained_model_name_or_path={settings.model}",
        f"--instance_data_dir={settings.dataset}",
        f"--output_dir={settings.output}",
        f"--instance_prompt={prompt}",
        f"--resolution={settings.resolution}",
        f"--train_batch_size={settings.batch}",
        "--guidance_scale=1",
        f"--gradient_accumulation_steps={settings.grad_accum}",
        "--gradient_checkpointing",
        "--cache_latents",
        "--use_8bit_adam",
        "--optimizer=adamW",
        f"--learning_rate={settings.learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=100",
        f"--max_train_steps={settings.steps}",
        f"--rank={settings.rank}",
        f"--lora_alpha={settings.rank}",
        "--mixed_precision=bf16",
        "--offload",
        f"--seed={settings.seed}",
        "--report_to=none",
        "--checkpointing_steps=200",
    ]


def write_script(settings: TrainSettings, path: str) -> Path:
    target = Path(path)
    target.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/app\n"
        f"if [ ! -f {shlex.quote(settings.trainer)} ]; then\n"
        f"  curl -L {TRAINER_URL} -o {shlex.quote(settings.trainer)}\n"
        "fi\n"
        f"{shlex.join(build_command(settings))} "
        f"> {shlex.quote(settings.output)}.log 2>&1\n",
        encoding="utf-8",
    )
    return target

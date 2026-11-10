"""Golden tests for the remote payload builders.

These assert the invariants that have broken real GPU runs before. They are
string-level on purpose: they run in milliseconds and need no GPU, so an agent
can verify a change to server/wan_worker.py before spending money.
"""

from __future__ import annotations

import re

import pytest

from server.wan_worker import (
    REMOTE_JOB_DIR,
    generate_cache_cleanup_code,
    generate_gpu_setup_code,
    generate_i2v_script,
    generate_t2v_script,
)


def frames_in(script: str) -> int:
    match = re.search(r"num_frames=(\d+)", script)
    assert match, "generated script has no num_frames argument"
    return int(match.group(1))


@pytest.mark.parametrize("requested", [17, 33, 49, 81, 153])
def test_t2v_keeps_valid_frame_counts(requested: int) -> None:
    """Wan requires num_frames == 4k + 1."""
    script = generate_t2v_script(prompt="x", num_frames=requested)
    assert frames_in(script) == requested


@pytest.mark.parametrize("requested", [18, 32, 48, 50, 80, 100])
def test_t2v_snaps_invalid_frame_counts_down_to_4k_plus_1(requested: int) -> None:
    script = generate_t2v_script(prompt="x", num_frames=requested)
    frames = frames_in(script)
    assert (frames - 1) % 4 == 0
    assert frames >= 17
    assert frames <= requested


def test_i2v_snaps_invalid_frame_counts() -> None:
    script = generate_i2v_script(remote_image_path="/tmp/x.png", prompt="x", num_frames=20)
    frames = frames_in(script)
    assert (frames - 1) % 4 == 0
    assert frames >= 17


def test_t2v_uses_cpu_offload_for_14b_only() -> None:
    small = generate_t2v_script(
        prompt="x", model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    )
    large = generate_t2v_script(
        prompt="x", model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )
    assert 'pipe.to("cuda")' in small
    assert "enable_model_cpu_offload()" not in small
    assert "enable_model_cpu_offload()" in large


def test_i2v_always_offloads_and_patches_ftfy() -> None:
    script = generate_i2v_script(remote_image_path="/tmp/x.png", prompt="x")
    assert "enable_model_cpu_offload()" in script
    # Wan I2V calls ftfy.fix_text(); the module must be injected explicitly.
    assert "w_i2v.ftfy = ftfy" in script


@pytest.mark.parametrize("builder", ["t2v", "i2v"])
def test_scripts_purge_stale_torchao(builder: str) -> None:
    if builder == "t2v":
        script = generate_t2v_script(prompt="x")
    else:
        script = generate_i2v_script(remote_image_path="/tmp/x.png", prompt="x")
    assert '"torchao" in k' in script


def test_scripts_write_to_the_remote_job_dir() -> None:
    script = generate_t2v_script(prompt="x", output_filename="out.mp4")
    assert f'"{REMOTE_JOB_DIR}/out.mp4"' in script


def test_prompts_are_escaped() -> None:
    script = generate_t2v_script(prompt='a "quoted" prompt')
    assert '\\"quoted\\"' in script


def test_setup_code_removes_torchao_and_installs_diffusers() -> None:
    setup = generate_gpu_setup_code()
    assert "torchao" in setup
    assert "diffusers.git" in setup
    assert "nvidia-smi" in setup


def test_cache_cleanup_does_not_use_content_paths() -> None:
    cleanup = generate_cache_cleanup_code()
    assert "/content" not in cleanup

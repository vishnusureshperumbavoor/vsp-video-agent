"""Pure tests for the MiniMax-H3 graph builder.

No GPU, no ComfyUI, no SSH: the graph is data, so it can be asserted directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from minimax_h3 import workflow  # noqa: E402


def test_length_snaps_to_the_frame_grid() -> None:
    for seconds in range(1, 16):
        frames = workflow.snap_length(seconds)
        assert (frames - 5) % 17 == 0
        assert frames >= seconds * 24


def test_known_lengths() -> None:
    assert workflow.snap_length(2) == 56
    assert workflow.snap_length(5) == 124
    assert workflow.snap_length(15) == 362


def test_size_must_be_a_multiple_of_32() -> None:
    assert workflow.validate_size(736, 416) == (736, 416)
    with pytest.raises(ValueError):
        workflow.validate_size(100, 416)
    with pytest.raises(ValueError):
        workflow.validate_size(736, 0)


def test_graph_wires_video_and_audio_from_the_same_latent() -> None:
    graph = workflow.build_graph("a saxophonist", length=124)

    assert graph["12"]["inputs"]["samples"] == ["11", 0]
    assert graph["13"]["inputs"]["samples"] == ["11", 0]
    assert graph["14"]["inputs"]["audio"] == ["13", 0]
    assert graph["14"]["inputs"]["images"] == ["12", 0]
    assert graph["15"]["inputs"]["video"] == ["14", 0]
    assert graph["6"]["inputs"]["length"] == 124
    assert graph["11"]["inputs"]["latent_image"] == ["6", 1]


def test_graph_uses_the_turbo_lora_and_minimax_clip_type() -> None:
    graph = workflow.build_graph("a saxophonist")

    assert graph["2"]["inputs"]["lora_name"] == workflow.TURBO_LORA
    assert graph["3"]["inputs"]["type"] == "minimax"
    assert graph["14"]["inputs"]["fps"] == 24


def test_graph_rejects_empty_prompts_and_bad_steps() -> None:
    with pytest.raises(ValueError):
        workflow.build_graph("   ")
    with pytest.raises(ValueError):
        workflow.build_graph("ok", steps=0)
    with pytest.raises(ValueError):
        workflow.build_graph("ok", length=2)


def test_estimate_grows_with_work() -> None:
    cheap = workflow.estimate(3, 608, 352, 4)
    pricey = workflow.estimate(10, 960, 544, 12)

    assert 0 < cheap[0] < pricey[0]
    assert 0 < cheap[1] < pricey[1]


def test_every_model_file_is_folder_scoped() -> None:
    for path, size in workflow.MODEL_FILES.items():
        assert path.split("/")[0] in {"diffusion_models", "text_encoders", "vae", "loras"}
        assert size > 0

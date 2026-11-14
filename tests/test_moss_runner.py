"""Unit tests for the MOSS-Image runner.

The GPU path cannot be exercised here, so the tests cover the two places where a
silent bug would cost a paid run: command quoting and the remote script's
argument validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import moss_image_remote as remote  # noqa: E402
import moss_image_runner as runner  # noqa: E402


def args_for(**overrides):
    base = dict(
        prompt="moss_xvstyle, a test prompt",
        negative_prompt="",
        count=2,
        size=1024,
        steps=50,
        cfg=3.0,
        seed=42,
        lora_scale=0.8,
        skip_lora=False,
        no_offload=False,
        gpu="A100",
        template="pytorch",
        disk_gb=80,
        machine_id=None,
        keep_instance=False,
        destroy_adopted=False,
        skip_setup=False,
        timeout=60,
        boot_timeout=600,
        out_dir=str(ROOT / "public" / "images"),
        dry_run=True,
        yes=False,
    )
    base.update(overrides)
    return type("Args", (), base)


def test_build_command_quotes_the_prompt() -> None:
    command = runner.build_command(args_for(prompt="a 'quoted' prompt; rm -rf /"), "/lora")

    assert "rm -rf" not in command or "'" in command
    assert command.count("--prompt") == 1
    # The prompt must survive as a single shell argument.
    assert "'a '\"'\"'quoted'\"'\"' prompt; rm -rf /'" in command


def test_build_command_includes_lora_only_when_present() -> None:
    with_lora = runner.build_command(args_for(), "/home/vsp_jobs/lora/moss")
    without = runner.build_command(args_for(), "")

    assert "--lora-dir" in with_lora
    assert "--lora-scale" in with_lora
    assert "--lora-dir" not in without


def test_the_axiomofmind_adapter_is_the_default() -> None:
    args = args_for()

    assert runner.LORA_REPO == "axiomofmind/Moss-image-nsfw-uncensored-lora"
    assert args.skip_lora is False
    assert args.lora_scale == 0.8


def test_build_command_passes_negative_prompt_when_set() -> None:
    command = runner.build_command(args_for(negative_prompt="blurry"), "")

    assert "--negative-prompt" in command
    assert "blurry" in command


def test_build_command_maps_size_to_width_and_height() -> None:
    command = runner.build_command(args_for(size=768), "")

    assert "--width 768" in command
    assert "--height 768" in command


def test_estimate_grows_with_image_count() -> None:
    one_minutes, one_cost = runner.estimate(args_for(count=1))
    four_minutes, four_cost = runner.estimate(args_for(count=4))

    assert four_minutes > one_minutes
    assert four_cost > one_cost
    assert one_cost > 0


def test_estimate_uses_the_selected_gpu_price() -> None:
    _, a100 = runner.estimate(args_for(gpu="A100"))
    _, a30 = runner.estimate(args_for(gpu="A30"))

    assert a30 < a100


def test_remote_rejects_sizes_that_are_not_multiples_of_16(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["x", "--prompt", "p", "--width", "1000"])

    with pytest.raises(SystemExit):
        remote.parse_args()


def test_remote_parses_skip_layers(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["x", "--prompt", "p", "--skip-layers", "22,18"]
    )

    parsed = remote.parse_args()

    assert parsed.skip_layers == [22, 18]


class StubClient:
    """Returns scripted log chunks and records the commands it was asked for."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.commands: list[str] = []

    def exec(self, command, **_kwargs):
        self.commands.append(command)
        return 0, self.chunks.pop(0) if self.chunks else ""


def test_follow_counts_blank_lines_in_the_tail_offset(monkeypatch) -> None:
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    client = StubClient(
        [
            "=== MOSS_LOADING: model\n\n=== IMAGE_SAVED: /tmp/a.png\n",
            "=== MOSS_SUCCESS: 1 images in 2.0s\n",
        ]
    )

    saved, ok = runner.follow(client, "mid", "/tmp/job.log", runner.time.time() + 60)

    assert ok is True
    assert saved == ["/tmp/a.png"]
    # 3 physical lines consumed, blank one included, so the next read starts at 4.
    assert "tail -n +4" in client.commands[1]


def test_follow_reports_failure_on_a_traceback(monkeypatch) -> None:
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    client = StubClient(["Traceback (most recent call last):\n"])

    saved, ok = runner.follow(client, "mid", "/tmp/job.log", runner.time.time() + 60)

    assert ok is False
    assert saved == []


def test_estimate_covers_the_lora_download_and_setup() -> None:
    minutes, _ = runner.estimate(args_for(count=1))

    # A 27 GB base download dominates; a plan under 15 minutes would be wrong.
    assert minutes >= 15

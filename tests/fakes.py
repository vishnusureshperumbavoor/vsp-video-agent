"""In-process stand-in for the JarvisLabs transport.

The real GPU costs money and needs an SSH connection, so an agent cannot verify a
change end to end. ``FakeGPU`` implements the same surface as
``server.gpu.JarvisGPU`` against a temporary directory, which makes the whole
job lifecycle testable in milliseconds for free.

Keep this class in sync with ``server/gpu.py``: if you add a transport method,
add it here too, otherwise the tests silently stop covering it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

SUCCESS_MARKER = "=== JOB_SUCCESS: {path} generated in 1.0s ({size} bytes) ==="


class FakeGPU:
    """Deterministic, offline JarvisGPU."""

    def __init__(
        self,
        root: Path,
        gpu_name: str = "NVIDIA A100-SXM4-40GB",
        connected: bool = True,
        log: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._gpu_name = gpu_name
        self._connected = connected
        self._log = log
        self.remote_files: dict[str, bytes] = {}
        self.logs: dict[str, str] = {}
        self.commands: list[str] = []
        self.uploaded: list[tuple[str, str]] = []
        self.downloaded: list[tuple[str, str]] = []
        self.spawned: list[str] = []

    # ------------------------------------------------------------------ state
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def gpu_info(self) -> str | None:
        return self._gpu_name

    @property
    def instance_id(self) -> str | None:
        return "fake-000000" if self._connected else None

    async def probe(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "gpu": self._gpu_name if self._connected else None,
            "instance_id": self.instance_id,
        }

    # ---------------------------------------------------------------- remote IO
    async def execute_code(self, code: str, timeout: float = 600.0) -> dict[str, Any]:
        self.commands.append(code)
        return {"output": "", "isError": False, "exit_code": 0}

    async def write_file(self, remote_path: str, content: str) -> int:
        payload = content.encode("utf-8")
        self.remote_files[remote_path] = payload
        return len(payload)

    async def spawn(
        self, command: str, log_path: str, workdir: str = "/home/vsp_jobs"
    ) -> int:
        self.spawned.append(command)
        if self._log is not None:
            self.logs[log_path] = self._log
        else:
            output = next(
                (p for p in self.remote_files if p.endswith(".mp4")), "generated.mp4"
            )
            size = len(self.remote_files.get(output, b"fake-mp4-bytes"))
            self.logs[log_path] = SUCCESS_MARKER.format(path=output, size=size)
        return 4242

    async def read_log(self, log_path: str, tail_bytes: int = 4000) -> str:
        return self.logs.get(log_path, "")[-tail_bytes:]

    async def exists(self, remote_path: str) -> bool:
        return remote_path in self.remote_files

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        self.uploaded.append((local_path, remote_path))
        self.remote_files[remote_path] = Path(local_path).read_bytes()
        return True

    async def download_file(self, remote_path: str, local_path: str) -> bool:
        self.downloaded.append((remote_path, local_path))
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.remote_files.get(remote_path)
        if payload is None:
            payload = b"\x00\x00\x00\x18ftypmp42" + b"fake video bytes"
        target.write_bytes(payload)
        return True

    # ------------------------------------------------------------------ helpers
    def stage_output(self, remote_path: str, local_file: Path) -> None:
        """Pretend the render produced ``local_file`` at ``remote_path``."""
        self.remote_files[remote_path] = Path(local_file).read_bytes()

    def reset(self) -> None:
        self.remote_files.clear()
        self.logs.clear()
        self.commands.clear()
        self.uploaded.clear()
        self.downloaded.clear()
        self.spawned.clear()
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


class FakeImageAPI:
    """Deterministic stand-in for ``server.transport.image_api.ImageAPIClient``.

    The real client calls a paid GPU. Keep the signature in sync with it so the
    service tests keep covering prompt, size, steps, seed and negative prompt.
    """

    def __init__(self, payload: bytes = b"\x89PNG\r\n\x1a\nfake", error=None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        size: int,
        steps: int,
        seed: int,
        negative_prompt: str = "",
    ) -> bytes:
        self.calls.append(
            {
                "prompt": prompt,
                "size": size,
                "steps": steps,
                "seed": seed,
                "negative_prompt": negative_prompt,
            }
        )
        if self.error is not None:
            raise self.error
        return self.payload

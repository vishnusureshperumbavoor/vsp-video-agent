"""JarvisLabs GPU transport for the video pipeline.

This replaces the former Google Colab MCP client. Everything runs over SSH on a
JarvisLabs instance, so there is no browser bridge, no notebook kernel and no
`colab-mcp` pairing step.

The public surface intentionally mirrors what the old Colab client exposed
(``is_connected``, ``gpu_info``, ``execute_code``, ``upload_file_to_colab``,
``download_file_from_colab``) so the job runner could be ported with minimal
churn. The names are kept for compatibility but the semantics are JarvisLabs.
"""

from __future__ import annotations

import asyncio
import base64
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

from server.jarvis import JarvisLabsClient

# Where job scripts, logs and outputs live on the GPU box.
REMOTE_ROOT = "/home/vsp_jobs"
REMOTE_UPLOADS = f"{REMOTE_ROOT}/uploads"


class GPUError(RuntimeError):
    """Raised when the GPU box is unreachable or a remote command fails."""


class JarvisGPU:
    """Thin async wrapper around :class:`JarvisLabsClient`."""

    def __init__(self) -> None:
        self.client = JarvisLabsClient()
        self._gpu_info: Optional[str] = None
        self._instance_id: Optional[str] = None

    # ------------------------------------------------------------------ state
    def _active(self) -> Optional[Dict[str, Any]]:
        inst = self.client.get_active_instance()
        if inst and str(inst.get("status", "")).lower() == "running":
            self._instance_id = str(inst.get("machine_id"))
            return inst
        return None

    @property
    def is_connected(self) -> bool:
        return self._active() is not None

    @property
    def instance_id(self) -> Optional[str]:
        self._active()
        return self._instance_id

    @property
    def gpu_info(self) -> Optional[str]:
        """Last probed GPU name, or None before :meth:`probe` has run."""
        return self._gpu_info

    # ---------------------------------------------------------------- helpers
    def _run(self, command: str, timeout: int = 600) -> tuple[int, str]:
        if not self._active():
            raise GPUError(
                "No running JarvisLabs instance. Create or adopt one first "
                "(`python cli.py status`, then see README for provisioning)."
            )
        try:
            return self.client.exec(command, instance_id=self._instance_id, stream=False)
        except Exception as exc:  # network / ssh failure
            raise GPUError(f"Remote command failed: {exc}") from exc

    async def _arun(self, command: str, timeout: int = 600) -> tuple[int, str]:
        return await asyncio.to_thread(self._run, command, timeout)

    # ----------------------------------------------------------------- public
    async def probe(self) -> Dict[str, Any]:
        """Check SSH reachability and read the GPU identity."""
        code, out = await self._arun(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1"
        )
        name = out.strip().splitlines()[0] if out.strip() else ""
        self._gpu_info = name or None
        return {"connected": code == 0 and bool(name), "gpu": name, "instance_id": self._instance_id}

    async def execute_code(self, code: str, timeout: float = 600.0) -> Dict[str, Any]:
        """Run Python ``code`` on the GPU box and return its output.

        The code is base64-encoded so quoting, newlines and shell metacharacters
        cannot break the SSH command.
        """
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        command = f"echo {payload} | base64 -d | python3 -"
        rc, out = await self._arun(command, timeout=int(timeout))
        return {"output": out, "isError": rc != 0, "exit_code": rc}

    async def write_file(self, remote_path: str, content: str) -> int:
        """Write text content to the remote host, returning the byte count."""
        payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parent = str(Path(remote_path).parent)
        command = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"echo {payload} | base64 -d > {shlex.quote(remote_path)} && "
            f"stat -Lc %s {shlex.quote(remote_path)}"
        )
        rc, out = await self._arun(command)
        if rc != 0:
            raise GPUError(f"Failed writing {remote_path}: {out[-400:]}")
        try:
            return int(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0

    async def spawn(self, command: str, log_path: str, workdir: str = REMOTE_ROOT) -> int:
        """Start ``command`` detached and return its PID."""
        await self._arun(f"mkdir -p {shlex.quote(workdir)}")
        # `cd dir; cmd &` — not `cd dir && cmd &`, which makes bash background the
        # whole AND-list and block until it finishes.
        rc, out = await self._arun(
            f"cd {shlex.quote(workdir)}; "
            f"setsid nohup {command} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
            f"sleep 0.3; pgrep -f {shlex.quote(command)} | head -1"
        )
        match = re.search(r"\b(\d+)\b", out)
        return int(match.group(1)) if match else 0

    async def read_log(self, log_path: str, tail_bytes: int = 4000) -> str:
        rc, out = await self._arun(
            f"tail -c {int(tail_bytes)} {shlex.quote(log_path)} 2>/dev/null || true"
        )
        return out

    async def exists(self, remote_path: str) -> bool:
        rc, out = await self._arun(f"test -e {shlex.quote(remote_path)} && echo yes || echo no")
        return "yes" in out

    async def upload_file_to_colab(self, local_path: str, remote_path: str) -> bool:
        """Upload a local file to the GPU box (name kept for call-site compatibility)."""
        parent = str(Path(remote_path).parent)
        await self._arun(f"mkdir -p {shlex.quote(parent)}")
        return await asyncio.to_thread(
            self.client.sync_up, local_path, remote_path=remote_path, instance_id=self._instance_id
        )

    async def download_file_from_colab(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the GPU box (name kept for call-site compatibility)."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(
            self.client.sync_down, remote_path, local_path=local_path, instance_id=self._instance_id
        )


# Shared instance, mirroring the old `internal_colab_client` import style.
internal_gpu_client = JarvisGPU()

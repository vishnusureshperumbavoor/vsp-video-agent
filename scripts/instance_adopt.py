"""Adopt an existing JarvisLabs instance into the local registry.

Both the MOSS runner and the MiniMax-H3 runner need the same registration call
before they can execute over SSH. Keeping it in one place stops the two runners
from drifting apart.
"""

from __future__ import annotations


def adopt_instance(client, machine_id: str, default_name: str = "gpu") -> dict | None:
    """Register a running instance by id and return its details, or None."""
    for inst in client.list_instances():
        if str(inst.get("machine_id")) != str(machine_id):
            continue
        client.register_instance(
            host=inst["host"],
            machine_id=str(machine_id),
            name=inst.get("name") or default_name,
            user=inst.get("user", "root"),
            port=int(inst.get("port", 22)),
            gpu_type=inst.get("gpu_type", "Unknown"),
            status=inst.get("status", "Running"),
            jupyter_url=inst.get("jupyter_url"),
            password=inst.get("password", ""),
        )
        return inst
    return None

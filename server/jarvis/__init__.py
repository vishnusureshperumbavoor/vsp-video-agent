"""Vendored JarvisLabs harness.

This is the `jarvis` module of `vspagent`, copied into the project so the
application no longer depends on a sibling editable checkout. The Colab,
trainer and dataset modules were deliberately left out: this project only uses
JarvisLabs.

Source: https://github.com/vishnusureshperumbavoor/vspagent (v3.3.0).
Keep `CLIENT_API_VERSION` in sync if you re-vendor.
"""

from server.jarvis.client import CLIENT_API_VERSION, JarvisLabsClient
from server.jarvis.tunnel import (
    list_active_tunnels,
    start_ssh_tunnel,
    stop_ssh_tunnel,
)

__all__ = [
    "CLIENT_API_VERSION",
    "JarvisLabsClient",
    "list_active_tunnels",
    "start_ssh_tunnel",
    "stop_ssh_tunnel",
]

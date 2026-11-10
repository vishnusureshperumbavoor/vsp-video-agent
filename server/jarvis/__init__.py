"""Vendored JarvisLabs harness.

Copied into the project so the application is self-contained and does not depend
on a sibling editable checkout. Only the JarvisLabs transport lives here.

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

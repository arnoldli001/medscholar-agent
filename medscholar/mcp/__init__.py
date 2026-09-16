"""MCP（Model Context Protocol）集成层。"""

from __future__ import annotations

from .server import MCP_AVAILABLE, build_server, main

__all__ = ["MCP_AVAILABLE", "build_server", "main"]

"""HTTP 服务层。"""

from __future__ import annotations

from .app import WEB_DIR, app, create_app

__all__ = ["app", "create_app", "WEB_DIR"]

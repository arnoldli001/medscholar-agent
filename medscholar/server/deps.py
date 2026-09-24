"""路由层共享依赖：全局单例取用口（get_db/get_runtime/get_registry/get_config）与复用纯函数。

依赖方向单向 app → routes/* → deps → 业务模块；本模块不得 import 任何 routes.*，否则循环导入。
"""

from __future__ import annotations

from pathlib import Path

from ..agent.runtime import get_runtime
from ..api import get_registry
from ..config import get_config
from ..db.connect import get_db

#: 前端静态资源目录 medscholar/web/
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

__all__ = [
    "WEB_DIR",
    "get_config",
    "get_db",
    "get_registry",
    "get_runtime",
    "is_resumable",
    "render_index",
    "sse_headers",
]


def is_resumable(status: str, phase: str, done_phases: list[str] | None = None) -> bool:
    """是否可续跑取决于阶段快照 ``done_phases``，而非 agent_runs.phase。

    中断时 phase 常停在 await_approval（不属于流水线阶段），按它判断会误判。
    """
    if status in {"done", "cancelled"}:
        return False
    if done_phases is None:
        # 未知快照情况时退化为按阶段名判断（旧行为）
        return (phase or "") in {"plan", "execute", "reflect", "synthesize", "review"}
    return bool(done_phases)


def render_index(index_file: Path) -> str:
    """读 index.html 并给静态资源加 mtime 版本号。

    no-store 防不住浏览器已缓存的旧副本，只有换新 URL 才能强制更新。
    """
    html = index_file.read_text(encoding="utf-8")
    for asset in ("app.js", "style.css"):
        path = WEB_DIR / asset
        try:
            token = str(int(path.stat().st_mtime))
        except OSError:  # pragma: no cover - 文件不存在时保持原样
            continue
        html = html.replace(f"./static/{asset}", f"./static/{asset}?v={token}")
    return html


def sse_headers() -> dict[str, str]:
    """SSE 响应头（stream 与 /api/ask 共用）；每次返回新字典，避免共享可变对象被 Starlette 改写串味。"""
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

"""路由层共享的依赖与纯工具。

``app.py`` 原来是一个 1384 行的"上帝模块"：单个 ``_register_routes`` 注册了
54 个接口，还混着 18 个 Pydantic 请求模型与一堆局部闭包。按业务标签拆成
``routes/`` 下的六个 APIRouter 之后，跨 router 共享的东西必须有一个唯一归属，
否则要么 ``routes/a.py`` 水平 import ``routes/b.py``，要么每个 router 各自
import 一遍、各自建一份——数据库与 Agent 运行时被重复初始化，本地单用户工具
不能发生这种事。

本模块承担两个职责：

1. 全局单例的唯一取用口：``get_db`` / ``get_runtime`` / ``get_registry`` /
   ``get_config``。它们仍是各自模块里原有的单例（这里只是重导出，不新建实例、
   不加缓存），"谁先调用谁初始化、之后处处同一份"的语义与拆分前一致。
2. 被多处复用的纯函数与常量：首页渲染（静态资源版本号注入）、运行可续跑判定、
   SSE 响应头。

依赖方向单向：``app.py`` → ``routes/*`` → ``deps.py`` → 业务模块。
本模块不得 import 任何 ``routes.*``，否则立刻循环导入。
"""

from __future__ import annotations

from pathlib import Path

from ..agent.runtime import get_runtime
from ..api import get_registry
from ..config import get_config
from ..db.connect import get_db

#: 前端静态资源目录（``medscholar/web/``）。从前在 app.py，路径表达式一字未改。
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
    """这次运行还能不能"接着跑"。

    必须先有阶段快照才谈得上续跑；判断依据是快照（done_phases），
    而不是 agent_runs.phase —— 被中断的运行那一列常常是 await_approval，
    它不属于流水线阶段，用它判断会把可续跑的运行误判成不可续跑。
    """
    if status in {"done", "cancelled"}:
        return False
    if done_phases is None:
        # 未知快照情况时退化为按阶段名判断（旧行为）
        return (phase or "") in {"plan", "execute", "reflect", "synthesize", "review"}
    return bool(done_phases)


def render_index(index_file: Path) -> str:
    """读取 index.html 并给静态资源加上基于文件修改时间的版本号。

    纯本地工具，静态资源缓存没有任何收益，却会导致"服务端修好了、用户浏览器
    还在跑旧 JS"——实测踩到过，表现为反复收到已经修复过的报错。
    ``Cache-Control: no-store`` 只能防止再次缓存，对"浏览器已经缓存了旧副本"
    无能为力；换成带版本号的新 URL 才能强制更新。
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
    """SSE 响应头（``/api/agent/stream/*`` 与 ``/api/ask`` 共用）。

    返回新字典而不是共享常量：Starlette 会在这份 header 上做处理，
    共享同一个可变对象迟早会串味。
    """
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

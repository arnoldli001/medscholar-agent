"""FastAPI 应用：本地 Web 工作台的后端 —— **组合根**。

本模块只负责"装配"，不再负责"实现"：

* 54 个接口按业务标签拆到 ``medscholar/server/routes/`` 的六个 APIRouter 里
  （见 ``routes/__init__.py`` 的对照表）；
* 跨 router 共享的单例与纯函数集中在 ``medscholar/server/deps.py``；
* 本模块保留 ``lifespan`` 生命周期、静态资源挂载、请求模型的重导出，
  以及模块级 ``app`` 对象 —— 这三样是 CLI / MCP / 冒烟脚本依赖的公开入口。

设计原则（拆分前后一致）：

* 所有数据库重操作走 ``asyncio.to_thread``，不阻塞事件循环（SSE 需要它保持流畅）；
* 单源检索失败、嵌入失败、LLM 失败都只降级、不抛 500；
* 绝不返回 API Key 明文。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agent.runtime import get_runtime
from ..api import get_registry
from ..config import get_config
from ..db import repo
from .deps import WEB_DIR, get_db
from .deps import is_resumable as _is_resumable  # 兼容旧入口（原 app._is_resumable）
from .deps import render_index as _render_index  # 兼容旧入口（原 app._render_index）
from .routes import agent, library, papers, search, system, writing
from .routes.agent import AgentRunRequest, ApprovalRequest
from .routes.library import (
    EmbedRequest,
    FulltextBackfillRequest,
    ProjectCreateRequest,
    ProjectPapersRequest,
    SessionCreateRequest,
)
from .routes.papers import DeletePapersRequest, ImportRequest, LocalSearchRequest
from .routes.search import AskRequest, LiveSearchRequest, QueryPreviewRequest
from .routes.writing import CiteRequest, ExportRequest, FeedbackRequest, ManuscriptRequest

logger = logging.getLogger(__name__)

#: 公开入口。列在这里的名字都允许外部 ``from medscholar.server.app import ...``：
#: 其中请求模型历史上就定义在本模块，拆分后随各自的 router 走，这里只是重导出，
#: 以保证 ``scripts/smoke_http.py``、CLI、MCP 等既有引用不被破坏。
__all__ = [
    "WEB_DIR",
    "_is_resumable",
    "_mount_static",
    "_render_index",
    "app",
    "create_app",
    "lifespan",
    # ---- 请求模型（重导出）----
    "AgentRunRequest",
    "ApprovalRequest",
    "AskRequest",
    "CiteRequest",
    "DeletePapersRequest",
    "EmbedRequest",
    "ExportRequest",
    "FeedbackRequest",
    "FulltextBackfillRequest",
    "ImportRequest",
    "LiveSearchRequest",
    "LocalSearchRequest",
    "ManuscriptRequest",
    "ProjectCreateRequest",
    "ProjectPapersRequest",
    "QueryPreviewRequest",
    "SessionCreateRequest",
]


# ============================================================ 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：单例的**唯一**初始化点与销毁点。

    数据库、Agent 运行时、数据源注册表都是进程级单例（见 ``deps.py``），
    这里启动时初始化一次、退出时统一收尾；router 只通过 ``deps`` 取用现成实例，
    绝不自己再建一份。
    """
    cfg = get_config()
    cfg.ensure_dirs()
    db = get_db()
    # 把上次退出时仍在进行中的运行标记为「已中断」。
    # 否则用户会对着一个永远等不到草稿的页面发呆 —— 实测踩到过：
    # 用户 17:00 发起研究，服务 17:10 重启，运行被杀死，界面既不报错也无内容。
    try:
        interrupted = await asyncio.to_thread(repo.mark_interrupted_runs, db=db)
        if interrupted:
            logger.warning("已将 %d 条上次未完成的运行标记为中断", interrupted)
    except Exception as exc:  # pragma: no cover
        logger.debug("标记中断运行失败：%s", exc)
    logger.info(
        "MedScholar 启动：数据库 %s（向量后端 %s）", db.path, "sqlite-vec" if db.vec_available else "python"
    )
    if not db.vec_available:
        logger.warning("sqlite-vec 不可用，已启用纯 Python 向量检索回退：%s", db.vec_note)
    try:
        yield
    finally:
        runtime = get_runtime()
        await runtime.shutdown()
        try:
            await get_registry().close()
        except Exception:  # pragma: no cover
            pass
        from ..db.connect import close_db

        close_db()
        logger.info("MedScholar 已停止")


# ============================================================ 应用装配
def create_app() -> FastAPI:
    """组合根：把 Cors、静态资源、六个业务 router 装进一个 FastAPI 应用。

    顺序有意如此：先中间件、再静态资源、最后路由。CORS 与"静态资源禁用缓存"
    两个中间件的添加次序决定包裹层次（后加的在外层），拆分前也是这个次序。
    """
    cfg = get_config()
    app = FastAPI(
        title="MedScholar Agent",
        version=__version__,
        description="面向医学研究者的本地化学术智能体",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _mount_static(app)

    # 按业务分组挂载：基础 / 文献库 / 检索问答 / Agent 与产物 / 会话课题维护 / 引用写作
    app.include_router(system.router)
    app.include_router(papers.router)
    app.include_router(search.router)
    app.include_router(agent.router)
    app.include_router(library.router)
    app.include_router(writing.router)
    return app


def _mount_static(app: FastAPI) -> None:
    """挂载 ``/static`` 并禁止静态资源被浏览器缓存。

    首页 ``/`` 与 ``/favicon.ico`` 也属于"页面"而非"接口"，但它们的实现放在
    ``routes/system.py``（那里能看到首页渲染的完整逻辑），这里只负责静态目录与
    响应头。
    """
    if WEB_DIR.is_dir():
        # **必须禁用浏览器缓存。**
        #
        # 这是一个纯本地单用户工具，静态资源没有任何缓存收益，却有一个致命代价：
        # 修好前端的 bug 后，用户的浏览器仍在跑旧版 app.js，于是"修复没生效"。
        # 实测就踩到了 —— 服务端日志里完全看不到用户浏览器的请求，
        # 因为页面里的旧 JS 早就放弃了重试。
        # 用 no-store 保证每次刷新都拿到当前磁盘上的文件。
        app.mount(
            "/static",
            StaticFiles(directory=str(WEB_DIR)),
            name="static",
        )

    @app.middleware("http")
    async def _no_store_for_assets(request: Request, call_next):  # noqa: ANN001
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app = create_app()

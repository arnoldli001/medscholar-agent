"""FastAPI 组合根：接口拆到 routes/ 六个 APIRouter，共享依赖在 deps.py，本模块只管装配。

约束：数据库重操作走 asyncio.to_thread 不阻塞事件循环；单源检索/嵌入/LLM 失败只降级不抛 500；绝不返回 API Key 明文。
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

#: 请求模型随各自 router 定义，此处重导出以兼容 scripts/CLI/MCP 的既有引用。
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
    """应用生命周期：进程级单例（DB/Agent 运行时/数据源注册表）的唯一初始化与销毁点。"""
    cfg = get_config()
    cfg.ensure_dirs()
    db = get_db()
    # 把上次退出时仍在进行的运行标记为「已中断」，否则前端会无限等待草稿
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
    """装配 CORS、静态资源与六个业务 router；中间件后加的在外层，添加次序不可随意调换。"""
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

    app.include_router(system.router)
    app.include_router(papers.router)
    app.include_router(search.router)
    app.include_router(agent.router)
    app.include_router(library.router)
    app.include_router(writing.router)
    return app


def _mount_static(app: FastAPI) -> None:
    """挂载 ``/static`` 并按 mtime 版本号正确缓存；首页与 favicon 路由在 ``routes/system.py``。

    P1-9 修复：原实现 ``server/deps.py`` 把 ``app.js``/``style.css`` 注成 ``?v=<mtime>``
    但中间件又对 ``/static/*`` 强加 ``no-store, must-revalidate``，两者矛盾——
    浏览器本来能命中缓存的强缓存被 no-store 强制每次都向服务器校验，
    198KB JS + 62KB CSS 每次刷新全量重下。

    正确做法：
    - 带 ``?v=`` 的静态资源 → ``immutable, max-age=31536000``（一年强缓存）。
      URL 含 mtime，文件修改时 URL 自动变，浏览器从旧 URL 转向新 URL = 缓存自然失效。
    - 根路径 ``/`` → ``no-store``（render_index 每次渲染可能不同，确保拿最新 HTML）。
    - 不带 ``?v=`` 的 /static 子资源（如 favicon） → 短 max-age 默认。
    """
    if WEB_DIR.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(WEB_DIR)),
            name="static",
        )

    @app.middleware("http")
    async def _cache_strategy_for_assets(request: Request, call_next):  # noqa: ANN001
        response = await call_next(request)
        path = request.url.path
        query = request.url.query
        if path == "/" or path == "/index.html":
            # HTML 入口必须每次拿最新（render_index 注入了 ?v=）
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif path.startswith("/static/") and "v=" in query:
            # 带版本号的静态资源：URL 含 mtime，强缓存一年
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/static/"):
            # 不带版本号的（如 favicon）：短缓存以减少重复请求，但不强制刷新
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response


app = create_app()

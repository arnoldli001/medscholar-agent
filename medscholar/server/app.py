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
    """挂载 ``/static`` 并禁止浏览器缓存；首页与 favicon 路由在 ``routes/system.py``。"""
    if WEB_DIR.is_dir():
        # 必须 no-store：本地工具静态缓存无收益，浏览器跑旧版 app.js 会表现为"修复不生效"
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

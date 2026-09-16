"""FastAPI 应用：本地 Web 工作台的后端。

本模块是**唯一的 HTTP 入口**，严格实现 ``docs/API.md`` 中的契约。
设计原则：

* 所有数据库重操作走 ``asyncio.to_thread``，不阻塞事件循环（SSE 需要它保持流畅）；
* 单源检索失败、嵌入失败、LLM 失败都只降级、不抛 500；
* 绝不返回 API Key 明文。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.runtime import get_runtime
from ..agent.state import Phase
from ..api import ALL_SOURCES, SearchFilters, get_registry
from ..cite import STYLE_LABELS, STYLES, detect_style
from ..config import get_config
from ..db import repo
from ..db.connect import get_db
from ..embedding.pipeline import embedding_status, run_embedding_pipeline_async
from ..export.exporters import EXPORT_FORMATS, export_csv, export_json
from ..llm.client import get_llm

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _is_resumable(status: str, phase: str, done_phases: list[str] | None = None) -> bool:
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


def _render_index(index_file: Path) -> str:
    """读取 index.html 并给静态资源加上**基于文件修改时间的版本号**。

    为什么要这么做：这是一个纯本地工具，静态资源缓存没有任何收益，
    却会导致"服务端修好了、用户浏览器还在跑旧 JS"——实测踩到过，
    表现为反复收到已经修复过的报错。``Cache-Control: no-store`` 只能防止**再次**
    缓存，对"浏览器已经缓存了旧副本"无能为力；换成带版本号的新 URL 才能强制更新。
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


# ============================================================ 请求体模型
class LiveSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    sources: list[str] | None = None
    limit: int = Field(default=40, ge=1, le=500)
    per_source_limit: int = Field(default=20, ge=1, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)
    save: bool = True
    embed: bool = True


class LocalSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryPreviewRequest(BaseModel):
    """检索式预览请求。"""

    query: str = ""
    sources: list[str] | None = None


class AskRequest(BaseModel):
    """知识库问答请求。"""

    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=30)
    max_tokens: int = Field(default=1200, ge=128, le=8000)
    max_abstract: int = Field(default=700, ge=100, le=3000)


class DeletePapersRequest(BaseModel):
    ids: list[int]


class ImportRequest(BaseModel):
    """导入题录文件。

    ``content`` 直接放文件文本（前端用 FileReader 读出来），
    这样不需要 multipart 上传，也方便命令行/脚本调用。
    """

    content: str = Field(min_length=1)
    filename: str = ""
    source: str = "import"
    embed: bool = True
    dry_run: bool = False


class FeedbackRequest(BaseModel):
    """用户对 AI 输出的反馈 / 质疑。"""

    target_type: str = "message"      # message / artifact / paper / plan / manuscript / search_result
    target_id: str = ""
    run_id: str = ""
    session_id: int | None = None
    verdict: str = "up"               # up / down / challenge
    category: str = ""
    comment: str = ""
    #: 用户给出的正确版本 —— 有它才能变成"纠错记忆"与训练偏好对
    corrected_text: str = ""
    quoted_text: str = ""
    topic: str = ""


class ManuscriptRequest(BaseModel):
    """论文生成请求：用户的目标 / 方法 / 实验数据。"""

    brief: dict[str, Any] = Field(default_factory=dict)
    session_id: int | None = None
    #: 复用的文献材料（默认自动取本地知识库中与本课题最相关的若干篇）
    use_library: bool = True
    top_k: int = 12
    manuscript_id: int | None = None
    save: bool = True


class AgentRunRequest(BaseModel):
    topic: str = Field(min_length=1)
    sources: list[str] | None = None
    session_id: int | None = None
    project_id: int | None = None
    citation_style: str = "gb7714"
    require_approval: bool = True
    offline: bool | None = None
    #: 综述正文目标字数范围（中文字符，不含参考文献）。留空则用 config 默认值。
    review_min_chars: int | None = Field(default=None, ge=0, le=60000)
    review_max_chars: int | None = Field(default=None, ge=0, le=60000)


class ApprovalRequest(BaseModel):
    decision: str = "approve"
    feedback: str = ""


class SessionCreateRequest(BaseModel):
    title: str = "新会话"
    topic: str = ""


class CiteRequest(BaseModel):
    paper_ids: list[int]
    style: str = "gb7714"
    format: str = "list"


class ExportRequest(BaseModel):
    paper_ids: list[int] = Field(default_factory=list)
    format: str = "bibtex"
    name: str = "medscholar"


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class ProjectPapersRequest(BaseModel):
    paper_ids: list[int]
    note: str = ""


class EmbedRequest(BaseModel):
    limit: int = Field(default=200, ge=1, le=5000)
    force_ids: list[int] | None = None


class FulltextBackfillRequest(BaseModel):
    """补齐开放获取全文的请求。"""

    limit: int = Field(default=20, ge=1, le=500)
    paper_ids: list[int] | None = None


# ============================================================ 应用装配
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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


def create_app() -> FastAPI:
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

    _register_routes(app)
    _mount_static(app)
    return app


def _mount_static(app: FastAPI) -> None:
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

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        index_file = WEB_DIR / "index.html"
        if not index_file.is_file():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "前端文件缺失：medscholar/web/index.html 不存在。",
                    "api_docs": "/docs",
                },
            )
        html = await asyncio.to_thread(_render_index, index_file)
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        icon = WEB_DIR / "favicon.ico"
        if icon.is_file():
            return FileResponse(icon)
        return JSONResponse(status_code=204, content=None)


def _register_routes(app: FastAPI) -> None:
    # ------------------------------------------------------------ 健康与配置
    #
    # 设计要点：``/api/health`` **绝不能阻塞**。
    #
    # 早期实现每次调用都真的去生成一次 LLM 回复（"回复两个字：可用"）并探测嵌入模型。
    # 实测：模型已载入时耗时 4~5.6 秒，冷启动（模型不在内存）时 25~35 秒 ——
    # 而前端首屏就调它、超时只有 15 秒。结果是页面一打开就满屏"连不上后端"，
    # 而且每次刷新都白白烧掉一次推理。
    #
    # 现在改为：立即返回上次探测结果（可能标记为"检测中"），后台任务异步刷新。
    probe_cache: dict[str, Any] = {"llm": None, "embedding": None, "at": 0.0}
    probe_task: dict[str, "asyncio.Task[None] | None"] = {"task": None}
    started_at = time.time()

    async def _run_probes() -> None:
        cfg = get_config()
        try:
            llm_ok, llm_message = await get_llm(cfg).health()
        except Exception as exc:  # noqa: BLE001 - 探测失败只影响展示
            cfg = get_config()
            llm_ok, llm_message = False, f"{type(exc).__name__}: {exc}"
        try:
            embedding = await embedding_status(cfg)
        except Exception as exc:  # noqa: BLE001
            embedding = {
                "ok": False,
                "provider": cfg.embedding.provider,
                "model": cfg.embedding.model,
                "dim": cfg.embedding.dim,
                "message": str(exc),
            }
        probe_cache["llm"] = {
            "ok": llm_ok,
            "message": llm_message,
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
        }
        probe_cache["embedding"] = embedding
        probe_cache["at"] = time.time()

    def _schedule_probe(force: bool) -> None:
        """按需在后台刷新探测结果（同一时刻最多一个探测任务）。"""
        task = probe_task["task"]
        if task is not None and not task.done():
            return
        if not force and probe_cache["at"] and time.time() - probe_cache["at"] < 120.0:
            return
        probe_task["task"] = asyncio.create_task(_run_probes(), name="health-probe")

    @app.get("/api/ping", tags=["基础"])
    async def ping() -> dict[str, Any]:
        """极轻量的连通性探测：不读数据库、不碰模型，用于前端判断后端是否在线。"""
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": round(time.time() - started_at, 1),
        }

    @app.get("/api/health", tags=["基础"])
    async def health(probe: bool = False) -> dict[str, Any]:
        cfg = get_config()
        db = get_db()
        _schedule_probe(force=probe)

        llm_cached = probe_cache["llm"]
        embed_cached = probe_cache["embedding"]
        return {
            "ok": True,
            "version": __version__,
            "offline": cfg.offline,
            "uptime_s": round(time.time() - started_at, 1),
            # ok 为 None 表示"正在后台检测"，前端据此显示"检测中"而非"不可用"
            "llm": llm_cached
            or {
                "ok": None,
                "provider": cfg.llm.provider,
                "model": cfg.llm.model,
                "message": "正在后台检测模型可用性…",
            },
            "embedding": embed_cached
            or {
                "ok": None,
                "provider": cfg.embedding.provider,
                "model": cfg.embedding.model,
                "dim": cfg.embedding.dim,
                "message": "正在后台检测嵌入模型…",
            },
            "database": await asyncio.to_thread(db.stats),
            "sources": get_registry(cfg).describe(),
        }

    @app.get("/api/config", tags=["基础"])
    async def config_view() -> dict[str, Any]:
        return get_config().public_dict()

    @app.get("/api/stats", tags=["基础"])
    async def stats() -> dict[str, Any]:
        return get_db().stats()

    @app.get("/api/sources", tags=["基础"])
    async def sources() -> dict[str, Any]:
        cfg = get_config()
        return {"items": get_registry(cfg).describe(), "all": list(ALL_SOURCES)}

    # ---------------------------------------------------------------- 文献库
    @app.get("/api/papers", tags=["文献库"])
    async def papers_list(
        q: str | None = None,
        limit: int = Query(20, ge=1, le=500),
        offset: int = Query(0, ge=0),
        year_from: int | None = None,
        year_to: int | None = None,
        min_cited: int | None = None,
        open_access: bool | None = None,
        source: str | None = None,
        sources: str | None = None,
        journal: str | None = None,
        project_id: int | None = None,
        order_by: str = "created_desc",
    ) -> dict[str, Any]:
        db = get_db()
        filters: dict[str, Any] = {}
        if year_from:
            filters["year_from"] = year_from
        if year_to:
            filters["year_to"] = year_to
        if min_cited:
            filters["min_cited"] = min_cited
        if open_access:
            filters["open_access"] = True
        if source:
            filters["source"] = source
        if sources:
            filters["sources"] = [s.strip() for s in sources.split(",") if s.strip()]
        if journal:
            filters["journal"] = journal

        total = await asyncio.to_thread(
            repo.count_papers, filters=filters or None, project_id=project_id, db=db
        )
        items = await asyncio.to_thread(
            repo.list_papers,
            filters=filters or None,
            limit=limit,
            offset=offset,
            order_by=order_by,
            project_id=project_id,
            db=db,
        )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [p.to_dict() for p in items],
            "has_more": offset + len(items) < total,
            "query": q or "",
        }

    @app.post("/api/papers/search", tags=["文献库"])
    async def papers_search(req: LocalSearchRequest) -> dict[str, Any]:
        from ..retrieval import search_knowledge_base

        hits = await search_knowledge_base(
            req.query, top_k=req.top_k, filters=req.filters or None
        )
        return {"query": req.query, "count": len(hits), "items": [h.to_dict() for h in hits]}

    # ------------------------------------------------- 用户反馈与学习闭环
    @app.post("/api/feedback", tags=["学习闭环"])
    async def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
        """记录赞/踩/质疑。

        质疑若带上了「正确说法」，会**立刻**成为后续生成的纠错记忆
        （见 feedback.memories_as_prompt），不需要重新训练模型。
        """
        from ..feedback import FeedbackEntry, record_feedback

        entry = FeedbackEntry(
            target_type=req.target_type,
            target_id=req.target_id,
            run_id=req.run_id,
            session_id=req.session_id,
            verdict=req.verdict,
            category=req.category,
            comment=req.comment,
            corrected_text=req.corrected_text,
            quoted_text=req.quoted_text,
            topic=req.topic,
        )
        feedback_id = await asyncio.to_thread(record_feedback, entry, db=get_db())
        became_memory = bool(
            req.verdict == "challenge" and req.corrected_text.strip()
        )
        return {
            "ok": True,
            "feedback_id": feedback_id,
            "became_memory": became_memory,
            "message": (
                "已记住这条纠正，后续同类生成会参考它（立即生效）。"
                if became_memory
                else "已记录你的反馈。"
            ),
        }

    @app.get("/api/feedback", tags=["学习闭环"])
    async def list_feedback(
        run_id: str = "",
        session_id: int | None = None,
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        from ..feedback import list_feedback as _list

        items = await asyncio.to_thread(
            _list, run_id=run_id, session_id=session_id, limit=limit, db=get_db()
        )
        return {"count": len(items), "items": [i.to_dict() for i in items]}

    @app.get("/api/feedback/summary", tags=["学习闭环"])
    async def feedback_summary() -> dict[str, Any]:
        """学习闭环的可见进度：多少条反馈、多少条已成为生效记忆。"""
        from ..feedback import feedback_summary as _summary

        return await asyncio.to_thread(_summary, db=get_db())

    @app.get("/api/feedback/export", tags=["学习闭环"])
    async def feedback_export(limit: int = Query(1000, ge=1, le=10000)) -> dict[str, Any]:
        """导出 DPO 风格偏好对（把真实使用数据变成训练数据）。"""
        from ..feedback import export_preference_pairs

        pairs = await asyncio.to_thread(export_preference_pairs, limit=limit, db=get_db())
        return {"count": len(pairs), "pairs": pairs}

    # ------------------------------------------------------- 论文（IMRaD）
    @app.post("/api/manuscript/draft", tags=["论文写作"])
    async def manuscript_draft(req: ManuscriptRequest) -> dict[str, Any]:
        """基于用户的目标/方法/实验数据生成论文初稿，并做数字溯源校验。"""
        from ..manuscript import (
            ManuscriptBrief,
            assemble,
            check_number_provenance,
            draft_manuscript,
            save_manuscript,
        )

        brief = ManuscriptBrief.from_dict(req.brief)
        missing = brief.missing()
        if missing:
            raise HTTPException(
                status_code=400,
                detail="还缺少必要的论文要素：" + "、".join(missing) + "，请补齐后再生成。",
            )

        literature_text = ""
        if req.use_library:
            try:
                from ..retrieval import build_context_digest, search_knowledge_base

                hits = await search_knowledge_base(
                    " ".join(filter(None, [brief.title, brief.goal, brief.outcomes]))
                    or brief.title
                    or brief.goal,
                    top_k=req.top_k,
                )
                entries = [
                    (index, hit.paper) for index, hit in enumerate(hits, start=1)
                ]
                if entries:
                    literature_text = build_context_digest(entries, max_abstract=600)
            except Exception as exc:  # 文献材料拿不到不该阻断写作
                logger.warning("检索本地文献材料失败：%s", exc)

        # 纠错记忆：把用户此前明确指出过的错误带进提示词
        from ..feedback import memories_as_prompt

        memory_text = await asyncio.to_thread(
            memories_as_prompt, brief.goal or brief.title, db=get_db()
        )

        try:
            result = await draft_manuscript(
                brief, literature_text=literature_text, memory_text=memory_text
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("论文生成失败")
            raise HTTPException(status_code=500, detail=f"论文生成失败：{exc}") from exc

        draft = assemble(result["sections"], result["titles"], result["order"])
        checks = check_number_provenance(
            draft, brief=brief, literature_text=literature_text
        )
        checks["sections"] = result["order"]
        checks["generation_errors"] = result["errors"]
        checks["used_literature"] = bool(literature_text)
        checks["used_memories"] = bool(memory_text)

        manuscript_id = None
        if req.save:
            manuscript_id = await asyncio.to_thread(
                save_manuscript,
                title=brief.title or brief.goal[:50],
                brief=brief.to_dict(),
                draft=draft,
                checks=checks,
                session_id=req.session_id,
                manuscript_id=req.manuscript_id,
                db=get_db(),
            )

        return {
            "ok": True,
            "manuscript_id": manuscript_id,
            "draft": draft,
            "sections": result["sections"],
            "order": result["order"],
            "checks": checks,
        }

    @app.get("/api/manuscripts", tags=["论文写作"])
    async def manuscripts_list(
        session_id: int | None = None, limit: int = Query(20, ge=1, le=200)
    ) -> dict[str, Any]:
        from ..manuscript import list_manuscripts

        items = await asyncio.to_thread(
            list_manuscripts, session_id=session_id, limit=limit, db=get_db()
        )
        return {"count": len(items), "items": items}

    @app.get("/api/manuscripts/{manuscript_id}", tags=["论文写作"])
    async def manuscript_detail(manuscript_id: int) -> dict[str, Any]:
        from ..manuscript import get_manuscript

        record = await asyncio.to_thread(get_manuscript, manuscript_id, db=get_db())
        if record is None:
            raise HTTPException(status_code=404, detail=f"未找到稿件 id={manuscript_id}")
        return record

    @app.get("/api/manuscript/recheck/{manuscript_id}", tags=["论文写作"])
    async def manuscript_recheck(manuscript_id: int) -> dict[str, Any]:
        """重新做数字溯源校验（用户改过数据后可以再查一次）。"""
        from ..manuscript import ManuscriptBrief, check_number_provenance, get_manuscript

        record = await asyncio.to_thread(get_manuscript, manuscript_id, db=get_db())
        if record is None:
            raise HTTPException(status_code=404, detail=f"未找到稿件 id={manuscript_id}")
        brief = ManuscriptBrief.from_dict(record.get("brief") or {})
        checks = check_number_provenance(record.get("draft") or "", brief=brief)
        return {"manuscript_id": manuscript_id, "checks": checks}

    @app.post("/api/import", tags=["文献库"])
    async def papers_import(req: ImportRequest) -> dict[str, Any]:
        """导入题录文件（RIS / BibTeX / EndNote 标记 / WoS 纯文本 / CSV）。

        用于把 Web of Science、Scopus、Embase、CNKI、万方 等**导出**的题录
        搬进本地库。只解析用户提供的文件，不联网、不使用任何账号。
        """
        from ..importers import import_text

        if len(req.content) > 64 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件过大（上限 64 MB）")

        report = await import_text(
            req.content,
            filename=req.filename,
            source=req.source or "import",
            embed=req.embed,
            dry_run=req.dry_run,
            db=get_db(),
        )
        if not report.parsed and report.errors:
            # 格式不认识 / 没解析出条目：这是用户输入问题，400 更合适
            raise HTTPException(status_code=400, detail=report.errors[0])
        return report.to_dict()

    @app.get("/api/papers/{paper_id}", tags=["文献库"])
    async def paper_detail(paper_id: int) -> dict[str, Any]:
        paper = await asyncio.to_thread(repo.get_paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"未找到文献 paper_id={paper_id}")
        return paper.to_dict()

    @app.delete("/api/papers", tags=["文献库"])
    async def papers_delete(req: DeletePapersRequest) -> dict[str, Any]:
        removed = await asyncio.to_thread(repo.delete_papers, req.ids)
        return {"deleted": removed}

    @app.get("/api/papers/{paper_id}/references", tags=["文献库"])
    async def paper_references(paper_id: int, remote: bool = False) -> dict[str, Any]:
        local = await asyncio.to_thread(repo.get_references, paper_id)
        citing = await asyncio.to_thread(repo.get_citing_papers, paper_id)
        payload: dict[str, Any] = {"local_references": local, "local_citing": citing}
        if remote:
            from ..tools import call_tool

            result = await call_tool("fetch_references", {"paper_id": paper_id})
            payload["remote"] = result
        return payload

    @app.get("/api/papers/{paper_id}/fulltext", tags=["文献库"])
    async def paper_fulltext(paper_id: int, fetch: bool = False) -> dict[str, Any]:
        content = await asyncio.to_thread(repo.get_fulltext, paper_id)
        if not content and fetch:
            from ..tools import call_tool

            result = await call_tool("fetch_fulltext", {"paper_id": paper_id})
            content = result.get("text", "") if result.get("ok") else ""
            if not result.get("ok"):
                raise HTTPException(status_code=404, detail=result.get("error", "全文获取失败"))
        return {"paper_id": paper_id, "content": content, "char_count": len(content)}

    # ------------------------------------------------------------ 联网检索
    @app.post("/api/query/preview", tags=["检索"])
    async def query_preview(req: QueryPreviewRequest) -> dict[str, Any]:
        """把用户的检索输入解析成结构化查询，并给出各数据源的实际检索式。

        前端用它做「将检索：…」的实时回显 —— 用户不必记布尔语法，
        看一眼就知道空格/逗号/竖线/减号被理解成了什么。

        解析逻辑与真正检索时**完全同源**（同一个 ``medscholar.query``），
        因此预览所见即实际所发。
        """
        from ..query import BOOLEAN_SOURCES, SUPPORTED_SYNTAX_HELP, for_source, parse_query

        parsed = parse_query(req.query)
        sources = req.sources or ["pubmed", "europepmc", "openalex", "crossref"]
        return {
            "query": req.query,
            "parsed": parsed.to_dict(),
            "help": SUPPORTED_SYNTAX_HELP,
            "boolean_sources": sorted(BOOLEAN_SOURCES),
            "per_source": {name: for_source(parsed, name) for name in sources},
        }

    @app.post("/api/search/live", tags=["检索"])
    async def live_search(req: LiveSearchRequest) -> dict[str, Any]:
        cfg = get_config()
        if cfg.offline:
            raise HTTPException(status_code=409, detail="当前处于离线模式，无法联网检索")

        filters = SearchFilters(
            year_from=req.filters.get("year_from"),
            year_to=req.filters.get("year_to"),
            open_access_only=bool(req.filters.get("open_access")),
            sort=str(req.filters.get("sort") or "relevance"),
        )
        outcome = await get_registry(cfg).search(
            req.query,
            sources=req.sources,
            limit=req.limit,
            per_source_limit=req.per_source_limit,
            filters=filters,
        )

        saved = {"new": 0, "updated": 0}
        embedded: dict[str, Any] = {}
        if req.save and outcome.papers:
            result = await asyncio.to_thread(
                repo.insert_papers,
                outcome.papers,
                embed=req.embed and cfg.embedding.auto_embed,
            )
            saved = {"new": result["new"], "updated": result["updated"]}
            for paper, pid in zip(outcome.papers, result["ids"]):
                paper.paper_id = pid
            embedded = result.get("embedded") or {}

        return {
            "query": req.query,
            "count": len(outcome.papers),
            "raw_count": outcome.raw_count,
            "duration_ms": outcome.duration_ms,
            "sources": [s.to_dict() for s in outcome.statuses],
            "saved": saved,
            "embedded": embedded,
            "items": [p.to_dict() for p in outcome.papers],
        }

    # ------------------------------------------------------------ 知识库问答
    @app.post("/api/ask", tags=["问答"])
    async def ask(req: AskRequest, request: Request) -> StreamingResponse:
        """基于**本地知识库**回答问题（不启动完整研究工作流）。

        与 ``/api/agent/run`` 的区别：这里不做联网检索、不写综述、不走审批，
        只做「混合检索 → 拼接材料 → LLM 回答」，因此几秒到一分钟就能给出答案。

        以 SSE 流式返回，因为本地 8B 模型生成 300 字要约一分钟，
        不流式的话用户会以为卡死。
        """
        from ..llm.client import LLMError, get_llm
        from ..llm.prompts import ASK_SYSTEM, ask_user
        from ..retrieval import build_context_digest, search_knowledge_base

        question = (req.question or "").strip()
        cfg = get_config()
        db = get_db()

        async def generator() -> AsyncIterator[bytes]:
            def pack(event_type: str, payload: dict[str, Any]) -> bytes:
                body = json.dumps(payload, ensure_ascii=False, default=str)
                return f"event: {event_type}\ndata: {body}\n\n".encode("utf-8")

            try:
                # 1) 本地混合检索
                yield pack("status", {"message": "正在检索本地知识库…"})
                hits = await search_knowledge_base(question, top_k=req.top_k, db=db)

                # 2) 组装材料并分配引用编号
                entries: list[tuple[int, Any]] = []
                for index, hit in enumerate(hits, start=1):
                    paper = hit.paper
                    if paper.paper_id is None:
                        continue
                    entries.append((index, paper))
                digest = build_context_digest(entries, max_abstract=req.max_abstract) if entries else ""

                references = [
                    {
                        "index": index,
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "journal": paper.journal,
                        "pub_year": paper.pub_year,
                        "score": next(
                            (round(h.score, 5) for h in hits if h.paper.paper_id == paper.paper_id), None
                        ),
                    }
                    for index, paper in entries
                ]
                yield pack(
                    "references",
                    {
                        "count": len(references),
                        "items": references,
                        "message": (
                            f"命中 {len(references)} 篇相关文献"
                            if references
                            else "本地知识库中没有相关文献，将基于通用知识回答"
                        ),
                    },
                )

                # 3) 流式生成回答
                if cfg.offline and cfg.llm.provider != "ollama":
                    yield pack("error", {"message": "离线模式下无法调用云端模型，且本地模型不可用。"})
                    yield pack("done", {"ok": False})
                    return

                client = get_llm(cfg)
                await client.start()
                buffer = ""
                async for chunk in client.stream(
                    [{"role": "user", "content": ask_user(question, digest, paper_count=len(entries))}],
                    system=ASK_SYSTEM,
                    temperature=0.25,
                    max_tokens=req.max_tokens,
                ):
                    buffer += chunk
                    if await request.is_disconnected():
                        break
                    yield pack("token", {"text": chunk})

                text = buffer.strip()
                yield pack(
                    "done",
                    {
                        "ok": bool(text),
                        "answer": text,
                        "char_count": len(text),
                        "references": references,
                    },
                )
            except LLMError as exc:
                yield pack("error", {"message": f"模型调用失败：{exc}"})
                yield pack("done", {"ok": False, "answer": ""})
            except Exception as exc:  # pragma: no cover
                logger.exception("知识库问答失败")
                yield pack("error", {"message": f"{type(exc).__name__}: {exc}"})
                yield pack("done", {"ok": False, "answer": ""})

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------ Agent 工作流
    @app.post("/api/agent/run", tags=["Agent"])
    async def agent_run(req: AgentRunRequest) -> dict[str, Any]:
        try:
            handle = await get_runtime().start(
                topic=req.topic,
                sources=req.sources,
                session_id=req.session_id,
                project_id=req.project_id,
                citation_style=detect_style(req.citation_style),
                require_approval=req.require_approval,
                offline=req.offline,
                review_min_chars=req.review_min_chars,
                review_max_chars=req.review_max_chars,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": handle.run_id, "session_id": handle.state.session_id}

    @app.get("/api/agent/stream/{run_id}", tags=["Agent"])
    async def agent_stream(run_id: str, request: Request) -> StreamingResponse:
        async def generator() -> AsyncIterator[bytes]:
            try:
                async for event in get_runtime().stream(run_id):
                    if await request.is_disconnected():
                        break
                    payload = json.dumps(event.data, ensure_ascii=False, default=str)
                    yield f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")
            except asyncio.CancelledError:  # 客户端断开
                raise
            except Exception as exc:  # pragma: no cover
                logger.exception("SSE 推送异常")
                data = json.dumps({"message": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {data}\n\n".encode("utf-8")
                yield b"event: done\ndata: {}\n\n"

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/agent/approve/{run_id}", tags=["Agent"])
    async def agent_approve(run_id: str, req: ApprovalRequest) -> dict[str, Any]:
        ok = await get_runtime().approve(run_id, req.decision, req.feedback)
        if not ok:
            raise HTTPException(
                status_code=409, detail="该运行不存在、已完成或已经审批过"
            )
        return {"ok": True, "decision": req.decision}

    @app.post("/api/agent/cancel/{run_id}", tags=["Agent"])
    async def agent_cancel(run_id: str) -> dict[str, Any]:
        return {"ok": await get_runtime().cancel(run_id)}

    @app.get("/api/agent/runs", tags=["Agent"])
    async def agent_runs(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        """运行列表：内存中的实时运行 + 数据库里的历史记录（含被中断的）。

        历史记录很重要：服务重启会清空内存中的运行，而用户界面仍在等一个
        永远不会到来的草稿。有了历史，前端就能明确显示"这次运行在综合阶段被中断"。
        """
        live = {r["run_id"]: r for r in get_runtime().list_runs(limit=limit)}
        db = get_db()
        history = await asyncio.to_thread(repo.list_runs, limit=limit, db=db)
        steps_map = await asyncio.to_thread(
            repo.run_step_phases, [row["run_id"] for row in history], db=db
        )

        merged: list[dict[str, Any]] = []
        for row in history:
            run_id = row["run_id"]
            if run_id in live:
                item = dict(live.pop(run_id))
                item["artifact_title"] = row.get("artifact_title")
                merged.append(item)
            else:
                merged.append(
                    {
                        "run_id": run_id,
                        "topic": row["topic"],
                        "phase": row["phase"],
                        "phase_label": Phase(row["phase"]).label
                        if row["phase"] in {p.value for p in Phase}
                        else row["phase"],
                        "status": row["status"],
                        "papers": row["papers"],
                        "citations": row["citations"] or 0,
                        "artifact_id": row["artifact_id"],
                        "artifact_title": row.get("artifact_title"),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "errors": [row["error"]] if row["error"] else [],
                        "interrupted": row["status"] == "interrupted",
                        "from_history": True,
                        "done_phases": steps_map.get(run_id, []),
                        "resumable": _is_resumable(
                            row["status"], row["phase"], steps_map.get(run_id, [])
                        ),
                    }
                )
        # 内存里有、数据库里还没有的（极少见）也补上
        merged.extend(live.values())
        return {"runs": merged[:limit]}

    @app.post("/api/agent/resume/{run_id}", tags=["Agent"])
    async def agent_resume(run_id: str) -> dict[str, Any]:
        """从阶段快照继续一次被中断的运行。

        只重跑没做完的阶段：已经检索入库的文献、已写好的草稿都会直接复用，
        不必因为一次断线就从头再来（检索 + 撰写通常要几十分钟）。
        """
        try:
            handle = await get_runtime().resume(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run_id": handle.run_id,
            "session_id": handle.state.session_id,
            "resumed_from": handle.state.resumed_from,
            "papers": len(handle.state.papers),
        }

    @app.get("/api/agent/steps/{run_id}", tags=["Agent"])
    async def agent_steps(run_id: str) -> dict[str, Any]:
        """列出一次运行各阶段的快照摘要（供界面展示"已完成了哪些内容"）。"""
        steps = await asyncio.to_thread(repo.list_run_steps, run_id, db=get_db())
        out: list[dict[str, Any]] = []
        for step in steps:
            data = step.get("data") or {}
            plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
            draft = data.get("draft") if isinstance(data.get("draft"), str) else ""
            out.append(
                {
                    "phase": step["phase"],
                    "created_at": step["created_at"],
                    "queries": len(plan.get("queries") or []),
                    "outline": len(plan.get("outline") or []),
                    "papers": len(data.get("paper_ids") or []),
                    "has_critique": bool(data.get("critique")),
                    "draft_chars": len(draft),
                    "artifact_id": data.get("artifact_id"),
                }
            )
        return {"run_id": run_id, "steps": out}

    @app.get("/api/agent/latest", tags=["Agent"])
    async def agent_latest() -> dict[str, Any]:
        """最近一次可续跑或已完成的运行 + 其阶段摘要，供页面加载时恢复现场。"""
        db = get_db()
        runs = await asyncio.to_thread(repo.list_runs, limit=5, db=db)
        if not runs:
            return {"run": None, "steps": []}
        steps_map = await asyncio.to_thread(
            repo.run_step_phases, [r["run_id"] for r in runs], db=db
        )
        chosen = next(
            (
                r
                for r in runs
                if _is_resumable(r["status"], r["phase"], steps_map.get(r["run_id"], []))
            ),
            runs[0],
        )
        steps = await asyncio.to_thread(repo.list_run_steps, chosen["run_id"], db=db)
        summary = [
            {
                "phase": s["phase"],
                "created_at": s["created_at"],
                "papers": len((s.get("data") or {}).get("paper_ids") or []),
                "draft_chars": len((s.get("data") or {}).get("draft") or ""),
                "has_critique": bool((s.get("data") or {}).get("critique")),
            }
            for s in steps
        ]
        return {
            "run": {
                "run_id": chosen["run_id"],
                "topic": chosen["topic"],
                "phase": chosen["phase"],
                "status": chosen["status"],
                "papers": chosen["papers"],
                "artifact_id": chosen["artifact_id"],
                "artifact_title": chosen.get("artifact_title"),
                "updated_at": chosen["updated_at"],
                "done_phases": steps_map.get(chosen["run_id"], []),
                "resumable": _is_resumable(
                    chosen["status"],
                    chosen["phase"],
                    steps_map.get(chosen["run_id"], []),
                ),
            },
            "steps": summary,
        }

    @app.get("/api/agent/runs/{run_id}", tags=["Agent"])
    async def agent_run_detail(run_id: str) -> dict[str, Any]:
        handle = get_runtime().get(run_id)
        if handle is not None:
            return {**handle.to_dict(), "summary": handle.state.summary(), "errors": handle.state.errors}
        # 内存里没有（服务重启过）→ 回退到数据库记录，让前端能说明发生了什么
        record = await asyncio.to_thread(repo.get_run, run_id, db=get_db())
        if record is None:
            raise HTTPException(status_code=404, detail=f"运行不存在：{run_id}")
        return {
            "run_id": record["run_id"],
            "topic": record["topic"],
            "phase": record["phase"],
            "status": record["status"],
            "papers": record["papers"],
            "artifact_id": record["artifact_id"],
            "created_at": record["created_at"],
            "errors": [record["error"]] if record["error"] else [],
            "interrupted": record["status"] == "interrupted",
            "from_history": True,
        }

    # ---------------------------------------------------------------- 会话
    @app.get("/api/sessions", tags=["会话"])
    async def sessions_list(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        return {"items": await asyncio.to_thread(repo.list_sessions, limit=limit)}

    @app.post("/api/sessions", tags=["会话"])
    async def session_create(req: SessionCreateRequest) -> dict[str, Any]:
        session_id = await asyncio.to_thread(
            repo.create_session, title=req.title, topic=req.topic
        )
        return {"id": session_id}

    @app.get("/api/sessions/{session_id}/messages", tags=["会话"])
    async def session_messages(session_id: int, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        return {"items": await asyncio.to_thread(repo.list_messages, session_id, limit=limit)}

    @app.delete("/api/sessions/{session_id}", tags=["会话"])
    async def session_delete(session_id: int) -> dict[str, Any]:
        return {"ok": await asyncio.to_thread(repo.delete_session, session_id)}

    # ---------------------------------------------------------------- 产物
    @app.get("/api/artifacts", tags=["产物"])
    async def artifacts_list(
        session_id: int | None = None, limit: int = Query(50, ge=1, le=500)
    ) -> dict[str, Any]:
        return {"items": await asyncio.to_thread(repo.list_artifacts, session_id=session_id, limit=limit)}

    @app.get("/api/artifacts/{artifact_id}", tags=["产物"])
    async def artifact_detail(artifact_id: int) -> dict[str, Any]:
        artifact = await asyncio.to_thread(repo.get_artifact, artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"未找到产物 id={artifact_id}")
        return artifact

    @app.delete("/api/artifacts/{artifact_id}", tags=["产物"])
    async def artifact_delete(artifact_id: int) -> dict[str, Any]:
        return {"ok": await asyncio.to_thread(repo.delete_artifact, artifact_id)}

    # ------------------------------------------------------------ 引用与导出
    @app.get("/api/cite/styles", tags=["引用"])
    async def cite_styles() -> dict[str, Any]:
        return {"styles": [{"key": key, "label": STYLE_LABELS[key]} for key in STYLES]}

    @app.post("/api/cite", tags=["引用"])
    async def cite(req: CiteRequest) -> dict[str, Any]:
        from ..cite import format_records, format_reference_list

        if not req.paper_ids:
            raise HTTPException(status_code=400, detail="paper_ids 不能为空")
        mapping = await asyncio.to_thread(repo.get_papers_by_ids, req.paper_ids)
        papers = [mapping[i] for i in req.paper_ids if i in mapping]
        if not papers:
            raise HTTPException(status_code=404, detail="指定的文献不存在于本地库")

        style = detect_style(req.style)
        if req.format == "bibtex" or style == "bibtex":
            content = format_records(papers, "bibtex")
        elif req.format == "ris" or style == "ris":
            content = format_records(papers, "ris")
        else:
            content = format_reference_list(papers, style)
        return {
            "style": style,
            "label": STYLE_LABELS.get(style, style),
            "content": content,
            "count": len(papers),
        }

    @app.post("/api/export", tags=["引用"])
    async def export(req: ExportRequest) -> dict[str, Any]:
        from ..cite import format_records
        from ..export.exporters import write_export

        fmt = str(req.format).lower()
        mapping = await asyncio.to_thread(repo.get_papers_by_ids, req.paper_ids)
        papers = [mapping[i] for i in req.paper_ids if i in mapping]

        if fmt in {"bibtex", "ris"}:
            content = format_records(papers, fmt)
        elif fmt == "csv":
            content = export_csv(papers)
        elif fmt == "json":
            content = export_json(papers)
        elif fmt in {"apa7", "vancouver", "gb7714", "chicago"}:
            content = format_records(papers, fmt)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的导出格式：{req.format}（可选：{', '.join(EXPORT_FORMATS)}）",
            )

        path = await asyncio.to_thread(
            write_export, content, name=req.name, fmt=fmt
        )
        return {
            "path": str(path),
            "format": fmt,
            "bytes": len(content.encode("utf-8")),
            "count": len(papers),
        }

    # ---------------------------------------------------------------- 课题
    @app.get("/api/projects", tags=["课题"])
    async def projects_list() -> dict[str, Any]:
        return {"items": await asyncio.to_thread(repo.list_projects)}

    @app.post("/api/projects", tags=["课题"])
    async def project_create(req: ProjectCreateRequest) -> dict[str, Any]:
        project_id = await asyncio.to_thread(
            repo.create_project, req.name, description=req.description, keywords=req.keywords
        )
        return {"id": project_id}

    @app.get("/api/projects/{project_id}", tags=["课题"])
    async def project_detail(project_id: int) -> dict[str, Any]:
        project = await asyncio.to_thread(repo.get_project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"未找到课题 id={project_id}")
        return project

    @app.post("/api/projects/{project_id}/papers", tags=["课题"])
    async def project_add_papers(project_id: int, req: ProjectPapersRequest) -> dict[str, Any]:
        added = await asyncio.to_thread(
            repo.add_papers_to_project, project_id, req.paper_ids, note=req.note
        )
        return {"added": added}

    @app.delete("/api/projects/{project_id}", tags=["课题"])
    async def project_delete(project_id: int) -> dict[str, Any]:
        return {"ok": await asyncio.to_thread(repo.delete_project, project_id)}

    # ---------------------------------------------------------------- 维护
    @app.post("/api/maintenance/embed", tags=["维护"])
    async def maintenance_embed(req: EmbedRequest) -> dict[str, Any]:
        # 必须用异步版本：run_embedding_pipeline 是同步封装，await 它会直接抛 TypeError
        report = await run_embedding_pipeline_async(
            ids=req.force_ids, limit=req.limit, force=bool(req.force_ids)
        )
        return report.to_dict()

    @app.post("/api/maintenance/fulltext", tags=["维护"])
    async def maintenance_fulltext(req: FulltextBackfillRequest) -> dict[str, Any]:
        """为**开放获取**文献补齐全文（Europe PMC JATS → PMC → OA PDF）。

        默认入库的是元数据与摘要，不含全文 —— 这是刻意的：
        全文体积大、抓取慢，而且只有开放获取文献才允许保存。
        用户需要全文检索或更深入的综述引用时，用本接口按需补齐。
        """
        db = get_db()

        if req.paper_ids:
            candidates = [repo.get_paper(pid, db=db) for pid in req.paper_ids]
            papers = [p for p in candidates if p is not None]
        else:
            # 优先 PMCID（Europe PMC 稳定）→ 再 OA 链接；并跳过已有全文的，
            # 因此本操作可以反复执行而不会重复劳动
            papers = await asyncio.to_thread(
                repo.fulltext_candidates, limit=req.limit, db=db
            )

        # 只处理开放获取文献：非 OA 的一律跳过，绝不绕过付费墙
        targets = [p for p in papers if p and (p.is_open_access or p.pmcid)]
        skipped = len([p for p in papers if p]) - len(targets)

        from ..agent.reader import ReaderAgent, classify_fulltext_error

        reader = ReaderAgent(config=get_config(), db=db)
        fetched = 0
        failed = 0
        errors: list[str] = []
        #: 按原因归类统计 —— 只回一句"失败 100 条"让人无从判断，
        #: 而实际上其中近一半是"文献本身就没有正文"这类**正常情况**。
        reasons: dict[str, dict[str, Any]] = {}

        async def note_failure(paper: Any, message: str) -> None:
            nonlocal failed
            failed += 1
            errors.append(f"{paper.paper_id}: {message}")
            label, permanent = classify_fulltext_error(message)
            bucket = reasons.setdefault(
                label, {"label": label, "count": 0, "permanent": permanent, "example": ""}
            )
            bucket["count"] += 1
            bucket["example"] = bucket["example"] or f"#{paper.paper_id} {message[:120]}"
            if paper.paper_id:
                # 永久性的不再重试；网络类错误留待下次
                await asyncio.to_thread(
                    repo.record_fulltext_attempt,
                    paper.paper_id,
                    message,
                    permanent=permanent,
                    db=db,
                )

        try:
            for paper in targets:
                try:
                    result = await reader.fetch_fulltext(paper)
                except Exception as exc:  # noqa: BLE001 - 单篇失败不影响整批
                    await note_failure(paper, f"{type(exc).__name__}: {exc}")
                    continue
                if result.ok:
                    fetched += 1
                else:
                    await note_failure(paper, result.error or "未知原因")
        finally:
            await reader.close()

        reason_list = sorted(reasons.values(), key=lambda r: -r["count"])
        return {
            "candidates": len(papers),
            "open_access": len(targets),
            "skipped_not_oa": skipped,
            "fetched": fetched,
            "failed": failed,
            "reasons": reason_list,
            "permanent_failures": sum(r["count"] for r in reason_list if r["permanent"]),
            "retryable_failures": sum(r["count"] for r in reason_list if not r["permanent"]),
            "errors": errors[:10],
            "stats": await asyncio.to_thread(db.stats),
        }

    @app.post("/api/maintenance/reindex", tags=["维护"])
    async def maintenance_reindex() -> dict[str, Any]:
        db = get_db()
        await asyncio.to_thread(db.optimize_fts)
        return {"ok": True, "fts": "optimized", "stats": db.stats()}

    @app.post("/api/maintenance/vacuum", tags=["维护"])
    async def maintenance_vacuum() -> dict[str, Any]:
        db = get_db()
        await asyncio.to_thread(db.vacuum)
        return {"ok": True, "stats": db.stats()}

    # ---------------------------------------------------------------- 速读
    @app.post("/api/papers/{paper_id}/summarize", tags=["文献库"])
    async def paper_summarize(paper_id: int, topic: str = "") -> dict[str, Any]:
        paper = await asyncio.to_thread(repo.get_paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"未找到文献 paper_id={paper_id}")
        from ..agent.writer import WriterAgent

        try:
            text = await WriterAgent().summarize_paper(paper, topic=topic)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"生成速读笔记失败：{exc}") from exc
        return {"paper_id": paper_id, "summary": text}


app = create_app()

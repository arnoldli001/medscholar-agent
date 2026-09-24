"""会话、课题与维护路由：``tags=["会话"]`` + ``tags=["课题"]`` + ``tags=["维护"]``。

三者归在同一个 router，是因为它们都是围绕本地库的组织与保养：会话/消息
记录对话现场，课题把文献分组，维护接口负责嵌入补齐、开放获取全文补齐、FTS 优化与
VACUUM。它们都不触发 LLM 生成（全文补齐只抓开放获取正文，绝不绕过付费墙）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...db import repo
from ...embedding.pipeline import run_embedding_pipeline_async
from ..deps import get_config, get_db

router = APIRouter()


# ============================================================ 请求体模型
class SessionCreateRequest(BaseModel):
    title: str = "新会话"
    topic: str = ""


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


# ============================================================ 会话
@router.get("/api/sessions", tags=["会话"])
async def sessions_list(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    return {"items": await asyncio.to_thread(repo.list_sessions, limit=limit)}


@router.post("/api/sessions", tags=["会话"])
async def session_create(req: SessionCreateRequest) -> dict[str, Any]:
    session_id = await asyncio.to_thread(
        repo.create_session, title=req.title, topic=req.topic
    )
    return {"id": session_id}


@router.get("/api/sessions/{session_id}/messages", tags=["会话"])
async def session_messages(session_id: int, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    return {"items": await asyncio.to_thread(repo.list_messages, session_id, limit=limit)}


@router.delete("/api/sessions/{session_id}", tags=["会话"])
async def session_delete(session_id: int) -> dict[str, Any]:
    return {"ok": await asyncio.to_thread(repo.delete_session, session_id)}


# ============================================================ 课题
@router.get("/api/projects", tags=["课题"])
async def projects_list() -> dict[str, Any]:
    return {"items": await asyncio.to_thread(repo.list_projects)}


@router.post("/api/projects", tags=["课题"])
async def project_create(req: ProjectCreateRequest) -> dict[str, Any]:
    project_id = await asyncio.to_thread(
        repo.create_project, req.name, description=req.description, keywords=req.keywords
    )
    return {"id": project_id}


@router.get("/api/projects/{project_id}", tags=["课题"])
async def project_detail(project_id: int) -> dict[str, Any]:
    project = await asyncio.to_thread(repo.get_project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"未找到课题 id={project_id}")
    return project


@router.post("/api/projects/{project_id}/papers", tags=["课题"])
async def project_add_papers(project_id: int, req: ProjectPapersRequest) -> dict[str, Any]:
    added = await asyncio.to_thread(
        repo.add_papers_to_project, project_id, req.paper_ids, note=req.note
    )
    return {"added": added}


@router.delete("/api/projects/{project_id}", tags=["课题"])
async def project_delete(project_id: int) -> dict[str, Any]:
    return {"ok": await asyncio.to_thread(repo.delete_project, project_id)}


# ============================================================ 维护
@router.post("/api/maintenance/embed", tags=["维护"])
async def maintenance_embed(req: EmbedRequest) -> dict[str, Any]:
    # 必须用异步版本：run_embedding_pipeline 是同步封装，await 它会直接抛 TypeError
    report = await run_embedding_pipeline_async(
        ids=req.force_ids, limit=req.limit, force=bool(req.force_ids)
    )
    return report.to_dict()


@router.post("/api/maintenance/fulltext", tags=["维护"])
async def maintenance_fulltext(req: FulltextBackfillRequest) -> dict[str, Any]:
    """为开放获取文献补齐全文（Europe PMC JATS → PMC → OA PDF）。

    默认入库的是元数据与摘要，不含全文——这是刻意的：
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

    from ...agent.reader import ReaderAgent, classify_fulltext_error

    reader = ReaderAgent(config=get_config(), db=db)
    fetched = 0
    failed = 0
    errors: list[str] = []
    #: 按原因归类统计 —— 只回一句"失败 100 条"让人无从判断，
    #: 而实际上其中近一半是"文献本身就没有正文"这类正常情况。
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


@router.post("/api/maintenance/reindex", tags=["维护"])
async def maintenance_reindex() -> dict[str, Any]:
    db = get_db()
    await asyncio.to_thread(db.optimize_fts)
    return {"ok": True, "fts": "optimized", "stats": db.stats()}


@router.post("/api/maintenance/vacuum", tags=["维护"])
async def maintenance_vacuum() -> dict[str, Any]:
    db = get_db()
    await asyncio.to_thread(db.vacuum)
    return {"ok": True, "stats": db.stats()}

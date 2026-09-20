"""文献库路由（``tags=["文献库"]``）：列表、详情、删除、本地检索、导入、全文与速读。

这里的接口全部围绕**本地知识库**中的文献对象，不联网、不启动 Agent 工作流
（联网检索在 ``search.py``，综述生成在 ``agent.py``）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...constants import MAX_UPLOAD_BYTES
from ...db import repo
from ..deps import get_db

router = APIRouter()


# ============================================================ 请求体模型
class LocalSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)


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


# ============================================================ 路由
@router.get("/api/papers", tags=["文献库"])
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


@router.post("/api/papers/search", tags=["文献库"])
async def papers_search(req: LocalSearchRequest) -> dict[str, Any]:
    from ...retrieval import search_knowledge_base

    hits = await search_knowledge_base(
        req.query, top_k=req.top_k, filters=req.filters or None
    )
    return {"query": req.query, "count": len(hits), "items": [h.to_dict() for h in hits]}


@router.post("/api/import", tags=["文献库"])
async def papers_import(req: ImportRequest) -> dict[str, Any]:
    """导入题录文件（RIS / BibTeX / EndNote 标记 / WoS 纯文本 / CSV）。

    用于把 Web of Science、Scopus、Embase、CNKI、万方 等**导出**的题录
    搬进本地库。只解析用户提供的文件，不联网、不使用任何账号。
    """
    from ...importers import import_text

    if len(req.content) > MAX_UPLOAD_BYTES:
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


@router.get("/api/papers/{paper_id}", tags=["文献库"])
async def paper_detail(paper_id: int) -> dict[str, Any]:
    paper = await asyncio.to_thread(repo.get_paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"未找到文献 paper_id={paper_id}")
    return paper.to_dict()


@router.delete("/api/papers", tags=["文献库"])
async def papers_delete(req: DeletePapersRequest) -> dict[str, Any]:
    removed = await asyncio.to_thread(repo.delete_papers, req.ids)
    return {"deleted": removed}


@router.get("/api/papers/{paper_id}/references", tags=["文献库"])
async def paper_references(paper_id: int, remote: bool = False) -> dict[str, Any]:
    local = await asyncio.to_thread(repo.get_references, paper_id)
    citing = await asyncio.to_thread(repo.get_citing_papers, paper_id)
    payload: dict[str, Any] = {"local_references": local, "local_citing": citing}
    if remote:
        from ...tools import call_tool

        result = await call_tool("fetch_references", {"paper_id": paper_id})
        payload["remote"] = result
    return payload


@router.get("/api/papers/{paper_id}/fulltext", tags=["文献库"])
async def paper_fulltext(paper_id: int, fetch: bool = False) -> dict[str, Any]:
    content = await asyncio.to_thread(repo.get_fulltext, paper_id)
    if not content and fetch:
        from ...tools import call_tool

        result = await call_tool("fetch_fulltext", {"paper_id": paper_id})
        content = result.get("text", "") if result.get("ok") else ""
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=result.get("error", "全文获取失败"))
    return {"paper_id": paper_id, "content": content, "char_count": len(content)}


@router.post("/api/papers/{paper_id}/summarize", tags=["文献库"])
async def paper_summarize(paper_id: int, topic: str = "") -> dict[str, Any]:
    paper = await asyncio.to_thread(repo.get_paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"未找到文献 paper_id={paper_id}")
    from ...agent.writer import WriterAgent

    try:
        text = await WriterAgent().summarize_paper(paper, topic=topic)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"生成速读笔记失败：{exc}") from exc
    return {"paper_id": paper_id, "summary": text}

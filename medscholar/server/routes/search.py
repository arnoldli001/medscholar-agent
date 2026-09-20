"""检索与问答路由：``tags=["检索"]``（检索式预览、联网检索）+ ``tags=["问答"]``。

三者共享同一个"用户输入 → 检索式"的解析源头（``medscholar.query``），
预览与真正检索因此**所见即所发**；问答只打本地知识库，不联网、不写综述。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...api import SearchFilters
from ...constants import LLM_TEMPERATURE_PLAN
from ...db import repo
from ..deps import get_config, get_db, get_registry, sse_headers

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================ 请求体模型
class QueryPreviewRequest(BaseModel):
    """检索式预览请求。"""

    query: str = ""
    sources: list[str] | None = None


class LiveSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    sources: list[str] | None = None
    limit: int = Field(default=40, ge=1, le=500)
    per_source_limit: int = Field(default=20, ge=1, le=200)
    filters: dict[str, Any] = Field(default_factory=dict)
    save: bool = True
    embed: bool = True


class AskRequest(BaseModel):
    """知识库问答请求。"""

    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=30)
    max_tokens: int = Field(default=1200, ge=128, le=8000)
    max_abstract: int = Field(default=700, ge=100, le=3000)


# ============================================================ 路由
@router.post("/api/query/preview", tags=["检索"])
async def query_preview(req: QueryPreviewRequest) -> dict[str, Any]:
    """把用户的检索输入解析成结构化查询，并给出各数据源的实际检索式。

    前端用它做「将检索：…」的实时回显 —— 用户不必记布尔语法，
    看一眼就知道空格/逗号/竖线/减号被理解成了什么。

    解析逻辑与真正检索时**完全同源**（同一个 ``medscholar.query``），
    因此预览所见即实际所发。
    """
    from ...query import BOOLEAN_SOURCES, SUPPORTED_SYNTAX_HELP, for_source, parse_query

    parsed = parse_query(req.query)
    sources = req.sources or ["pubmed", "europepmc", "openalex", "crossref"]
    return {
        "query": req.query,
        "parsed": parsed.to_dict(),
        "help": SUPPORTED_SYNTAX_HELP,
        "boolean_sources": sorted(BOOLEAN_SOURCES),
        "per_source": {name: for_source(parsed, name) for name in sources},
    }


@router.post("/api/search/live", tags=["检索"])
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


@router.post("/api/ask", tags=["问答"])
async def ask(req: AskRequest, request: Request) -> StreamingResponse:
    """基于**本地知识库**回答问题（不启动完整研究工作流）。

    与 ``/api/agent/run`` 的区别：这里不做联网检索、不写综述、不走审批，
    只做「混合检索 → 拼接材料 → LLM 回答」，因此几秒到一分钟就能给出答案。

    以 SSE 流式返回，因为本地 8B 模型生成 300 字要约一分钟，
    不流式的话用户会以为卡死。
    """
    from ...llm.client import LLMError, get_llm
    from ...llm.prompts import ASK_SYSTEM, ask_user
    from ...retrieval import build_context_digest, search_knowledge_base

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
                temperature=LLM_TEMPERATURE_PLAN,
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
        headers=sse_headers(),
    )

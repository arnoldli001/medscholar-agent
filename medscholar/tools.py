"""工具层：以统一 schema 暴露给 MCP Server 与外部 Agent 调用。

每个工具都是「纯函数 + JSON Schema」，不依赖 FastAPI，
因此既能被 :mod:`medscholar.mcp.server` 包装成 MCP Tool，
也能被 CLI 直接调用。

设计约定：

* 输入输出只使用可无损 JSON 化的数据（不返回 ORM/连接对象）；
* 每个工具都自行捕获异常并返回 ``{"ok": false, "error": ...}``，
  避免一个工具失败导致整个 Agent 会话崩溃。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .api import SearchFilters, search_all
from .cite import STYLE_LABELS, detect_style, format_records, format_reference_list
from .config import get_config
from .db import (
    get_paper,
    get_papers_by_ids,
    list_papers,
    save_fulltext,
)
from .db.connect import get_db
from .models import Paper

logger = logging.getLogger(__name__)

__all__ = ["ToolSpec", "TOOLS", "tool_schemas", "call_tool"]


@dataclass(slots=True)
class ToolSpec:
    """一个工具的定义。"""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[..., Awaitable[dict[str, Any]]]

    def to_mcp(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema,
        }


# ------------------------------------------------------------------ 工具实现
async def _search_literature(
    query: str,
    sources: Sequence[str] | None = None,
    limit: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """联网检索学术文献（PubMed / Europe PMC / OpenAlex / Crossref / S2 / arXiv）。"""
    cfg = get_config()
    if cfg.offline:
        return {"ok": False, "error": "当前处于离线模式，无法联网检索"}

    filters = SearchFilters(
        year_from=year_from, year_to=year_to, open_access_only=open_access_only
    )
    outcome = await search_all(
        query, sources=list(sources or []) or None, limit=limit, filters=filters
    )

    saved = {"new": 0, "updated": 0}
    ids: list[int] = []
    papers = outcome.papers
    if save and papers:
        from .db.repo import insert_papers

        result = insert_papers(papers, embed=cfg.embedding.auto_embed)
        saved = {"new": result["new"], "updated": result["updated"]}
        ids = result["ids"]
        for paper, pid in zip(papers, ids):
            paper.paper_id = pid

    return {
        "ok": True,
        "query": query,
        "count": len(papers),
        "raw_count": outcome.raw_count,
        "duration_ms": outcome.duration_ms,
        "saved": saved,
        "sources": [s.to_dict() for s in outcome.statuses],
        "papers": [p.to_dict() for p in papers],
    }


async def _search_knowledge_base(
    query: str,
    top_k: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> dict[str, Any]:
    """在本地知识库中做混合检索（BM25 + 向量 + RRF 融合）。"""
    from .retrieval import search_knowledge_base as kb_search

    filters: dict[str, Any] = {}
    if year_from:
        filters["year_from"] = year_from
    if year_to:
        filters["year_to"] = year_to
    if open_access_only:
        filters["open_access"] = True

    hits = await kb_search(query, top_k=top_k, filters=filters or None)
    return {
        "ok": True,
        "query": query,
        "count": len(hits),
        "items": [h.to_dict() for h in hits],
    }


async def _get_paper(paper_id: int, with_fulltext: bool = False) -> dict[str, Any]:
    """按 ID 读取本地文献记录；可选附带已入库的开放获取全文。"""
    paper = get_paper(int(paper_id))
    if paper is None:
        return {"ok": False, "error": f"未找到文献 paper_id={paper_id}"}
    data = paper.to_dict()
    if with_fulltext:
        from .db.repo import get_fulltext

        data["full_text"] = get_fulltext(paper.paper_id or 0)
    return {"ok": True, "paper": data}


async def _list_library(
    limit: int = 20,
    offset: int = 0,
    order_by: str = "created_desc",
    year_from: int | None = None,
    year_to: int | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    """列出本地知识库中的文献。"""
    filters: dict[str, Any] = {}
    if year_from:
        filters["year_from"] = year_from
    if year_to:
        filters["year_to"] = year_to
    papers = list_papers(
        filters=filters or None,
        limit=int(limit),
        offset=int(offset),
        order_by=order_by,
        project_id=int(project_id) if project_id else None,
    )
    return {"ok": True, "count": len(papers), "papers": [p.to_dict() for p in papers]}


async def _library_stats() -> dict[str, Any]:
    """本地知识库统计（文献数、向量覆盖率、全文数等）。"""
    return {"ok": True, "stats": get_db().stats()}


async def _fetch_fulltext(paper_id: int) -> dict[str, Any]:
    """抓取某篇开放获取文献的全文（Europe PMC JATS → PMC → OA PDF）。"""
    paper = get_paper(int(paper_id))
    if paper is None:
        return {"ok": False, "error": f"未找到文献 paper_id={paper_id}"}

    from .agent.reader import ReaderAgent

    agent = ReaderAgent()
    try:
        result = await agent.fetch_fulltext(paper)
    finally:
        await agent.close()

    if not result.ok:
        return {"ok": False, "error": result.error, "paper_id": paper_id}
    return {
        "ok": True,
        "paper_id": paper_id,
        "origin": result.origin,
        "char_count": len(result.content),
        "text": result.content[:20000],
    }


async def _fetch_references(paper_id: int, limit: int = 50) -> dict[str, Any]:
    """获取某篇文献的参考文献（Europe PMC / Crossref / S2 自动择优）。"""
    paper = get_paper(int(paper_id))
    if paper is None:
        return {"ok": False, "error": f"未找到文献 paper_id={paper_id}"}

    from .api import CrossrefClient, EuropePMCClient, SemanticScholarClient

    for cls in (EuropePMCClient, CrossrefClient, SemanticScholarClient):
        client = cls()
        try:
            await client.start()
            refs = await client.references(paper)
        except Exception as exc:
            logger.debug("%s 取参考文献失败：%s", cls.__name__, exc)
            continue
        finally:
            await client.close()
        if refs:
            return {"ok": True, "source": cls.source_id, "count": len(refs), "references": refs[:limit]}
    return {"ok": True, "source": "", "count": 0, "references": [], "note": "各数据源均未返回参考文献"}


async def _format_citation(
    paper_ids: Sequence[int],
    style: str = "gb7714",
    fmt: str = "list",
) -> dict[str, Any]:
    """把文献格式化成指定引用样式（APA 7th / Vancouver / GB-T 7714 / BibTeX / RIS）。"""
    ids = [int(i) for i in (paper_ids or [])]
    if not ids:
        return {"ok": False, "error": "paper_ids 不能为空"}
    mapping = get_papers_by_ids(ids)
    papers: list[Paper] = [mapping[i] for i in ids if i in mapping]
    if not papers:
        return {"ok": False, "error": "指定的 paper_id 在本地库中不存在"}

    key = detect_style(style)
    if fmt == "bibtex" or key == "bibtex":
        content = format_records(papers, "bibtex")
    elif fmt == "ris" or key == "ris":
        content = format_records(papers, "ris")
    else:
        content = format_reference_list(papers, key)
    return {
        "ok": True,
        "style": key,
        "label": STYLE_LABELS.get(key, key),
        "count": len(papers),
        "content": content,
    }


async def _save_paper(
    title: str,
    abstract: str = "",
    authors: Sequence[str] | None = None,
    journal: str = "",
    pub_year: int | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    """手工录入一篇文献（例如从 Zotero 或浏览器复制来的条目）。"""
    from .db.repo import insert_paper

    if not (title or "").strip():
        return {"ok": False, "error": "title 不能为空"}
    paper = Paper(
        title=title,
        abstract=abstract,
        authors=list(authors or []),
        journal=journal,
        pub_year=pub_year,
        doi=doi,
        pmid=pmid,
        source=source,
    )
    paper_id, created = insert_paper(paper)
    return {"ok": True, "paper_id": paper_id, "created": created}


async def _save_fulltext(paper_id: int, content: str, origin: str = "manual") -> dict[str, Any]:
    """把一段全文文本写入知识库并建立 FTS5 索引。"""
    if not content.strip():
        return {"ok": False, "error": "content 不能为空"}
    save_fulltext(int(paper_id), content, origin=origin)
    return {"ok": True, "paper_id": int(paper_id), "char_count": len(content)}


# ------------------------------------------------------------------ 注册表
TOOLS: dict[str, ToolSpec] = {
    "search_literature": ToolSpec(
        name="search_literature",
        description=(
            "联网检索学术文献。支持 PubMed、Europe PMC、OpenAlex、Crossref、"
            "Semantic Scholar、arXiv。返回标题、作者、期刊、年份、摘要、DOI、被引数，"
            "并可选自动存入本地知识库。"
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词（中英文均可）"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "数据源列表；留空使用默认组合",
                },
                "limit": {"type": "integer", "description": "返回条数上限", "default": 20},
                "year_from": {"type": "integer", "description": "起始年份"},
                "year_to": {"type": "integer", "description": "截止年份"},
                "open_access_only": {"type": "boolean", "description": "仅开放获取", "default": False},
                "save": {"type": "boolean", "description": "是否存入本地知识库", "default": True},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_search_literature,
    ),
    "search_knowledge_base": ToolSpec(
        name="search_knowledge_base",
        description=(
            "在本地知识库中做混合检索（BM25 关键词 + 向量语义 + RRF 融合）。"
            "适合回答「我之前存过哪些相关文献」这类问题，无需联网。"
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词"},
                "top_k": {"type": "integer", "default": 20},
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "open_access_only": {"type": "boolean", "default": False},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_search_knowledge_base,
    ),
    "get_paper": ToolSpec(
        name="get_paper",
        description="按 paper_id 读取本地文献的完整元数据，可选附带已入库的全文。",
        schema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer"},
                "with_fulltext": {"type": "boolean", "default": False},
            },
            "required": ["paper_id"],
            "additionalProperties": False,
        },
        handler=_get_paper,
    ),
    "list_library": ToolSpec(
        name="list_library",
        description="分页列出本地知识库中的文献，可按年份与课题过滤、按年份或被引排序。",
        schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
                "order_by": {
                    "type": "string",
                    "enum": ["created_desc", "year_desc", "year_asc", "cited_desc", "title_asc"],
                    "default": "created_desc",
                },
                "year_from": {"type": "integer"},
                "year_to": {"type": "integer"},
                "project_id": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        handler=_list_library,
    ),
    "library_stats": ToolSpec(
        name="library_stats",
        description="返回本地知识库统计：文献总数、向量覆盖率、全文数、年份跨度、数据库体积。",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_library_stats,
    ),
    "fetch_fulltext": ToolSpec(
        name="fetch_fulltext",
        description=(
            "抓取某篇开放获取文献的全文纯文本（Europe PMC JATS → PMC → OA PDF），"
            "并写入本地库。非开放获取文献会被拒绝，不会绕过付费墙。"
        ),
        schema={
            "type": "object",
            "properties": {"paper_id": {"type": "integer"}},
            "required": ["paper_id"],
            "additionalProperties": False,
        },
        handler=_fetch_fulltext,
    ),
    "fetch_references": ToolSpec(
        name="fetch_references",
        description="获取某篇文献的参考文献列表（自动在 Europe PMC / Crossref / Semantic Scholar 中择优）。",
        schema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer"},
                "limit": {"type": "integer", "default": 50},
            },
            "required": ["paper_id"],
            "additionalProperties": False,
        },
        handler=_fetch_references,
    ),
    "format_citation": ToolSpec(
        name="format_citation",
        description=(
            "把本地文献格式化为参考文献。支持 apa7、vancouver、gb7714、chicago、"
            "bibtex、ris。"
        ),
        schema={
            "type": "object",
            "properties": {
                "paper_ids": {"type": "array", "items": {"type": "integer"}},
                "style": {
                    "type": "string",
                    "enum": ["apa7", "vancouver", "gb7714", "chicago", "bibtex", "ris"],
                    "default": "gb7714",
                },
                "format": {"type": "string", "enum": ["list", "bibtex", "ris"], "default": "list"},
            },
            "required": ["paper_ids"],
            "additionalProperties": False,
        },
        handler=_format_citation,
    ),
    "save_paper": ToolSpec(
        name="save_paper",
        description="手工录入一篇文献到本地知识库（用于补充无法通过 API 获取的条目）。",
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "abstract": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "journal": {"type": "string"},
                "pub_year": {"type": "integer"},
                "doi": {"type": "string"},
                "pmid": {"type": "string"},
                "source": {"type": "string", "default": "manual"},
            },
            "required": ["title"],
            "additionalProperties": False,
        },
        handler=_save_paper,
    ),
    "save_fulltext": ToolSpec(
        name="save_fulltext",
        description="把一段全文文本写入知识库并建立全文索引（供后续全文检索）。",
        schema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer"},
                "content": {"type": "string"},
                "origin": {"type": "string", "default": "manual"},
            },
            "required": ["paper_id", "content"],
            "additionalProperties": False,
        },
        handler=_save_fulltext,
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """返回全部工具的 MCP 风格 schema。"""
    return [spec.to_mcp() for spec in TOOLS.values()]


async def call_tool(name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """调用工具；任何异常都被转换为 ``{"ok": false, "error": ...}``。"""
    spec = TOOLS.get(name)
    if spec is None:
        return {"ok": False, "error": f"未知工具：{name}（可用：{', '.join(TOOLS)}）"}
    try:
        return await spec.handler(**dict(arguments or {}))
    except TypeError as exc:
        return {"ok": False, "error": f"参数不合法：{exc}"}
    except Exception as exc:
        logger.exception("工具 %s 执行失败", name)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def call_tool_sync(name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """同步调用工具；在事件循环内自动切到独立线程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_tool(name, arguments))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tool") as pool:
        return pool.submit(lambda: asyncio.run(call_tool(name, arguments))).result()

"""本地知识库检索门面。

把「查询词 → 向量 → BM25 + KNN + RRF → 排序结果」这条链路封装成一个调用，
供 Agent、HTTP API 和 MCP 工具共用。

嵌入后端不可用时会**自动退化为纯 BM25 关键词检索**，而不是整条链路失败。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from .config import AppConfig, get_config
from .db.connect import Database, get_db
from .db.repo import hybrid_search, search_fts, search_fulltext
from .embedding.pipeline import embed_query
from .models import Paper, ScoredPaper

logger = logging.getLogger(__name__)

__all__ = [
    "search_knowledge_base",
    "search_knowledge_base_sync",
    "keyword_search",
    "search_full_text",
    "papers_to_digest",
    "build_context_digest",
]


async def search_knowledge_base(
    query: str,
    *,
    top_k: int | None = None,
    filters: Mapping[str, Any] | None = None,
    use_vector: bool = True,
    db: Database | None = None,
    config: AppConfig | None = None,
) -> list[ScoredPaper]:
    """混合检索本地知识库。"""
    cfg = config or get_config()
    database = db or get_db()
    top_k = top_k or cfg.retrieval.top_k

    vector = None
    if use_vector and not cfg.offline:
        vector = await embed_query(query, config=cfg)

    return hybrid_search(
        query, embedding=vector, top_k=top_k, filters=filters, db=database
    )


def search_knowledge_base_sync(
    query: str,
    *,
    top_k: int | None = None,
    filters: Mapping[str, Any] | None = None,
    use_vector: bool = True,
    db: Database | None = None,
    config: AppConfig | None = None,
) -> list[ScoredPaper]:
    """同步入口；在事件循环内调用时自动切换到独立线程。"""
    kwargs = {
        "query": query,
        "top_k": top_k,
        "filters": filters,
        "use_vector": use_vector,
        "db": db,
        "config": config,
    }
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(search_knowledge_base(**kwargs))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kb") as pool:
        return pool.submit(lambda: asyncio.run(search_knowledge_base(**kwargs))).result()


def keyword_search(
    query: str,
    *,
    limit: int = 50,
    filters: Mapping[str, Any] | None = None,
    db: Database | None = None,
) -> list[tuple[int, float]]:
    """纯 BM25 关键词检索（不需要嵌入后端）。"""
    return search_fts(query, limit=limit, filters=filters, db=db)


def search_full_text(
    query: str, *, limit: int = 50, db: Database | None = None
) -> list[tuple[int, float]]:
    """在已入库的开放获取全文中检索。"""
    return search_fulltext(query, limit=limit, db=db)


def papers_to_digest(
    papers: Sequence[Paper | ScoredPaper],
    *,
    start_index: int = 1,
    max_abstract: int = 900,
    max_papers: int = 25,
) -> str:
    """把文献列表压成提示词材料（编号即正文引用序号）。"""
    from .llm.prompts import digest_papers

    plain: list[dict[str, Any]] = []
    for item in papers[:max_papers]:
        paper = item.paper if isinstance(item, ScoredPaper) else item
        plain.append(paper.to_dict())
    return digest_papers(plain, start_index=start_index, max_abstract=max_abstract)


def build_context_digest(
    entries: Sequence[tuple[int, Paper | ScoredPaper]],
    *,
    max_abstract: int = 900,
) -> str:
    """按**显式编号**构建材料块（编号与正文引用严格对应）。"""
    from .llm.prompts import digest_papers

    if not entries:
        return ""
    ordered = sorted(entries, key=lambda pair: pair[0])
    base = ordered[0][0]
    payload: list[dict[str, Any]] = []
    for index, item in ordered:
        paper = item.paper if isinstance(item, ScoredPaper) else item
        data = paper.to_dict()
        data["__index__"] = index
        payload.append(data)
    return digest_papers(payload, start_index=base, max_abstract=max_abstract)

"""Crossref 客户端。

Crossref 是 DOI 的官方注册机构，元数据权威、完全免费、无需 Key，
覆盖大量中文期刊与外文期刊，是 DOI 补全与引用列表的可靠来源。

* ``/works?query.bibliographic=`` —— 书目检索
* ``/works/{doi}`` —— 单篇详情（含 ``reference[]`` 参考文献表）
* 带 ``mailto`` 进入 polite pool
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..models import Paper
from ..query import for_source
from ..textutil import normalize_doi
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["CrossrefClient"]

_BASE = "https://api.crossref.org"
_TAG_RE = re.compile(r"<[^>]{1,80}>")
_WS_RE = re.compile(r"\s+")


def _strip_jats(text: str | None) -> str:
    """Crossref 的 abstract 是 JATS 片段，需要剥离标签。"""
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", str(text))
    for entity, repl in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&nbsp;", " ")):
        cleaned = cleaned.replace(entity, repl)
    return _WS_RE.sub(" ", cleaned).strip()


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


class CrossrefClient(BaseClient):
    name = "crossref"
    label = "Crossref"
    source_id = "crossref"
    base_url = _BASE

    def _common_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"rows": 20}
        if self.settings.email:
            params["mailto"] = self.settings.email
        return params

    @staticmethod
    def _build_filter(filters: SearchFilters | None) -> str:
        clauses: list[str] = []
        if not filters:
            return ""
        if filters.year_from:
            clauses.append(f"from-pub-date:{filters.year_from}-01-01")
        if filters.year_to:
            clauses.append(f"until-pub-date:{filters.year_to}-12-31")
        for ptype in filters.publication_types:
            clauses.append(f"type:{ptype}")
        return ",".join(clauses)

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        # 该数据源只做相关度检索、不支持布尔语法，把输入归一成核心词
        query = for_source(query or "", self.name).strip()
        if not query:
            return []

        sort = {
            "date": "published",
            "citations": "is-referenced-by-count",
        }.get(filters.sort if filters else "relevance", "relevance")

        params: dict[str, Any] = {
            **self._common_params(),
            "query.bibliographic": query,
            "rows": min(limit, 100),
            "sort": sort,
            "order": "desc" if sort != "relevance" else "desc",
        }
        filter_expr = self._build_filter(filters)
        if filter_expr:
            params["filter"] = filter_expr
        if filters and filters.open_access_only:
            params["filter"] = (params.get("filter", "") + ",has-license:true").strip(",")

        data = await self.request("GET", f"{_BASE}/works", params=params)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise SourceError(self.name, "Crossref 响应结构异常")

        papers: list[Paper] = []
        for item in message.get("items") or []:
            paper = self._parse_item(item)
            if paper and paper.title:
                papers.append(paper)
        return papers[:limit]

    # ---------------------------------------------------------------- 解析
    def _parse_item(self, item: dict[str, Any]) -> Paper | None:
        try:
            title = _first(item.get("title"))
            if not title:
                title = _first(item.get("subtitle"))

            authors: list[str] = []
            for author in item.get("author") or []:
                if not isinstance(author, dict):
                    continue
                name = author.get("name") or (
                    f"{author.get('family', '')} {author.get('given', '')}".strip()
                )
                if name:
                    authors.append(str(name))

            issued = item.get("issued") or item.get("published") or {}
            parts = (issued.get("date-parts") or [[None]])[0]
            pub_year = None
            if parts and parts[0]:
                try:
                    pub_year = int(parts[0])
                except (TypeError, ValueError):
                    pub_year = None

            published_online = item.get("published-online") or {}
            online_parts = (published_online.get("date-parts") or [[None]])[0]
            if pub_year is None and online_parts and online_parts[0]:
                try:
                    pub_year = int(online_parts[0])
                except (TypeError, ValueError):
                    pub_year = None

            # 参考文献表（用于本地引用图谱）
            refs = item.get("reference") or []
            keywords = [str(s) for s in (item.get("subject") or []) if s]

            return Paper(
                title=title,
                abstract=_strip_jats(item.get("abstract")),
                authors=authors,
                journal=_first(item.get("container-title")) or _first(item.get("publisher")),
                pub_year=pub_year,
                source=self.source_id,
                source_id=normalize_doi(item.get("DOI")),
                doi=normalize_doi(item.get("DOI")),
                keywords=keywords,
                cited_by_count=int(item.get("is-referenced-by-count") or 0),
                is_open_access=bool(item.get("license")),
                full_text_url=_first(item.get("link", [{}])[0].get("URL"))
                if isinstance(item.get("link"), list) and item.get("link")
                else "",
                url=str(item.get("URL") or ""),
                volume=str(item.get("volume") or ""),
                issue=str(item.get("issue") or ""),
                pages=str(item.get("page") or ""),
                publication_type=str(item.get("type") or ""),
                language=str(item.get("language") or ""),
                note=f"refs:{len(refs)}" if refs else "",
            )
        except Exception as exc:
            logger.debug("Crossref 记录解析跳过：%s", exc)
            return None

    # ------------------------------------------------------------ 引用关系
    async def references(self, paper: Paper) -> list[dict[str, Any]]:
        """取 ``reference[]``（Crossref 只注册被引条目，无被引方向）。"""
        doi = paper.doi
        if not doi:
            return []
        try:
            data = await self.request("GET", f"{_BASE}/works/{doi}")
        except SourceError as exc:
            logger.debug("Crossref references 失败：%s", exc)
            return []
        message = data.get("message") if isinstance(data, dict) else {}
        out: list[dict[str, Any]] = []
        for ref in (message or {}).get("reference") or []:
            out.append(
                {
                    "doi": normalize_doi(ref.get("DOI")),
                    "title": ref.get("article-title") or ref.get("volume-title") or "",
                    "journal": ref.get("journal-title") or "",
                    "year": ref.get("year"),
                    "authors": ref.get("author") or "",
                    "source": "crossref",
                }
            )
        return out

    async def fulltext(self, paper: Paper) -> str:
        """Crossref 不托管全文。"""
        return ""

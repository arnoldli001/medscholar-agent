"""OpenAlex 客户端。

OpenAlex 覆盖 2.5 亿+ 学术记录，元数据质量高、完全免费。

* ``/works`` 支持 ``search`` 全文检索与丰富的 ``filter`` 语法
* 摘要以 ``abstract_inverted_index`` 倒排形式返回，需要还原成文本
* 带上 ``mailto`` 即进入 polite pool，速率与稳定性更好（配置 ``sources.openalex.email``）
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from ..models import Paper
from ..query import for_source
from ..textutil import normalize_doi, to_family_first
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["OpenAlexClient", "reconstruct_abstract"]

_BASE = "https://api.openalex.org"
_MAX_PER_PAGE = 200


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _merge_unique(primary: list[Paper], extra: list[Paper]) -> list[Paper]:
    """合并两组结果，按 DOI（回退标题）去重，保持 primary 顺序优先。"""
    seen: set[str] = set()
    out: list[Paper] = []
    for paper in [*primary, *extra]:
        key = paper.doi or paper.title.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(paper)
    return out


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """把 OpenAlex 的倒排索引还原为摘要文本。

    >>> reconstruct_abstract({"Post-stroke": [0], "depression": [1]})
    'Post-stroke depression'
    """
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        if not isinstance(indexes, (list, tuple)):
            continue
        for index in indexes:
            try:
                positions.append((int(index), str(word)))
            except (TypeError, ValueError):
                continue
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


class OpenAlexClient(BaseClient):
    name = "openalex"
    label = "OpenAlex"
    source_id = "openalex"
    base_url = _BASE

    _FIELDS = (
        "id,doi,title,display_name,publication_year,publication_date,type,language,"
        "authorships,primary_location,best_oa_location,open_access,cited_by_count,"
        "abstract_inverted_index,mesh,keywords,concepts,referenced_works,ids,biblio"
    )

    def _common_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"per-page": 50}
        if self.settings.email:
            params["mailto"] = self.settings.email
        if self.settings.api_key:
            params["api_key"] = self.settings.api_key
        return params

    @staticmethod
    def _build_filter(filters: SearchFilters | None) -> str:
        clauses: list[str] = []
        if not filters:
            return ""
        if filters.year_from:
            clauses.append(f"from_publication_date:{filters.year_from}-01-01")
        if filters.year_to:
            clauses.append(f"to_publication_date:{filters.year_to}-12-31")
        if filters.open_access_only:
            clauses.append("is_oa:true")
        if filters.language:
            clauses.append(f"language:{filters.language}")
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
        if not query or not query.strip():
            return []

        # OpenAlex 的 search 是相关度检索，不支持布尔语法；用核心词
        # （含 OR 组的全部同义词，多给词能让排序更准）
        effective = for_source(query, self.name) or query.strip()
        papers = await self._search_once(effective, limit=limit, filters=filters)

        # 中文课题额外跑一遍 language:zh —— OpenAlex 对中文文献的语种过滤实测有效，
        # 这是在 CNKI 公开检索失效后获取中文文献最稳定的通路。
        if _has_cjk(query) and not (filters and filters.language):
            zh_filters = replace(filters, language="zh") if filters else SearchFilters(language="zh")
            try:
                zh_papers = await self._search_once(
                    query, limit=max(limit // 2, 5), filters=zh_filters
                )
            except SourceError as exc:
                logger.debug("OpenAlex 中文过滤检索失败：%s", exc)
                zh_papers = []
            if zh_papers:
                papers = _merge_unique(papers, zh_papers)

        return papers[:limit]

    async def _search_once(
        self,
        query: str,
        *,
        limit: int,
        filters: SearchFilters | None,
    ) -> list[Paper]:
        sort = {
            "date": "publication_date:desc",
            "citations": "cited_by_count:desc",
        }.get(filters.sort if filters else "relevance", "relevance_score:desc")

        collected: list[Paper] = []
        page = 1
        while len(collected) < limit and page <= 5:
            params: dict[str, Any] = {
                **self._common_params(),
                "search": query.strip(),
                "per-page": min(limit - len(collected), _MAX_PER_PAGE),
                "page": page,
                "sort": sort,
                "select": self._FIELDS,
            }
            filter_expr = self._build_filter(filters)
            if filter_expr:
                params["filter"] = filter_expr

            data = await self.request("GET", f"{_BASE}/works", params=params)
            results = data.get("results") or []
            if not results:
                break
            for item in results:
                paper = self._parse_work(item)
                if paper and paper.title:
                    collected.append(paper)
            if len(results) < int(params["per-page"]):
                break
            page += 1

        return collected[:limit]

    # ---------------------------------------------------------------- 解析
    def _parse_work(self, work: dict[str, Any]) -> Paper | None:
        try:
            title = work.get("title") or work.get("display_name") or ""

            authors: list[str] = []
            for authorship in work.get("authorships") or []:
                name = ((authorship or {}).get("author") or {}).get("display_name")
                if name:
                    authors.append(to_family_first(str(name)))

            primary = work.get("primary_location") or {}
            source_info = primary.get("source") or {}
            journal = str(source_info.get("display_name") or "")

            oa = work.get("open_access") or {}
            best_oa = work.get("best_oa_location") or {}
            full_text_url = str(best_oa.get("pdf_url") or best_oa.get("landing_page_url") or "")

            mesh: list[str] = []
            for heading in work.get("mesh") or []:
                descriptor = heading.get("descriptor_name")
                if descriptor:
                    mesh.append(str(descriptor))

            keywords: list[str] = []
            for kw in work.get("keywords") or []:
                if kw.get("display_name"):
                    keywords.append(str(kw["display_name"]))
            if not keywords:
                for concept in work.get("concepts") or []:
                    name = concept.get("display_name")
                    score = concept.get("score") or 0
                    if name and score >= 0.3:
                        keywords.append(str(name))

            biblio = work.get("biblio") or {}

            return Paper(
                title=str(title),
                abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
                authors=authors,
                journal=journal,
                pub_year=work.get("publication_year"),
                source=self.source_id,
                source_id=str(work.get("id") or "").rsplit("/", 1)[-1] or None,
                pmid=str((work.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1] or None,
                pmcid=str((work.get("ids") or {}).get("pmcid") or "").rsplit("/", 1)[-1] or None,
                doi=normalize_doi(work.get("doi")),
                mesh_terms=mesh,
                keywords=keywords,
                cited_by_count=int(work.get("cited_by_count") or 0),
                is_open_access=bool(oa.get("is_oa")),
                full_text_url=full_text_url,
                url=str(work.get("id") or ""),
                volume=str(biblio.get("volume") or ""),
                issue=str(biblio.get("issue") or ""),
                pages=(
                    f"{biblio.get('first_page')}-{biblio.get('last_page')}"
                    if biblio.get("first_page") and biblio.get("last_page")
                    else str(biblio.get("first_page") or "")
                ),
                publication_type=str(work.get("type") or ""),
                language=str(work.get("language") or ""),
            )
        except Exception as exc:
            logger.debug("OpenAlex 记录解析跳过：%s", exc)
            return None

    # ------------------------------------------------------------ 引用关系
    async def references(self, paper: Paper) -> list[dict[str, Any]]:
        """OpenAlex 存储的是 referenced_works 的 OpenAlex ID 列表。"""
        work_id = paper.source_id if paper.source == "openalex" else None
        if not work_id:
            if paper.doi:
                work_id = f"doi:{paper.doi}"
            elif paper.pmid:
                work_id = f"pmid:{paper.pmid}"
            else:
                return []
        try:
            data = await self.request(
                "GET",
                f"{_BASE}/works/{work_id}",
                params={**self._common_params(), "select": "referenced_works"},
            )
        except SourceError as exc:
            logger.debug("OpenAlex references 失败：%s", exc)
            return []
        return [
            {"openalex_id": str(ref).rsplit("/", 1)[-1], "source": "openalex"}
            for ref in (data.get("referenced_works") or [])
        ]

    async def citations(self, paper: Paper) -> list[dict[str, Any]]:
        """被引：用 ``cites:<work_id>`` 过滤查询。"""
        work_id = paper.source_id if paper.source == "openalex" else None
        if not work_id:
            if paper.doi:
                work_id = f"doi:{paper.doi}"
            elif paper.pmid:
                work_id = f"pmid:{paper.pmid}"
            else:
                return []
        try:
            data = await self.request(
                "GET",
                f"{_BASE}/works",
                params={
                    **self._common_params(),
                    "filter": f"cites:{work_id}",
                    "per-page": 50,
                    "select": "id,doi,title,publication_year,primary_location",
                },
            )
        except SourceError as exc:
            logger.debug("OpenAlex citations 失败：%s", exc)
            return []
        out: list[dict[str, Any]] = []
        for item in data.get("results") or []:
            out.append(
                {
                    "openalex_id": str(item.get("id") or "").rsplit("/", 1)[-1],
                    "doi": normalize_doi(item.get("doi")),
                    "title": item.get("title") or item.get("display_name") or "",
                    "year": item.get("publication_year"),
                    "source": "openalex",
                }
            )
        return out

    async def fulltext(self, paper: Paper) -> str:
        """OpenAlex 不托管全文，仅提供 OA 链接，由 Reader Agent 另行抓取。"""
        return ""

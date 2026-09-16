"""DOAJ（Directory of Open Access Journals）客户端。

补的是什么缺口：PubMed / Europe PMC 偏生物医学，OpenAlex 虽然全但不能只筛
"开放获取期刊"。DOAJ 收录两万余种**完全开放获取**期刊的论文题录，
对以下情形特别有用：

* 开放获取的综合性/工程/社科期刊论文（PubMed 不收）；
* 需要"只找能合法拿到全文的文献"时。

免费、无需 API Key。接口：``GET https://doaj.org/api/search/articles/{query}``
文档：https://doaj.org/api/v2/docs
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from ..models import Paper, coerce_int
from ..query import for_source
from ..textutil import normalize_doi
from .base import BaseClient, SearchFilters

logger = logging.getLogger(__name__)

__all__ = ["DoajClient"]

_BASE = "https://doaj.org/api"
_MAX_PAGE = 100


class DoajClient(BaseClient):
    name = "doaj"
    label = "DOAJ"
    source_id = "doaj"
    base_url = _BASE

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        text = for_source(query, "doaj").strip()
        if not text:
            return []
        page_size = min(_MAX_PAGE, max(1, min(limit, self.settings.page_size)))
        # DOAJ 把查询串放在**路径**里，不是查询参数
        url = f"{self.base_url}/search/articles/{quote(text, safe='')}"
        params: dict[str, Any] = {"pageSize": page_size, "page": 1}

        data = await self.request("GET", url, params=params)
        if not isinstance(data, dict):
            return []
        results = data.get("results")
        if not isinstance(results, list):
            return []

        papers: list[Paper] = []
        for item in results:
            paper = self._to_paper(item)
            if paper is None:
                continue
            if filters:
                if filters.year_from and (paper.pub_year or 0) and paper.pub_year < filters.year_from:
                    continue
                if filters.year_to and (paper.pub_year or 9999) and paper.pub_year > filters.year_to:
                    continue
            papers.append(paper)
            if len(papers) >= limit:
                break
        return papers

    @staticmethod
    def _to_paper(item: Any) -> Paper | None:
        """把 DOAJ 的 BibJSON 记录转成 :class:`Paper`。"""
        if not isinstance(item, dict):
            return None
        bib = item.get("bibjson")
        if not isinstance(bib, dict):
            return None
        title = str(bib.get("title") or "").strip()
        if not title:
            return None

        authors: list[str] = []
        for author in bib.get("author") or []:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author or "").strip()
            if name:
                authors.append(name)

        journal = bib.get("journal") if isinstance(bib.get("journal"), dict) else {}
        doi = ""
        full_text_url = ""
        url = ""
        for identifier in bib.get("identifier") or []:
            if isinstance(identifier, dict) and str(identifier.get("type", "")).lower() == "doi":
                doi = str(identifier.get("id") or "").strip()
                break
        for link in bib.get("link") or []:
            if not isinstance(link, dict):
                continue
            href = str(link.get("url") or "").strip()
            kind = str(link.get("type") or "").lower()
            if not href:
                continue
            if kind == "fulltext" and not full_text_url:
                full_text_url = href
            elif not url:
                url = href

        year = coerce_int(bib.get("year"))
        if year is None:
            year = coerce_int(str(bib.get("month") or "")[:4])

        keywords: list[str] = []
        for keyword in bib.get("keywords") or []:
            if isinstance(keyword, str) and keyword.strip():
                keywords.append(keyword.strip())

        return Paper(
            title=title,
            source="doaj",
            abstract=str(bib.get("abstract") or ""),
            authors=authors,
            journal=str(journal.get("title") or ""),
            pub_year=year,
            doi=normalize_doi(doi) if doi else None,
            keywords=keywords,
            volume=str(journal.get("volume") or ""),
            issue=str(journal.get("number") or ""),
            pages=str(bib.get("start_page") or ""),
            language="",
            publication_type="",
            url=url or (f"https://doi.org/{doi}" if doi else ""),
            # DOAJ 全部是完全开放获取期刊，因此标记 OA 是准确的
            is_open_access=True,
            full_text_url=full_text_url,
            source_id=str(item.get("id") or ""),
        )

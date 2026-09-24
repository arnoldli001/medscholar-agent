"""CORE 客户端（聚合全球机构知识库）。

机构库（大学/研究所自建库）里的学位论文、技术报告、会议论文和作者自存档
的录用稿，PubMed / OpenAlex / Crossref 收录不全；CORE 聚合了上万个机构库。

需要免费 API Key（注册后立即得到）：https://core.ac.uk/services/api
未配置 key 时注册表会跳过本客户端，不会报错中断检索。
接口：``GET https://api.core.ac.uk/v3/search/works?q=...``
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Paper, coerce_int
from ..query import for_source
from ..textutil import normalize_doi
from .base import BaseClient, SearchFilters

logger = logging.getLogger(__name__)

__all__ = ["CoreClient"]

_BASE = "https://api.core.ac.uk/v3"


class CoreClient(BaseClient):
    name = "core"
    label = "CORE"
    source_id = "core"
    base_url = _BASE

    def enabled(self) -> bool:
        """没有 API Key 就不能用，提前返回 False 让注册表跳过。"""
        return bool(self.settings.enabled and self.settings.api_key.strip())

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        text = for_source(query, "core").strip()
        if not text or not self.settings.api_key.strip():
            return []

        page_size = max(1, min(limit, self.settings.page_size))
        params: dict[str, Any] = {"q": text, "limit": page_size}
        # CORE 支持年份范围过滤，直接用可以省流量
        if filters and (filters.year_from or filters.year_to):
            low = filters.year_from or 1900
            high = filters.year_to or 2100
            params["q"] = f"{text} AND yearPublished>={low} AND yearPublished<={high}"

        data = await self.request(
            "GET",
            f"{self.base_url}/search/works",
            params=params,
            headers={"Authorization": f"Bearer {self.settings.api_key.strip()}"},
        )
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
            papers.append(paper)
            if len(papers) >= limit:
                break
        return papers

    @staticmethod
    def _to_paper(item: Any) -> Paper | None:
        if not isinstance(item, dict):
            return None
        title = str(item.get("title") or "").strip()
        if not title:
            return None

        authors: list[str] = []
        for author in item.get("authors") or []:
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author or "").strip()
            if name:
                authors.append(name)

        doi = normalize_doi(str(item.get("doi") or "")) or None
        download = str(item.get("downloadUrl") or "").strip()
        landing = str(item.get("sourceFulltextUrls") or "").strip()
        if not landing:
            landing = ""
        links = item.get("links")
        if isinstance(links, list) and links and not landing:
            landing = str(links[0] or "").strip()

        year = item.get("yearPublished")
        if year is None:
            year = item.get("publishedDate")

        return Paper(
            title=title,
            source="core",
            abstract=str(item.get("abstract") or ""),
            authors=authors,
            journal=str(item.get("publisher") or ""),
            pub_year=coerce_int(year),
            doi=doi,
            keywords=[
                str(k).strip() for k in (item.get("fieldOfStudy") or []) if str(k).strip()
            ],
            language=str((item.get("language") or {}).get("code") if isinstance(item.get("language"), dict) else item.get("language") or ""),
            url=landing or (f"https://doi.org/{doi}" if doi else ""),
            # 机构库里的东西基本都是开放获取的
            is_open_access=bool(download),
            full_text_url=download,
            source_id=str(item.get("id") or ""),
        )

"""Semantic Scholar Graph API 客户端。

强项是引用图谱（``references`` / ``citations``）与 TLDR 机器摘要，用于：

* Critic Agent 判断某篇文献在领域内的影响力
* 综述写作时回溯奠基性工作

限流说明：无 Key 约 100 次 / 5 分钟（≈0.33 次/秒），有 Key 为 1000 次 / 5 分钟。
无 Key 时 429 非常常见，因此本客户端实现了**自适应降速**：连续被限流就主动
拉低令牌桶速率，而不是硬撞限流墙。
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Paper
from ..query import for_source
from ..textutil import normalize_doi, to_family_first
from .base import BaseClient, RateLimited, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["SemanticScholarClient"]

_BASE = "https://api.semanticscholar.org/graph/v1"
_MIN_RPS = 0.05  # 最快也要 20 秒一次，避免被彻底封禁

_PAPER_FIELDS = (
    "paperId,title,abstract,year,venue,publicationVenue,journal,externalIds,"
    "authors,citationCount,influentialCitationCount,referenceCount,openAccessPdf,"
    "tldr,publicationTypes,fieldsOfStudy,publicationDate"
)


class SemanticScholarClient(BaseClient):
    name = "semantic_scholar"
    label = "Semantic Scholar"
    source_id = "s2"
    base_url = _BASE

    def __init__(self, settings=None, *, config=None, client=None) -> None:
        super().__init__(settings, config=config, client=client)
        if self.settings.api_key:
            self.bucket.update_rps(max(self.settings.rps, 3.0))
        self._consecutive_429 = 0

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.settings.api_key} if self.settings.api_key else {}

    # ------------------------------------------------------------ 限流自适应
    async def request(self, method: str, url: str, *, params=None, headers=None, expect="json"):
        merged = {**self._headers(), **(headers or {})}
        try:
            result = await super().request(
                method, url, params=params, headers=merged, expect=expect
            )
            self._consecutive_429 = 0
            return result
        except RateLimited:
            self._consecutive_429 += 1
            new_rps = max(_MIN_RPS, self.bucket.rps / 2)
            if new_rps < self.bucket.rps:
                logger.warning(
                    "Semantic Scholar 持续限流，速率 %.2f → %.2f 次/秒（配置免费 API Key 可解除）",
                    self.bucket.rps,
                    new_rps,
                )
                self.bucket.update_rps(new_rps)
            raise

    # ---------------------------------------------------------------- 检索
    @staticmethod
    def _year_param(filters: SearchFilters | None) -> str | None:
        if not filters:
            return None
        if filters.year_from and filters.year_to:
            return f"{filters.year_from}-{filters.year_to}"
        if filters.year_from:
            return f"{filters.year_from}-"
        if filters.year_to:
            return f"-{filters.year_to}"
        return None

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        # 该数据源只做相关度检索、不支持布尔语法，把输入归一成核心词
        query = for_source(query or "", self.name).strip()
        if len(query) < 3:
            return []

        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": _PAPER_FIELDS,
        }
        year = self._year_param(filters)
        if year:
            params["year"] = year
        if filters and filters.open_access_only:
            params["openAccessPdf"] = ""

        try:
            data = await self.request("GET", f"{_BASE}/paper/search", params=params)
        except SourceError as exc:
            # 检索端点最容易被限流；交给上层降级，不影响其他数据源
            logger.info("Semantic Scholar 检索不可用：%s", exc)
            raise

        papers: list[Paper] = []
        for item in data.get("data") or []:
            paper = self._parse_paper(item)
            if paper and paper.title:
                papers.append(paper)
        return papers

    # ---------------------------------------------------------------- 解析
    def _parse_paper(self, item: dict[str, Any]) -> Paper | None:
        try:
            external = item.get("externalIds") or {}
            authors = [
                to_family_first(str(a.get("name")))
                for a in (item.get("authors") or [])
                if a and a.get("name")
            ]

            journal = ""
            journal_info = item.get("journal")
            if isinstance(journal_info, dict):
                journal = str(journal_info.get("name") or "")
            if not journal:
                venue = item.get("publicationVenue")
                if isinstance(venue, dict):
                    journal = str(venue.get("name") or "")
            if not journal:
                journal = str(item.get("venue") or "")

            oa_pdf = item.get("openAccessPdf") or {}
            tldr = item.get("tldr") or {}
            abstract = item.get("abstract") or ""
            if not abstract and tldr.get("text"):
                abstract = f"[TLDR] {tldr['text']}"

            pages = ""
            if isinstance(journal_info, dict) and journal_info.get("pages"):
                pages = str(journal_info["pages"])

            return Paper(
                title=str(item.get("title") or ""),
                abstract=str(abstract),
                authors=authors,
                journal=journal,
                pub_year=item.get("year"),
                source=self.source_id,
                source_id=str(item.get("paperId") or "") or None,
                pmid=str(external.get("PubMed") or "") or None,
                pmcid=str(external.get("PubMedCentral") or "") or None,
                doi=normalize_doi(external.get("DOI")),
                keywords=[str(f) for f in (item.get("fieldsOfStudy") or []) if f],
                cited_by_count=int(item.get("citationCount") or 0),
                is_open_access=bool(oa_pdf.get("url")),
                full_text_url=str(oa_pdf.get("url") or ""),
                url=str(item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"),
                volume=str(journal_info.get("volume") or "") if isinstance(journal_info, dict) else "",
                pages=pages,
                publication_type="; ".join(
                    str(t) for t in (item.get("publicationTypes") or []) if t
                ),
            )
        except Exception as exc:
            logger.debug("Semantic Scholar 记录解析跳过：%s", exc)
            return None

    # ------------------------------------------------------------ 引用图谱
    def _paper_ref(self, paper: Paper) -> str | None:
        if paper.source == "s2" and paper.source_id:
            return paper.source_id
        if paper.doi:
            return f"DOI:{paper.doi}"
        if paper.pmid:
            return f"PMID:{paper.pmid}"
        return None

    async def references(self, paper: Paper, *, limit: int = 100) -> list[dict[str, Any]]:
        ref = self._paper_ref(paper)
        if not ref:
            return []
        try:
            data = await self.request(
                "GET",
                f"{_BASE}/paper/{ref}/references",
                params={"fields": "title,year,externalIds,venue,authors,citationCount", "limit": limit},
            )
        except SourceError as exc:
            logger.debug("S2 references 失败：%s", exc)
            return []
        return self._parse_ref_list(data, key="citedPaper")

    async def citations(self, paper: Paper, *, limit: int = 100) -> list[dict[str, Any]]:
        ref = self._paper_ref(paper)
        if not ref:
            return []
        try:
            data = await self.request(
                "GET",
                f"{_BASE}/paper/{ref}/citations",
                params={"fields": "title,year,externalIds,venue,authors,citationCount", "limit": limit},
            )
        except SourceError as exc:
            logger.debug("S2 citations 失败：%s", exc)
            return []
        return self._parse_ref_list(data, key="citingPaper")

    @staticmethod
    def _parse_ref_list(data: dict[str, Any], *, key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in data.get("data") or []:
            item = entry.get(key) or {}
            external = item.get("externalIds") or {}
            authors = [a.get("name") for a in (item.get("authors") or []) if a and a.get("name")]
            out.append(
                {
                    "s2_id": item.get("paperId"),
                    "doi": normalize_doi(external.get("DOI")),
                    "pmid": str(external.get("PubMed") or "") or None,
                    "title": item.get("title") or "",
                    "year": item.get("year"),
                    "journal": item.get("venue") or "",
                    "authors": ", ".join(str(a) for a in authors),
                    "cited_by_count": item.get("citationCount"),
                    "source": "s2",
                }
            )
        return out

    async def fulltext(self, paper: Paper) -> str:
        """S2 只提供 OA PDF 链接，不托管正文。"""
        return ""

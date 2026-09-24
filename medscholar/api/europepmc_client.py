"""Europe PMC 客户端。

Europe PMC 是覆盖面与开放程度都很好的免费数据源：

* ``/search``（``resultType=core``）—— 4000 万+ 文献元数据 + 摘要 + MeSH + 引用数
* ``cursorMark`` 游标分页 —— 可稳定翻取大结果集
* ``/{source}/{id}/fullTextXML`` —— 800 万+ 开放获取全文（本项目的全文主力）
* ``/{source}/{id}/references`` / ``/citations`` —— 引用图谱

无 Key 可用，公开速率约 10 次/秒。
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from ..models import Paper
from ..query import for_source
from ..textutil import normalize_doi, to_family_first
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["EuropePMCClient"]

_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_MAX_PAGE = 100
_TAG_RE = re.compile(r"<[^>]{1,60}>")


def _strip_tags(text: str | None) -> str:
    if not text:
        return ""
    return _TAG_RE.sub(" ", str(text))


class EuropePMCClient(BaseClient):
    name = "europepmc"
    label = "Europe PMC"
    source_id = "europepmc"
    base_url = _BASE

    # ---------------------------------------------------------------- 检索
    @staticmethod
    def _build_query(query: str, filters: SearchFilters | None) -> str:
        """翻译为 Europe PMC 查询语法（``AND`` + 字段标签）。"""
        parts = [f"({query.strip()})"] if query.strip() else []
        # SRC:PPR = 预印本（medRxiv / bioRxiv / Research Square 等都由 Europe PMC 索引），
        # 所以不需要再单独接 medRxiv 客户端。
        # 之前这行注释写成"排除预印本重复"，与实际的 OR 逻辑相反，已更正。
        parts.append("(SRC:MED OR SRC:PMC OR SRC:PPR OR SRC:AGR OR SRC:PAT)")
        if filters:
            if filters.year_from:
                parts.append(f"(PUB_YEAR:[{filters.year_from} TO 3000])")
            if filters.year_to:
                parts.append(f"(PUB_YEAR:[1000 TO {filters.year_to}])")
            if filters.open_access_only:
                parts.append("(OPEN_ACCESS:Y)")
            if filters.language:
                parts.append(f"(LANG:{filters.language})")
            for ptype in filters.publication_types:
                parts.append(f'(PUB_TYPE:"{ptype}")')
        return " AND ".join(parts)

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        # Europe PMC 支持与 PubMed 一致的布尔语法
        expr = self._build_query(for_source(query, self.name), filters)
        if not expr:
            return []

        sort = "CITED desc" if (filters and filters.sort == "citations") else ""
        results: list[Paper] = []
        cursor = "*"
        page_size = min(limit, _MAX_PAGE)

        while len(results) < limit:
            params: dict[str, Any] = {
                "query": expr,
                "format": "json",
                "resultType": "core",
                "pageSize": page_size,
                "cursorMark": cursor,
            }
            if sort:
                params["sort"] = sort

            data = await self.request("GET", f"{_BASE}/search", params=params)
            hits = (data.get("resultList") or {}).get("result") or []
            if not hits:
                break
            for hit in hits:
                paper = self._parse_hit(hit)
                if paper and paper.title:
                    results.append(paper)
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor or len(results) >= limit:
                break
            cursor = next_cursor

        return results[:limit]

    # ---------------------------------------------------------------- 解析
    def _parse_hit(self, hit: dict[str, Any]) -> Paper | None:
        try:
            pmid = (hit.get("pmid") or "").strip() or None
            pmcid = (hit.get("pmcid") or "").strip() or None
            doi = normalize_doi(hit.get("doi"))

            authors: list[str] = []
            author_list = (hit.get("authorList") or {}).get("author") or []
            for author in author_list:
                if author.get("collectiveName"):
                    authors.append(str(author["collectiveName"]))
                    continue
                # 优先用结构化的 lastName/firstName（天然是「姓 名」顺序）
                last = str(author.get("lastName") or "").strip()
                first = str(author.get("firstName") or "").strip()
                if last:
                    authors.append(f"{last} {first}".strip())
                    continue
                full = str(author.get("fullName") or "").strip()
                if full:
                    authors.append(to_family_first(full))
            if not authors and hit.get("authorString"):
                authors = [
                    to_family_first(a)
                    for a in str(hit["authorString"]).rstrip(".").split(",")
                    if a.strip()
                ]

            mesh = []
            for heading in (hit.get("meshHeadingList") or {}).get("meshHeading") or []:
                descriptor = heading.get("descriptorName")
                if descriptor:
                    mesh.append(str(descriptor))

            keywords = []
            for kw in (hit.get("keywordList") or {}).get("keyword") or []:
                if isinstance(kw, dict):
                    kw = kw.get("keyword") or kw.get("value")
                if kw:
                    keywords.append(str(kw))

            # 全文链接：优先 PMC，其次 publisher 的 OA 链接
            full_text_url = ""
            url_lists = ((hit.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
            for entry in url_lists:
                if str(entry.get("documentStyle", "")).lower() == "pdf":
                    continue
                availability = str(entry.get("availability", ""))
                if "Open access" in availability or "Free" in availability:
                    full_text_url = str(entry.get("url") or "")
                    if full_text_url:
                        break
            if not full_text_url and pmcid:
                full_text_url = f"https://europepmc.org/article/PMC/{pmcid}"

            journal = ""
            journal_info = hit.get("journalInfo") or {}
            if isinstance(journal_info, dict):
                journal = str((journal_info.get("journal") or {}).get("title") or "")
            pub_year = None
            try:
                pub_year = int(hit.get("pubYear")) if hit.get("pubYear") else None
            except (TypeError, ValueError):
                pub_year = None
            if pub_year is None and isinstance(journal_info, dict):
                try:
                    pub_year = int(journal_info.get("yearOfPublication"))
                except (TypeError, ValueError):
                    pub_year = None

            cited = 0
            try:
                cited = int(hit.get("citedByCount") or 0)
            except (TypeError, ValueError):
                cited = 0

            is_oa = str(hit.get("isOpenAccess", "")).upper() == "Y"

            return Paper(
                title=_strip_tags(hit.get("title")),
                abstract=_strip_tags(hit.get("abstractText")),
                authors=authors,
                journal=journal,
                pub_year=pub_year,
                source=self.source_id,
                source_id=str(hit.get("id") or "") or None,
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                mesh_terms=mesh,
                keywords=keywords,
                cited_by_count=cited,
                is_open_access=is_oa,
                full_text_url=full_text_url,
                url=f"https://europepmc.org/article/{hit.get('source', 'MED')}/{hit.get('id', '')}",
                volume=str(journal_info.get("volume") or "") if isinstance(journal_info, dict) else "",
                issue=str(journal_info.get("issue") or "") if isinstance(journal_info, dict) else "",
                pages=str(hit.get("pageInfo") or ""),
                publication_type=str(hit.get("pubType") or ""),
                language=str(hit.get("language") or ""),
            )
        except Exception as exc:  # 单条异常不影响整体
            logger.debug("Europe PMC 记录解析跳过：%s", exc)
            return None

    # ------------------------------------------------------------ 引用关系
    def _paper_ref(self, paper: Paper) -> tuple[str, str] | None:
        """Europe PMC 需要 ``source`` + ``id`` 组合定位文献。"""
        if paper.pmid:
            return "MED", paper.pmid
        if paper.pmcid:
            return "PMC", paper.pmcid
        if paper.doi:
            return "DOI", paper.doi
        return None

    async def references(self, paper: Paper) -> list[dict[str, Any]]:
        ref = self._paper_ref(paper)
        if not ref:
            return []
        source, ident = ref
        try:
            data = await self.request(
                "GET", f"{_BASE}/{source}/{ident}/references", params={"format": "json", "pageSize": 100}
            )
        except SourceError as exc:
            logger.debug("Europe PMC references 失败：%s", exc)
            return []
        return self._parse_citation_list(data)

    async def citations(self, paper: Paper) -> list[dict[str, Any]]:
        ref = self._paper_ref(paper)
        if not ref:
            return []
        source, ident = ref
        try:
            data = await self.request(
                "GET", f"{_BASE}/{source}/{ident}/citations", params={"format": "json", "pageSize": 100}
            )
        except SourceError as exc:
            logger.debug("Europe PMC citations 失败：%s", exc)
            return []
        return self._parse_citation_list(data)

    @staticmethod
    def _parse_citation_list(data: dict[str, Any]) -> list[dict[str, Any]]:
        listing = data.get("referenceList") or data.get("citationList") or {}
        entries = listing.get("reference") or listing.get("citation") or []
        out: list[dict[str, Any]] = []
        for entry in entries:
            src = str(entry.get("source") or "").upper()
            ident = str(entry.get("id") or "").strip()
            out.append(
                {
                    "pmid": ident if (src == "MED" and ident) else None,
                    "pmcid": ident if (src == "PMC" and ident) else None,
                    "doi": normalize_doi(entry.get("doi")),
                    "title": _strip_tags(entry.get("title")),
                    "year": entry.get("pubYear"),
                    "journal": entry.get("journalAbbreviation") or entry.get("journalTitle") or "",
                    "authors": entry.get("authorString") or "",
                    "source": "europepmc",
                }
            )
        return out

    # ---------------------------------------------------------------- 全文
    async def fulltext(self, paper: Paper) -> str:
        """获取开放获取全文纯文本；非 OA 文献一律返回空串（不越权）。

        实测 URL 形态：

        * ``/PMC/{PMCID}/fullTextXML``  → 404
        * ``/MED/{PMID}/fullTextXML``   → 404
        * ``/{PMCID}/fullTextXML``      → 200，返回 JATS 正文

        即全文端点不带 source 段，直接以 PMCID 作为路径；PMID 与 DOI 均不可用。
        """
        if not (paper.is_open_access or paper.pmcid):
            return ""

        candidates: list[str] = []
        if paper.pmcid:
            pmcid = paper.pmcid if paper.pmcid.upper().startswith("PMC") else f"PMC{paper.pmcid}"
            candidates.append(f"{_BASE}/{pmcid}/fullTextXML")

        for url in candidates:
            try:
                xml_text = await self.request("GET", url, expect="text")
            except SourceError as exc:
                logger.debug("Europe PMC 全文不可用 %s：%s", url, exc)
                continue
            text = _jats_to_text(xml_text)
            if text:
                return text
        return ""


def _jats_to_text(xml_text: str) -> str:
    """JATS XML → 纯文本（正文标题与段落，忽略表格细节）。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""
    chunks: list[str] = []
    for element in body.iter():
        tag = element.tag.split("}")[-1]
        if tag in {"title", "p", "td", "th", "caption", "label"}:
            text = "".join(element.itertext()).strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)

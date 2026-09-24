"""arXiv 客户端（预印本）。

医学影像 / 神经调控 / 脑机接口等方向的工作常先发在 arXiv 或 medRxiv。
arXiv 无 PMID/MeSH、不覆盖临床医学主体，默认只作补充来源，不参与去重优先级判断。
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from ..models import Paper
from ..query import for_source, parse_query
from ..textutil import normalize_doi, to_family_first
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["ArxivClient"]

_BASE = "http://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_WS_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()


class ArxivClient(BaseClient):
    name = "arxiv"
    label = "arXiv"
    source_id = "arxiv"
    base_url = _BASE

    @staticmethod
    def _significant_terms(query: str) -> list[str]:
        """提取用于相关度校验的实词（去掉停用词与过短词）。"""
        stop = {
            "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
            "were", "its", "their", "a", "an", "of", "in", "on", "to", "by", "at", "or",
            "as", "is", "be", "we", "our", "study", "trial", "effect", "effects",
        }
        terms = re.split(r"[\s,;:/()\[\]]+", query.lower())
        return [t for t in terms if len(t) > 2 and t not in stop]

    @staticmethod
    def _candidate_queries(query: str) -> list[str]:
        """渐进放宽的查询序列：精确短语 → 全部实词 AND → 前几个实词 AND。

        arXiv 的相关度排序对长自然语言查询很弱（实测整句短语常常 0 命中，
        而宽松 OR 又会返回完全无关的论文），因此这里逐级放宽并对结果做词面校验。
        """
        query = _clean(query)
        if not query:
            return []
        terms = ArxivClient._significant_terms(query)
        candidates: list[str] = []
        if " " in query:
            candidates.append(f'all:"{query}"')
        if len(terms) >= 2:
            candidates.append(" AND ".join(f"all:{t}" for t in terms))
        if len(terms) >= 3:
            candidates.append(" AND ".join(f"all:{t}" for t in terms[:3]))
        if not candidates:
            candidates.append(f"all:{query}")
        # 去重保序
        seen: set[str] = set()
        return [c for c in candidates if not (c in seen or seen.add(c))]

    @staticmethod
    def _is_relevant(paper: Paper, terms: list[str], *, min_hits: int | None = None) -> bool:
        """词面相关度校验：标题+摘要中至少命中若干个实词。"""
        if not terms:
            return True
        haystack = f"{paper.title} {paper.abstract}".lower()
        hits = sum(1 for term in terms if term in haystack)
        required = min_hits if min_hits is not None else min(2, len(terms))
        return hits >= required

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        candidates = self._candidate_queries(query)
        if not candidates:
            return []
        # 用户写了布尔语法（| / OR / - / NOT）时，arXiv 原生支持
        # AND/OR/ANDNOT，直接优先用它检索，比把整句塞进 all:"..." 精确得多。
        parsed = parse_query(query)
        if not parsed.is_empty and not parsed.is_simple:
            boolean_expr = for_source(parsed, self.name)
            if boolean_expr and boolean_expr not in candidates:
                candidates.insert(0, boolean_expr)
        terms = self._significant_terms(query)

        sort = {
            "date": "submittedDate",
            "citations": "relevance",
        }.get(filters.sort if filters else "relevance", "relevance")

        for expr in candidates:
            xml_text = await self.request(
                "GET",
                _BASE,
                params={
                    "search_query": expr,
                    "start": 0,
                    "max_results": min(limit * 3, 100),
                    "sortBy": sort,
                    "sortOrder": "descending",
                },
                expect="text",
            )
            parsed = self._parse_feed(xml_text)
            relevant = [p for p in parsed if self._is_relevant(p, terms)]
            if relevant:
                return relevant[:limit]
        return []

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise SourceError(self.name, f"arXiv Atom 解析失败：{exc}") from exc

        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _NS):
            try:
                paper = self._parse_entry(entry)
            except Exception as exc:
                logger.debug("arXiv 记录解析跳过：%s", exc)
                continue
            if paper and paper.title:
                papers.append(paper)
        return papers

    def _parse_entry(self, entry: ET.Element) -> Paper | None:
        raw_id = _clean(entry.findtext("atom:id", default="", namespaces=_NS))
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        # 去掉版本号（2401.12345v2 → 2401.12345）
        versionless = re.sub(r"v\d+$", "", arxiv_id)

        title = _clean(entry.findtext("atom:title", default="", namespaces=_NS))
        summary = _clean(entry.findtext("atom:summary", default="", namespaces=_NS))
        published = _clean(entry.findtext("atom:published", default="", namespaces=_NS))
        pub_year = None
        match = re.match(r"(\d{4})", published)
        if match:
            pub_year = int(match.group(1))

        authors = [
            to_family_first(_clean(node.findtext("atom:name", default="", namespaces=_NS)))
            for node in entry.findall("atom:author", _NS)
        ]
        authors = [a for a in authors if a]

        categories = [
            node.get("term", "")
            for node in entry.findall("atom:category", _NS)
            if node.get("term")
        ]

        doi = normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=_NS))
        journal_ref = _clean(entry.findtext("arxiv:journal_ref", default="", namespaces=_NS))

        return Paper(
            title=title,
            abstract=summary,
            authors=authors,
            journal=journal_ref or "arXiv preprint",
            pub_year=pub_year,
            source=self.source_id,
            source_id=versionless or None,
            doi=doi,
            keywords=categories,
            is_open_access=True,
            full_text_url=f"https://arxiv.org/pdf/{versionless}" if versionless else "",
            url=f"https://arxiv.org/abs/{versionless}" if versionless else raw_id,
            publication_type="preprint",
            language="en",
        )

    async def fulltext(self, paper: Paper) -> str:
        """arXiv 全文为 PDF，交由 Reader Agent 的 PDF 解析器处理。"""
        return ""

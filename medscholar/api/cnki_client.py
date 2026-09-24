"""CNKI 公开检索客户端（默认关闭，仅元数据）。

2026-02 复核结论：search.cnki.com.cn 结果页已改为纯前端 JS 渲染，
服务端 HTML（约 39 KB）不含文献条目；数据接口返回 HTTP 403。
不引入浏览器渲染就无法稳定获取元数据。

本模块默认 ``enabled: false``，被调用时抛 :class:`SourceError`
并给出替代方案，不静默返回空结果。

合规边界：

* 只访问公开检索页，不登录、不绕过付费墙、不破解验证码；
* 只提取标题、作者、期刊、年份、摘要、关键词等元数据；
* 不下载、不缓存、不再分发任何全文；全文请通过机构合法授权获取。

如需重新启用：

1. 配置 ``sources.cnki.base_url`` 指向你自己部署的渲染服务；
2. 使用机构订阅的 CNKI 官方接口/镜像（需自行取得授权）；
3. 直接用替代通路（推荐）：
   * ``OpenAlexClient`` 的 ``language:zh`` 过滤
   * ``PubMedClient`` 的 ``chinese[la]`` 过滤
   * ``CrossrefClient`` —— 注册了 DOI 的中文期刊
"""

from __future__ import annotations

import html
import logging
import re

from ..models import Paper
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["CnkiClient", "CNKI_STATUS"]

#: 供 UI / 文档复用的状态说明
CNKI_STATUS = (
    "CNKI 公开检索页已改为 JS 渲染，服务端 HTML 不含文献条目，"
    "数据接口返回 403；已改用 OpenAlex(language:zh)+PubMed(chinese[la])+Crossref 作为中文文献通路。"
)

_SEARCH_URL = "https://search.cnki.com.cn/Search/Result"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(19|20)\d{2}")
# 只在页面确实含文献条目链接时才认为抓取成功
_ARTICLE_LINK_RE = re.compile(
    r"(kns\.cnki\.net|/KCMS/detail|dbcode=|filename=)", re.IGNORECASE
)
_ITEM_SPLIT_RE = re.compile(
    r'<div[^>]*class="[^"]*(?:list-item|result-item|itemm)[^"]*"[^>]*>', re.IGNORECASE
)
_TITLE_RE = re.compile(
    r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE | re.DOTALL
)
_SUMMARY_RE = re.compile(
    r'<p[^>]*class="[^"]*(?:summary|abstract)[^"]*"[^>]*>(?P<body>.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)
# 「作者：…」在 CNKI 结果页里是独立的 info 段落，必须单独取；
# 早期实现把 info 也算进摘要，导致摘要被写成「作者：张三;李四;」（实测踩到）。
_INFO_RE = re.compile(
    r'<p[^>]*class="[^"]*info[^"]*"[^>]*>(?P<body>.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)
_SOURCE_RE = re.compile(
    r'<p[^>]*class="[^"]*(?:source|journal)[^"]*"[^>]*>(?P<body>.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)
# 作者/关键词限定在分号分隔的短片段内，否则会把后面的摘要整段吞掉。
_AUTHOR_RE = re.compile(r"作者[：:]\s*(?P<body>[^;；<\n]{1,60}(?:[;；][^;；<\n]{1,60})*)")
_KEYWORD_RE = re.compile(r"关键词[：:]\s*(?P<body>[^;；<\n]{1,60}(?:[;；][^;；<\n]{1,60})*)")


def _strip_tags(value: str | None) -> str:
    if not value:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()


class CnkiClient(BaseClient):
    name = "cnki"
    label = "CNKI"
    source_id = "cnki"
    base_url = _SEARCH_URL

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        query = (query or "").strip()
        if not query:
            return []
        if not _has_cjk(query):
            logger.debug("CNKI 跳过纯英文查询：%s", query)
            return []

        referer = "https://search.cnki.com.cn/Search/Result"
        page = await self.request(
            "GET",
            self.base_url,
            params={"content": query, "type": "0", "order": "1", "page": "1"},
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Referer": referer,
            },
            expect="text",
        )

        if not _ARTICLE_LINK_RE.search(page):
            raise SourceError(self.name, CNKI_STATUS)

        papers = self._parse_results(page, limit)
        if not papers:
            raise SourceError(self.name, CNKI_STATUS)

        if filters:
            papers = _apply_filters(papers, filters)
        return papers[:limit]

    # ---------------------------------------------------------------- 解析
    def _parse_results(self, page: str, limit: int) -> list[Paper]:
        blocks = _ITEM_SPLIT_RE.split(page)
        if len(blocks) <= 1:
            return []

        papers: list[Paper] = []
        seen: set[str] = set()
        for block in blocks[1:]:
            paper = self._parse_block(block)
            if paper is None or not paper.title:
                continue
            if paper.title in seen:
                continue
            seen.add(paper.title)
            papers.append(paper)
            if len(papers) >= limit * 2:
                break
        return papers

    def _parse_block(self, block: str) -> Paper | None:
        title_match = _TITLE_RE.search(block)
        if not title_match:
            return None
        title = _strip_tags(title_match.group("title"))
        if not title or len(title) < 4:
            return None

        href = html.unescape(title_match.group("href") or "")
        if href.startswith("/"):
            href = "https://search.cnki.com.cn" + href
        if not _ARTICLE_LINK_RE.search(href):
            return None  # 过滤导航链接

        summary_match = _SUMMARY_RE.search(block)
        abstract = _strip_tags(summary_match.group("body")) if summary_match else ""
        source_match = _SOURCE_RE.search(block)
        source_text = _strip_tags(source_match.group("body")) if source_match else ""
        info_match = _INFO_RE.search(block)
        info_text = _strip_tags(info_match.group("body")) if info_match else ""
        flat = _strip_tags(block)

        authors: list[str] = []
        author_match = _AUTHOR_RE.search(info_text) or _AUTHOR_RE.search(flat)
        if author_match:
            authors = [
                a.strip()
                for a in re.split(r"[,，;；]", author_match.group("body"))
                if a.strip() and a.strip() not in {"暂无", "无"}
            ]

        keywords: list[str] = []
        kw_match = _KEYWORD_RE.search(info_text) or _KEYWORD_RE.search(flat)
        if kw_match:
            keywords = [
                k.strip() for k in re.split(r"[,，;；]", kw_match.group("body")) if k.strip()
            ]

        pub_year = None
        year_match = _YEAR_RE.search(source_text) or _YEAR_RE.search(flat[:400])
        if year_match:
            pub_year = int(year_match.group())

        journal = re.sub(r"[-–—]?\s*(19|20)\d{2}.*$", "", source_text).strip(" -–—")

        return Paper(
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_year=pub_year,
            source=self.source_id,
            source_id=href or None,
            keywords=keywords,
            url=href,
            language="zh",
            publication_type="journal-article",
        )

    async def fulltext(self, paper: Paper) -> str:
        """按合规要求，CNKI 不提供全文抓取。"""
        return ""


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _apply_filters(papers: list[Paper], filters: SearchFilters) -> list[Paper]:
    out = []
    for paper in papers:
        if filters.year_from and paper.pub_year and paper.pub_year < filters.year_from:
            continue
        if filters.year_to and paper.pub_year and paper.pub_year > filters.year_to:
            continue
        out.append(paper)
    return out

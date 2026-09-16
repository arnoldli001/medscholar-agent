"""PubMed E-utilities 客户端（NCBI）。

覆盖 ESearch / EFetch / ELink 三个端点：

* ESearch —— 关键词、MeSH、日期范围检索，返回 PMID 列表
* EFetch  —— 按 PMID 批量取回完整 XML（标题、摘要、作者、MeSH、DOI、PMCID）
* ELink   —— 参考文献 / 相似文献

速率：无 Key 3 次/秒，注册免费 Key 后 10 次/秒（配置 ``sources.pubmed.api_key``）。
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from ..models import Paper
from ..query import for_source
from ..textutil import normalize_doi
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["PubMedClient"]

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MAX_EFETCH_BATCH = 200
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)


def _text(node: ET.Element | None) -> str:
    """取元素及其所有子节点的文本（标题里常嵌 ``<i>`` 等标签）。"""
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


class PubMedClient(BaseClient):
    name = "pubmed"
    label = "PubMed"
    source_id = "pubmed"
    base_url = _BASE

    def __init__(self, settings=None, *, config=None, client=None) -> None:
        super().__init__(settings, config=config, client=client)
        if self.settings.api_key:
            self.bucket.update_rps(max(self.settings.rps, 10.0))

    # ------------------------------------------------------------- 公共参数
    def _common_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"tool": "medscholar-agent"}
        if self.settings.email:
            params["email"] = self.settings.email
        if self.settings.api_key:
            params["api_key"] = self.settings.api_key
        return params

    # ---------------------------------------------------------------- 检索
    @staticmethod
    def _build_term(query: str, filters: SearchFilters | None) -> str:
        """把统一过滤条件翻译成 PubMed 检索语法。

        年份过滤交由 ESearch 的 ``mindate`` / ``maxdate`` 参数处理（更可靠），
        这里只补充 PMC 与出版类型等无法用参数表达的约束。
        """
        term = (query or "").strip()
        if not filters:
            return term
        if filters.open_access_only:
            term += " AND (pmc[filter] OR free full text[sb])"
        for ptype in filters.publication_types:
            term += f' AND "{ptype}"[pt]'
        if filters.language:
            term += f" AND {filters.language}[la]"
        return term.strip()

    async def esearch(
        self,
        term: str,
        *,
        retmax: int = 50,
        sort: str = "relevance",
        filters: SearchFilters | None = None,
    ) -> list[str]:
        """执行 ESearch，返回 PMID 列表。"""
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": min(retmax, 10_000),
            "retmode": "json",
            "sort": {"date": "pub_date", "relevance": "relevance"}.get(sort, "relevance"),
            **self._common_params(),
        }
        if filters:
            if filters.year_from:
                params["mindate"] = filters.year_from
                params["datetype"] = "pdat"
            if filters.year_to:
                params["maxdate"] = filters.year_to
                params["datetype"] = "pdat"

        data = await self.request("GET", f"{_BASE}/esearch.fcgi", params=params)
        try:
            return [str(x) for x in data["esearchresult"]["idlist"]]
        except (KeyError, TypeError) as exc:
            raise SourceError(self.name, f"ESearch 响应结构异常：{exc}") from exc

    async def efetch(self, pmids: Iterable[str]) -> list[Paper]:
        """按 PMID 批量取回并解析完整记录。"""
        ids = [str(p) for p in pmids if str(p).strip()]
        if not ids:
            return []
        papers: list[Paper] = []
        for start in range(0, len(ids), _MAX_EFETCH_BATCH):
            batch = ids[start : start + _MAX_EFETCH_BATCH]
            xml_text = await self.request(
                "GET",
                f"{_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "xml",
                    "rettype": "abstract",
                    **self._common_params(),
                },
                expect="text",
            )
            papers.extend(self._parse_articles(xml_text))
        return papers

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        # 先按 PubMed 的布尔语法翻译用户输入（支持 AND / OR / NOT / 短语）
        term = self._build_term(for_source(query, self.name), filters)
        if not term:
            return []
        pmids = await self.esearch(term, retmax=limit, sort=filters.sort if filters else "relevance")
        if not pmids:
            return []
        return (await self.efetch(pmids))[:limit]

    # ------------------------------------------------------------ XML 解析
    def _parse_articles(self, xml_text: str) -> list[Paper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise SourceError(self.name, f"EFetch XML 解析失败：{exc}") from exc

        papers: list[Paper] = []
        for article in root.iter("PubmedArticle"):
            try:
                paper = self._parse_one(article)
            except Exception as exc:  # 单条记录异常不应中断整批
                logger.debug("PubMed 记录解析跳过：%s", exc)
                continue
            if paper and paper.title:
                papers.append(paper)
        return papers

    def _parse_one(self, article: ET.Element) -> Paper | None:
        citation = article.find("MedlineCitation")
        if citation is None:
            return None
        art = citation.find("Article")
        if art is None:
            return None

        pmid = _text(citation.find("PMID")) or None
        title = _text(art.find("ArticleTitle"))

        # --- 摘要：多段 AbstractText，带 Label 时拼成结构化小标题
        abstract_parts: list[str] = []
        abstract_node = art.find("Abstract")
        if abstract_node is not None:
            for chunk in abstract_node.findall("AbstractText"):
                body = _text(chunk)
                if not body:
                    continue
                label = chunk.get("Label")
                abstract_parts.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(abstract_parts)

        # --- 作者
        authors: list[str] = []
        for author in art.findall("AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            last = _text(author.find("LastName"))
            fore = _text(author.find("ForeName")) or _text(author.find("Initials"))
            if last:
                authors.append(f"{last} {fore}".strip())

        # --- 期刊
        journal_node = art.find("Journal")
        journal = _text(journal_node.find("Title")) if journal_node is not None else ""
        if not journal and journal_node is not None:
            journal = _text(journal_node.find("ISOAbbreviation"))

        # --- 年份：优先 JournalIssue/PubDate，其次 ArticleDate，最后 MedlineDate
        pub_year = None
        if journal_node is not None:
            date_node = journal_node.find("JournalIssue/PubDate")
            if date_node is not None:
                pub_year = _year_from(date_node)
        if pub_year is None:
            pub_year = _year_from(art.find("ArticleDate"))
        if pub_year is None and journal_node is not None:
            medline = journal_node.find("JournalIssue/PubDate/MedlineDate")
            if medline is not None and medline.text:
                match = re.search(r"(19|20)\d{2}", medline.text)
                pub_year = int(match.group()) if match else None

        # --- 卷期页
        volume = issue = pages = ""
        if journal_node is not None:
            issue_node = journal_node.find("JournalIssue")
            if issue_node is not None:
                volume = _text(issue_node.find("Volume"))
                issue = _text(issue_node.find("Issue"))
        pages = _text(art.find("Pagination/MedlinePgn"))

        # --- DOI / PMCID
        doi = None
        for eloc in art.findall("ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.text:
                doi = normalize_doi(eloc.text)
                break
        pmcid = None
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            id_type = (aid.get("IdType") or "").lower()
            if id_type == "doi" and not doi and aid.text:
                doi = normalize_doi(aid.text)
            elif id_type in {"pmc", "pmcid"} and aid.text:
                value = aid.text.strip()
                pmcid = value if _PMCID_RE.match(value) else f"PMC{value}"

        # --- MeSH 主题词（描述符 + 限定词）
        mesh: list[str] = []
        for heading in citation.findall("MeshHeadingList/MeshHeading"):
            descriptor = _text(heading.find("DescriptorName"))
            if not descriptor:
                continue
            quals = [_text(q) for q in heading.findall("QualifierName")]
            quals = [q for q in quals if q]
            mesh.append(f"{descriptor} / {'; '.join(quals)}" if quals else descriptor)

        # --- 关键词
        keywords: list[str] = []
        for kw_list in citation.findall("KeywordList"):
            for kw in kw_list.findall("Keyword"):
                text = _text(kw)
                if text:
                    keywords.append(text)

        # --- 出版类型 / 语言
        pub_types = [
            _text(pt) for pt in art.findall("PublicationTypeList/PublicationType")
        ]
        pub_types = [p for p in pub_types if p]
        language = _text(art.find("Language"))

        is_oa = bool(pmcid)  # PMID 有对应 PMCID 即可在 PMC 免费获取

        return Paper(
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_year=pub_year,
            source=self.source_id,
            source_id=pmid,
            pmid=pmid,
            pmcid=pmcid,
            doi=doi,
            mesh_terms=mesh,
            keywords=keywords,
            is_open_access=is_oa,
            full_text_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            volume=volume,
            issue=issue,
            pages=pages,
            publication_type="; ".join(pub_types),
            language=language,
        )

    # ------------------------------------------------------------ 引用关系
    async def references(self, paper: Paper) -> list[dict[str, Any]]:
        """通过 ELink 取参考文献（PubMed 只给 PMID 列表，标题需另取）。"""
        if not paper.pmid:
            return []
        return await self._elink(paper.pmid, "pubmed_pubmed_refs")

    async def cited_by(self, paper: Paper) -> list[dict[str, Any]]:
        """PubMed 无原生"被引"接口，用 ELink 的 citedin 近似。"""
        if not paper.pmid:
            return []
        return await self._elink(paper.pmid, "pubmed_pubmed_citedin")

    async def similar(self, paper: Paper) -> list[dict[str, Any]]:
        if not paper.pmid:
            return []
        return await self._elink(paper.pmid, "pubmed_pubmed")

    async def _elink(self, pmid: str, linkname: str) -> list[dict[str, Any]]:
        data = await self.request(
            "GET",
            f"{_BASE}/elink.fcgi",
            params={
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "linkname": linkname,
                "retmode": "json",
                **self._common_params(),
            },
        )
        out: list[dict[str, Any]] = []
        try:
            for linkset in data.get("linksets", []):
                for db_links in linkset.get("linksetdbs", []):
                    if db_links.get("dbto") != "pubmed":
                        continue
                    for target in db_links.get("links", []):
                        out.append({"pmid": str(target), "source": "pubmed"})
        except (AttributeError, TypeError) as exc:  # pragma: no cover
            logger.debug("ELink 解析异常：%s", exc)
        return out

    # ---------------------------------------------------------------- 全文
    async def fulltext(self, paper: Paper) -> str:
        """仅对 PMC 开放获取子集可用（``db=pmc``）。"""
        if not paper.pmcid:
            return ""
        try:
            xml_text = await self.request(
                "GET",
                f"{_BASE}/efetch.fcgi",
                params={
                    "db": "pmc",
                    "id": paper.pmcid,
                    "retmode": "xml",
                    **self._common_params(),
                },
                expect="text",
            )
        except SourceError as exc:
            logger.debug("PMC 全文获取失败 %s：%s", paper.pmcid, exc)
            return ""
        return _strip_article_xml(xml_text)


def _year_from(node: ET.Element | None) -> int | None:
    if node is None:
        return None
    year = _text(node.find("Year"))
    if year:
        match = re.search(r"(19|20)\d{2}", year)
        if match:
            return int(match.group())
    text = _text(node)
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group()) if match else None


def _strip_article_xml(xml_text: str) -> str:
    """把 PMC 的 JATS XML 压成纯文本（保留段落换行）。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    body = root.find(".//body")
    if body is None:
        return ""
    chunks: list[str] = []
    for element in body.iter():
        if element.tag in {"title", "p", "td", "th", "label", "caption"}:
            text = _text(element)
            if text:
                chunks.append(text)
    return "\n".join(chunks)

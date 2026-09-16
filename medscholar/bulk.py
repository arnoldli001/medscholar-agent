"""官方批量数据包导入（PubMed baseline / PMC Open Access Subset）。

这是"合法爬取"的正确形态：这些数据是 NLM / Europe PMC **主动发布**给
批量使用的，README 明确允许下载、文本挖掘与再分发（PMC OA 子集）。

* **PubMed baseline**：每年一次的 37M+ 条题录 XML（解压后约 40 GB）
  https://pubmed.ncbi.nlm.nih.gov/download/
* **PMC Open Access Subset**：FTP 上的全文包（.tar.gz，含 .nxml/PDF）
  https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/

工程要点（不这么做就会 OOM 或跑不完）：

* 用 ``iterparse`` **流式**解析，处理完一个 ``<PubmedArticle>`` 立刻
  ``elem.clear()`` —— 一次性 ``parse()`` 一个 1 GB 的 XML 必然爆内存；
* PMC 的 tar.gz 用 ``r|gz`` **流式**模式读（不能 seek，因此顺序处理）；
* 必须能按关键词/年份**过滤**：全量 3700 万条不可能都入库，
  默认只收命中主题的那些。
"""

from __future__ import annotations

import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import AppConfig, get_config
from .db.connect import Database, get_db
from .models import Paper, coerce_int
from .textutil import clean_abstract, normalize_doi

logger = logging.getLogger(__name__)

__all__ = [
    "BulkReport",
    "parse_pubmed_article",
    "parse_jats",
    "iter_pubmed_xml",
    "iter_pmc_oa_tar",
    "import_pubmed_baseline",
    "import_pmc_oa",
]

_TAG_RE = re.compile(r"<[^>]{1,80}>")


def _text(elem: ET.Element | None) -> str:
    """取元素内全部文本（含子元素，如 <i>、<sup>），并清理 JATS 标签。"""
    if elem is None:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail)
    return _TAG_RE.sub(" ", "".join(parts))


@dataclass(slots=True)
class BulkReport:
    """一次批量导入的结果。"""

    seen: int = 0          # 扫描过的记录数
    matched: int = 0       # 命中过滤条件的记录数
    created: int = 0
    merged: int = 0
    failed: int = 0
    with_fulltext: int = 0
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "matched": self.matched,
            "created": self.created,
            "merged": self.merged,
            "failed": self.failed,
            "with_fulltext": self.with_fulltext,
            "files": list(self.files),
            "errors": list(self.errors[:10]),
            "sample_titles": list(self.sample_titles[:10]),
            "dry_run": self.dry_run,
        }

    def summary(self) -> str:
        parts = [f"扫描 {self.seen} 条 → 命中 {self.matched} 条"]
        if self.dry_run:
            parts.append("（演练模式，未写库）")
        else:
            parts.append(f"新增 {self.created} 篇")
            if self.merged:
                parts.append(f"合并 {self.merged} 篇")
            if self.with_fulltext:
                parts.append(f"含全文 {self.with_fulltext} 篇")
            if self.failed:
                parts.append(f"失败 {self.failed} 篇")
        return "，".join(parts) + "。"


# ====================================================== PubMed baseline XML
def parse_pubmed_article(elem: ET.Element) -> Paper | None:
    """把一个 ``<PubmedArticle>`` 元素转成 :class:`Paper`（无全文）。"""
    citation = elem.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID")).strip()
    article = citation.find("Article")
    if article is None:
        return None

    title = _text(article.find("ArticleTitle")).strip()
    if not title:
        return None

    # 摘要可能是多段，带 Label（BACKGROUND / METHODS ...）
    abstract_parts: list[str] = []
    abstract = article.find("Abstract")
    if abstract is not None:
        for chunk in abstract.findall("AbstractText"):
            label = (chunk.get("Label") or "").strip()
            body = _text(chunk).strip()
            if body:
                abstract_parts.append(f"{label}: {body}" if label else body)
    abstract_text = clean_abstract("\n".join(abstract_parts))

    journal = _text(article.find("Journal/Title")).strip() or _text(
        article.find("Journal/ISOAbbreviation")
    ).strip()

    pub_date = article.find("Journal/JournalIssue/PubDate")
    year = coerce_int(_text(pub_date.find("Year")) if pub_date is not None else "")
    medline_date = _text(pub_date.find("MedlineDate")) if pub_date is not None else ""
    if year is None and medline_date:
        year = coerce_int(medline_date)

    authors: list[str] = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for author in author_list.findall("Author"):
            collective = _text(author.find("CollectiveName")).strip()
            if collective:
                authors.append(collective)
                continue
            last = _text(author.find("LastName")).strip()
            fore = _text(author.find("ForeName")).strip()
            if last and fore:
                authors.append(f"{last}, {fore}")
            elif last:
                authors.append(last)

    # DOI 可能在 ELocationID 或 PubmedData/ArticleIdList
    doi = ""
    for elocation in article.findall("ELocationID"):
        if (elocation.get("EIdType") or "").lower() == "doi":
            doi = _text(elocation).strip()
            break
    pmcid = ""
    pubmed_data = elem.find("PubmedData")
    if pubmed_data is not None:
        for article_id in pubmed_data.findall("ArticleIdList/ArticleId"):
            kind = (article_id.get("IdType") or "").lower()
            value = _text(article_id).strip()
            if kind == "doi" and not doi:
                doi = value
            elif kind == "pmc":
                pmcid = value

    mesh: list[str] = []
    for heading in citation.findall("MeshHeadingList/MeshHeading/DescriptorName"):
        term = _text(heading).strip()
        if term:
            mesh.append(term)

    keywords: list[str] = []
    for keyword in citation.findall("KeywordList/Keyword"):
        term = _text(keyword).strip()
        if term:
            keywords.append(term)

    types = [
        _text(pt).strip()
        for pt in article.findall("PublicationTypeList/PublicationType")
        if _text(pt).strip()
    ]

    pages = _text(article.find("Pagination/MedlinePgn")).strip()
    language = ""
    langs = [_text(la).strip() for la in article.findall("Language")]
    if langs:
        language = langs[0]

    return Paper(
        title=title,
        source="bulk",
        abstract=abstract_text,
        authors=authors,
        journal=journal,
        pub_year=year,
        doi=normalize_doi(doi) if doi else None,
        pmid=pmid or None,
        pmcid=pmcid or None,
        mesh_terms=mesh,
        keywords=keywords,
        volume=_text(article.find("Journal/JournalIssue/Volume")).strip(),
        issue=_text(article.find("Journal/JournalIssue/Issue")).strip(),
        pages=pages,
        publication_type=types[0] if types else "",
        language=language,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        source_id=pmid,
    )


def iter_pubmed_xml(path: str | Path) -> Iterator[Paper]:
    """流式产出 XML 文件里的文献（内存占用与文件大小无关）。

    支持两种情况：整个文件是 ``<PubmedArticleSet>``，或者文件里是
    多个顶层 ``<PubmedArticle>``（NLM 的分卷文件就是这样）。
    """
    file_path = Path(path)
    if file_path.suffix.lower() == ".gz":
        import gzip

        with gzip.open(file_path, "rb") as handle:
            yield from _iter_pubmed_stream(handle)
        return
    with file_path.open("rb") as handle:
        yield from _iter_pubmed_stream(handle)


def _iter_pubmed_stream(handle: Any) -> Iterator[Paper]:
    try:
        for event, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag == "PubmedArticle":
                paper = parse_pubmed_article(elem)
                if paper is not None:
                    yield paper
                elem.clear()  # 关键：立刻释放，否则内存随文件线性增长
    except ET.ParseError as exc:
        # 分卷文件末尾偶尔有截断，不该让整批导入失败
        logger.warning("XML 解析中断（可能是文件截断）：%s", exc)


# ==================================================== PMC Open Access Subset
def parse_jats(nxml: bytes | str) -> tuple[Paper, str] | None:
    """解析 JATS（``.nxml``）全文，返回 ``(Paper, 全文)``。

    PMC OA 子集里每篇是 ``<pmcid>.nxml``，既有题录也有正文，是全文的主力来源。
    """
    try:
        root = ET.fromstring(nxml)
    except ET.ParseError as exc:
        logger.debug("JATS 解析失败：%s", exc)
        return None

    front = root.find(".//front")
    if front is None:
        return None

    title = _text(front.find(".//article-title")).strip()
    if not title:
        title = _text(front.find(".//title-group/article-title")).strip()
    if not title:
        return None

    journal = _text(front.find(".//journal-title")).strip()
    if not journal:
        journal = _text(front.find(".//journal-meta//abbrev-journal-title")).strip()

    year = (
        coerce_int(_text(front.find(".//pub-date/year")))
        or coerce_int(_text(front.find(".//pub-date//year")))
        or coerce_int(_text(front.find(".//pub-date//string-date")))
    )

    authors: list[str] = []
    for contrib in front.findall(".//contrib"):
        if (contrib.get("contrib-type") or "") not in {"author", ""}:
            continue
        surname = _text(contrib.find(".//surname")).strip()
        given = _text(contrib.find(".//given-names")).strip()
        collective = _text(contrib.find(".//collab")).strip()
        if collective:
            authors.append(collective)
        elif surname and given:
            authors.append(f"{surname}, {given}")
        elif surname:
            authors.append(surname)

    abstract = clean_abstract(_text(front.find(".//abstract")))

    doi = ""
    for article_id in front.findall(".//article-id"):
        if (article_id.get("pub-id-type") or "").lower() == "doi":
            doi = _text(article_id).strip()
            break

    pmcid = ""
    for article_id in front.findall(".//article-id"):
        if (article_id.get("pub-id-type") or "").lower() in {"pmc", "pmcid"}:
            pmcid = _text(article_id).strip()
            break

    # 正文：body 里的段落，按章节标题组织
    body_parts: list[str] = []
    body = root.find(".//body")
    if body is not None:
        for sec in body.iter("sec"):
            sec_title = _text(sec.find("title")).strip()
            if sec_title:
                body_parts.append(f"\n## {sec_title}")
            for para in sec.findall("p"):
                text = _text(para).strip()
                if text:
                    body_parts.append(text)
        if not body_parts:
            for para in body.findall(".//p"):
                text = _text(para).strip()
                if text:
                    body_parts.append(text)

    fulltext = "\n\n".join(body_parts).strip()
    if not fulltext:
        fulltext = abstract

    paper = Paper(
        title=title,
        source="bulk",
        abstract=abstract,
        authors=authors,
        journal=journal,
        pub_year=year,
        doi=normalize_doi(doi) if doi else None,
        pmcid=pmcid or None,
        is_open_access=True,
        url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
        source_id=pmcid,
    )
    return paper, fulltext


def iter_pmc_oa_tar(path: str | Path) -> Iterator[tuple[Paper, str]]:
    """流式读取 PMC OA 的 ``.tar.gz``，产出 ``(Paper, 全文)``。"""
    file_path = Path(path)
    try:
        with tarfile.open(file_path, "r|gz") as tar:
            for member in tar:
                name = member.name
                if not member.isfile():
                    continue
                if not name.lower().endswith((".nxml", ".xml")):
                    continue  # PDF 正文另行处理；这里只吃 JATS
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                parsed = parse_jats(handle.read())
                if parsed is not None:
                    yield parsed
    except tarfile.TarError as exc:
        logger.warning("tar 读取失败 %s：%s", file_path, exc)
    except (EOFError, OSError) as exc:  # pragma: no cover - 截断/占用
        logger.warning("tar 读取中断 %s：%s", file_path, exc)


# ============================================================ 过滤与入库
def _matches(
    paper: Paper,
    *,
    terms: Sequence[str],
    year_from: int | None,
    year_to: int | None,
    require_abstract: bool,
) -> bool:
    if year_from and paper.pub_year and paper.pub_year < year_from:
        return False
    if year_to and paper.pub_year and paper.pub_year > year_to:
        return False
    if require_abstract and not paper.abstract.strip():
        return False
    if not terms:
        return True
    haystack = " ".join(
        [paper.title, paper.abstract, " ".join(paper.mesh_terms), " ".join(paper.keywords)]
    ).lower()
    return all(term.lower() in haystack for term in terms)


def parse_terms(raw: str | Sequence[str] | None) -> list[str]:
    """把 ``"rTMS,depression"`` 或列表解析成过滤词。"""
    if not raw:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    terms: list[str] = []
    for item in items:
        for part in re.split(r"[,;，；]", str(item)):
            text = part.strip()
            if text:
                terms.append(text)
    return terms


async def _store_papers(
    papers: Iterable[Paper],
    *,
    report: BulkReport,
    db: Database,
    embed: bool,
    fulltexts: dict[str, str] | None = None,
) -> None:
    from .db.repo import insert_paper, save_fulltext
    from .dedupe import merge_papers

    batch = merge_papers(list(papers))
    new_ids: list[int] = []
    for paper in batch:
        try:
            paper_id, created = insert_paper(paper, db=db)
        except Exception as exc:
            report.failed += 1
            report.errors.append(f"{paper.title[:50]}：{exc}")
            continue
        paper.paper_id = paper_id
        if created:
            report.created += 1
            new_ids.append(paper_id)
        else:
            report.merged += 1
        if fulltexts:
            text = fulltexts.get(paper.source_id or "")
            if text:
                try:
                    save_fulltext(
                        paper_id, text, origin="pmc-oa", db=db
                    )
                    report.with_fulltext += 1
                except Exception as exc:  # pragma: no cover
                    logger.debug("全文入库失败 %s：%s", paper_id, exc)

    if embed and new_ids:
        try:
            from .embedding.pipeline import run_embedding_pipeline_async

            await run_embedding_pipeline_async(ids=new_ids, limit=len(new_ids), db=db)
        except Exception as exc:
            report.errors.append(f"向量生成失败（文献已入库）：{exc}")


async def import_pubmed_baseline(
    path: str | Path,
    *,
    terms: str | Sequence[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
    require_abstract: bool = True,
    dry_run: bool = False,
    embed: bool = True,
    config: AppConfig | None = None,
    db: Database | None = None,
) -> BulkReport:
    """导入 PubMed baseline XML（文件或目录）。

    默认要求有摘要：baseline 里大量记录只有题名（会议摘要、勘误、社论），
    入库后对综述没用，反而稀释检索结果。
    """
    import asyncio

    database = db or get_db()
    report = BulkReport(dry_run=dry_run)
    filter_terms = parse_terms(terms)

    paths = _collect_files(path, (".xml", ".xml.gz"))
    if not paths:
        report.errors.append(f"没有找到 PubMed XML 文件：{path}")
        return report

    buffer: list[Paper] = []
    for file_path in paths:
        report.files.append(file_path.name)
        for paper in iter_pubmed_xml(file_path):
            report.seen += 1
            if not _matches(
                paper,
                terms=filter_terms,
                year_from=year_from,
                year_to=year_to,
                require_abstract=require_abstract,
            ):
                continue
            report.matched += 1
            if limit and report.matched > limit:
                report.matched -= 1
                break
            if len(report.sample_titles) < 10:
                report.sample_titles.append(paper.title)
            if dry_run:
                continue
            buffer.append(paper)
            # 攒够一批就落库，避免 37M 条全堆在内存里
            if len(buffer) >= 500:
                await _store_papers(buffer, report=report, db=database, embed=False)
                buffer = []
        if limit and report.matched >= limit:
            break

    if buffer and not dry_run:
        await _store_papers(buffer, report=report, db=database, embed=False)
    if not dry_run and embed and report.created:
        try:
            from .embedding.pipeline import run_embedding_pipeline_async

            await run_embedding_pipeline_async(
                limit=report.created, db=database, config=config or get_config()
            )
        except Exception as exc:
            report.errors.append(f"向量生成失败（文献已入库）：{exc}")
    _ = asyncio  # 保持导入形式一致；本函数内部已 await
    return report


async def import_pmc_oa(
    path: str | Path,
    *,
    terms: str | Sequence[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    embed: bool = True,
    config: AppConfig | None = None,
    db: Database | None = None,
) -> BulkReport:
    """导入 PMC Open Access Subset 的 ``.tar.gz``（含 JATS 全文）。"""
    database = db or get_db()
    report = BulkReport(dry_run=dry_run)
    filter_terms = parse_terms(terms)

    paths = _collect_files(path, (".tar.gz", ".tgz"))
    if not paths:
        report.errors.append(f"没有找到 PMC OA 的 .tar.gz 文件：{path}")
        return report

    buffer: list[Paper] = []
    fulltexts: dict[str, str] = {}
    for file_path in paths:
        report.files.append(file_path.name)
        for paper, fulltext in iter_pmc_oa_tar(file_path):
            report.seen += 1
            if not _matches(
                paper,
                terms=filter_terms,
                year_from=year_from,
                year_to=year_to,
                require_abstract=False,
            ):
                continue
            report.matched += 1
            if limit and report.matched > limit:
                report.matched -= 1
                break
            if len(report.sample_titles) < 10:
                report.sample_titles.append(paper.title)
            if dry_run:
                continue
            buffer.append(paper)
            if fulltext and paper.source_id:
                fulltexts[paper.source_id] = fulltext
            if len(buffer) >= 200:
                await _store_papers(
                    buffer, report=report, db=database, embed=False, fulltexts=fulltexts
                )
                buffer, fulltexts = [], {}
        if limit and report.matched >= limit:
            break

    if buffer and not dry_run:
        await _store_papers(
            buffer, report=report, db=database, embed=False, fulltexts=fulltexts
        )
    if not dry_run and embed and report.created:
        try:
            from .embedding.pipeline import run_embedding_pipeline_async

            await run_embedding_pipeline_async(
                limit=report.created, db=database, config=config or get_config()
            )
        except Exception as exc:
            report.errors.append(f"向量生成失败（文献已入库）：{exc}")
    return report


def _collect_files(path: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    """把"文件或目录"统一成待处理文件列表。"""
    target = Path(path)
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(target.rglob("*")):
        if not child.is_file():
            continue
        name = child.name.lower()
        if any(name.endswith(suffix) for suffix in suffixes):
            found.append(child)
    return found

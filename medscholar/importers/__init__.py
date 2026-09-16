"""题录导入：解析 → 去重 → 入库 → 补齐向量 → 返回报告。

与"检索"共用同一条落库与嵌入管线，所以导入进来的文献**立刻**能被
本地检索（FTS + 向量）命中，也能被综述引用。

合规说明：本模块只处理**用户自己导出/提供的题录文件**，
不联网抓取任何数据库，也不涉及任何账号认证。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import AppConfig, get_config
from ..db.connect import Database, get_db
from ..db.repo import insert_paper
from ..dedupe import merge_papers
from ..models import Paper
from .parsers import SUPPORTED_FORMATS, detect_format, parse_any

logger = logging.getLogger(__name__)

__all__ = [
    "ImportReport",
    "import_text",
    "import_file",
    "import_paths",
    "import_from_zotero",
]


@dataclass(slots=True)
class ImportReport:
    """一次导入的结果。"""

    parsed: int = 0
    unique: int = 0
    created: int = 0
    merged: int = 0
    failed: int = 0
    embedded: dict[str, Any] = field(default_factory=dict)
    format: str = ""
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "unique": self.unique,
            "created": self.created,
            "merged": self.merged,
            "failed": self.failed,
            "embedded": dict(self.embedded),
            "format": self.format,
            "files": list(self.files),
            "errors": list(self.errors[:10]),
            "sample_titles": list(self.sample_titles[:10]),
        }

    def summary(self) -> str:
        parts = [f"解析 {self.parsed} 条 → 去重后 {self.unique} 条"]
        parts.append(f"新增 {self.created} 篇")
        if self.merged:
            parts.append(f"合并/更新 {self.merged} 篇")
        if self.failed:
            parts.append(f"失败 {self.failed} 篇")
        embedded = self.embedded.get("embedded")
        if embedded:
            parts.append(f"生成向量 {embedded} 条")
        return "，".join(parts) + "。"


async def import_text(
    text: str,
    *,
    filename: str = "",
    source: str = "import",
    embed: bool = True,
    dry_run: bool = False,
    config: AppConfig | None = None,
    db: Database | None = None,
) -> ImportReport:
    """导入一段题录文本。"""
    report = ImportReport(files=[filename] if filename else [])
    papers, fmt = parse_any(text, filename=filename, default_source=source)
    report.format = fmt
    report.parsed = len(papers)

    if not papers:
        if not fmt:
            report.errors.append(
                "无法识别文件格式。支持：" + "、".join(SUPPORTED_FORMATS)
                + "（RIS / BibTeX / EndNote 标记文本 / CSV）。"
                "Web of Science、Scopus、Embase、CNKI、万方 的导出文件都可以。"
            )
        else:
            report.errors.append(
                f"识别为 {fmt} 格式，但没有解析出任何条目 —— 请确认文件内容完整。"
            )
        return report

    # 跨条目去重（同一篇可能被多条记录命中）
    unique = merge_papers(papers)
    report.unique = len(unique)
    report.sample_titles = [p.title for p in unique[:10]]

    if dry_run:
        return report

    await _store(unique, report=report, embed=embed, config=config, db=db)
    return report


async def _store(
    papers: Sequence[Paper],
    *,
    report: ImportReport,
    embed: bool,
    config: AppConfig | None,
    db: Database | None,
) -> None:
    database = db or get_db()
    cfg = config or get_config()
    new_ids: list[int] = []

    for paper in papers:
        try:
            paper_id, created = insert_paper(paper, db=database)
        except Exception as exc:  # 单条失败不应中断整批导入
            report.failed += 1
            report.errors.append(f"{paper.title[:60]}：{exc}")
            logger.debug("导入失败：%s", exc)
            continue
        paper.paper_id = paper_id
        if created:
            report.created += 1
            new_ids.append(paper_id)
        else:
            report.merged += 1

    if embed and new_ids:
        try:
            from ..embedding.pipeline import run_embedding_pipeline_async

            result = await run_embedding_pipeline_async(
                ids=new_ids, limit=len(new_ids), config=cfg, db=database
            )
            report.embedded = (
                result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            )
        except Exception as exc:
            report.errors.append(f"向量生成失败（文献已入库，可稍后手动补齐）：{exc}")
            logger.warning("导入后嵌入失败：%s", exc)


def import_file(
    path: str | Path,
    *,
    source: str = "import",
    config: AppConfig | None = None,
    db: Database | None = None,
) -> tuple[str, ImportReport]:
    """读取并解析一个文件（同步，便于 CLI 与测试使用）。"""
    import asyncio

    file_path = Path(path)
    text = _read_text(file_path)
    report = asyncio.run(
        import_text(
            text, filename=file_path.name, source=source, embed=False, config=config, db=db
        )
    )
    return text, report


def import_paths(
    paths: Sequence[str | Path],
    *,
    source: str = "import",
    embed: bool = True,
    dry_run: bool = False,
    config: AppConfig | None = None,
    db: Database | None = None,
) -> ImportReport:
    """批量导入多个文件（CLI 入口）。"""
    import asyncio

    async def _run() -> ImportReport:
        total = ImportReport()
        for path in paths:
            file_path = Path(path)
            if not file_path.is_file():
                total.errors.append(f"文件不存在：{file_path}")
                continue
            try:
                text = _read_text(file_path)
            except Exception as exc:
                total.errors.append(f"读取失败 {file_path.name}：{exc}")
                continue
            report = await import_text(
                text,
                filename=file_path.name,
                source=source,
                embed=embed,
                dry_run=dry_run,
                config=config,
                db=db,
            )
            total.parsed += report.parsed
            total.unique += report.unique
            total.created += report.created
            total.merged += report.merged
            total.failed += report.failed
            total.files.extend(report.files)
            total.errors.extend(report.errors)
            total.sample_titles.extend(report.sample_titles)
            if report.format:
                total.format = report.format if not total.format else total.format
            if report.embedded:
                total.embedded = report.embedded
        total.sample_titles = total.sample_titles[:10]
        return total

    return asyncio.run(_run())


def _read_text(path: Path) -> str:
    """读题录文件：优先 UTF-8，失败则回退 GBK（中文数据库导出常见）。

    用 ``errors="replace"`` 而不是直接报错：老导出的编码很杂，
    宁可个别字符变 ``?``，也不该让整份文件导入失败。
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def detect_only(text: str, filename: str = "") -> str:
    """只做格式识别（供 API 预检用）。"""
    return detect_format(text, filename)


# ============================================================ Zotero 桥接
async def import_from_zotero(
    data_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    embed: bool = True,
    index_pdfs: bool = True,
    pdf_limit: int = 200,
    config: AppConfig | None = None,
    db: Database | None = None,
) -> ImportReport:
    """把本机 Zotero 库导入 MedScholar，并可选索引本地 PDF 全文。

    这是订阅资源**合规**的落地方式：Zotero 里那些 PDF 是用户自己合法取得的，
    MedScholar 只读本机文件、只做本地索引，不下载也不认证。
    """
    from ..zotero import read_zotero_library, to_papers

    report = ImportReport(format="zotero")
    items = read_zotero_library(data_dir, limit=limit)
    papers = to_papers(items)
    report.parsed = len(papers)
    if not papers:
        report.errors.append(
            "Zotero 库里没有可导入的文献条目（已跳过附件、笔记和已删除条目）。"
        )
        return report

    unique = merge_papers(papers)
    report.unique = len(unique)
    report.sample_titles = [p.title for p in unique[:10]]
    await _store(unique, report=report, embed=embed, config=config, db=db)

    if index_pdfs:
        indexed, failed = await _index_local_pdfs(
            items, config=config, db=db, limit=pdf_limit
        )
        if indexed:
            report.embedded = {**report.embedded, "pdf_indexed": indexed}
        if failed:
            report.errors.append(f"{failed} 个本地 PDF 解析失败（其余已入库）")
    return report


async def _index_local_pdfs(
    items: Sequence[Any],
    *,
    config: AppConfig | None,
    db: Database | None,
    limit: int,
) -> tuple[int, int]:
    """把 Zotero 里的本地 PDF 解析成全文并入库（建立全文索引）。

    只处理**本地已存在**的文件；读不到就跳过，绝不联网补齐。
    """
    from ..agent.reader import extract_pdf_text
    from ..db.repo import find_paper_id, save_fulltext

    database = db or get_db()
    indexed = 0
    failed = 0

    for item in items:
        if indexed >= limit:
            break
        if not item.pdf_paths:
            continue
        source_file = Path(item.pdf_paths[0])
        if not source_file.is_file():
            continue
        paper_id = find_paper_id(
            doi=(item.fields.get("doi") or "").strip() or None,
            title=item.title,
            db=database,
        )
        if not paper_id:
            continue
        try:
            data = await asyncio.to_thread(source_file.read_bytes)
            text = await asyncio.to_thread(extract_pdf_text, data)
        except Exception as exc:
            logger.debug("解析本地 PDF 失败 %s：%s", source_file, exc)
            failed += 1
            continue
        if not text or not text.strip():
            failed += 1
            continue
        try:
            await asyncio.to_thread(
                save_fulltext,
                paper_id,
                text,
                origin=f"zotero:{source_file.name}",
                db=database,
            )
            indexed += 1
        except Exception as exc:
            logger.debug("本地全文入库失败 %s：%s", paper_id, exc)
            failed += 1
    return indexed, failed

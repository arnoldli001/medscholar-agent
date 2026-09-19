"""Reader Agent —— 全文解析与精读。

职责（需求 3.1）：

* 获取**开放获取**文献全文（Europe PMC JATS → PubMed PMC → OA PDF），
  非开放获取文献只保留元数据与出版商链接，绝不绕过付费墙；
* 把全文解析成纯文本并落入 ``paper_fulltext``（同时建立 FTS5 索引）；
* 生成单篇速读笔记（研究设计 / 对象 / 干预 / 结局 / 局限）。

PDF 解析依赖可选的 PyMuPDF；未安装时给出明确的安装指引而不是静默失败。

**分层说明**：PDF 解析、落地页 PDF 定位、失败归类这些**基础设施**能力已搬到
:mod:`medscholar.importers.pdf`，本模块从那里导入并继续对外重导出。
原因见该模块的 docstring —— 简单说：Zotero 附件导入（infrastructure）也要读 PDF，
把它们留在应用层会让基础设施反向依赖上层。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from ..config import AppConfig, get_config
from ..db.connect import Database, get_db
from ..db.repo import get_fulltext, save_fulltext
from ..importers.pdf import (
    PDF_AVAILABLE,
    PDF_NOTE,
    _MAX_PDF_BYTES,
    classify_fulltext_error,
    extract_pdf_text,
    extract_pdf_url,
)
from ..models import Paper

logger = logging.getLogger(__name__)

__all__ = [
    "ReaderAgent",
    "FullTextResult",
    "extract_pdf_text",
    "extract_pdf_url",
    "classify_fulltext_error",
    "PDF_AVAILABLE",
]

#: 失败原因归类、PDF 后端探测等实现已搬到 ``medscholar/importers/pdf.py``：
#: 它们是基础设施关注点，且被导入器复用。此处的名字由上面的 import 重导出，
#: 既有调用方（含测试与脚本）不受影响。


@dataclass(slots=True)
class FullTextResult:
    """全文获取结果。"""

    paper_id: int
    content: str = ""
    origin: str = ""
    source_url: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.content.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "char_count": len(self.content),
            "origin": self.origin,
            "source_url": self.source_url,
            "ok": self.ok,
            "error": self.error,
        }


class ReaderAgent:
    """阅读智能体。"""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        db: Database | None = None,
        registry: Any = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db or get_db()
        self._clients: dict[str, Any] = {}
        self._registry = registry

    # ---------------------------------------------------------------- 客户端
    async def _client(self, name: str):
        if name not in self._clients:
            from ..api import EuropePMCClient, PubMedClient
            from ..api.unpaywall_client import UnpaywallClient

            cls = {
                "europepmc": EuropePMCClient,
                "pubmed": PubMedClient,
                "unpaywall": UnpaywallClient,
            }[name]
            client = cls(config=self.config)
            await client.start()
            self._clients[name] = client
        return self._clients[name]

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:  # pragma: no cover
                pass
        self._clients.clear()

    # ---------------------------------------------------------------- 全文
    async def fetch_fulltext(self, paper: Paper, *, persist: bool = True) -> FullTextResult:
        """按「Europe PMC → PubMed PMC → OA PDF」顺序取全文。

        失败时返回的 ``error`` 必须说明**真正试过什么、卡在哪一步**。
        早期实现无论哪一步失败都回落到同一句"该文献非开放获取"，
        对有 PMCID 的文献是**误导性**的 —— 会让人以为该文献本就不该有全文。
        这里逐级记录尝试轨迹。
        """
        paper_id = paper.paper_id or 0

        if persist and paper_id:
            cached = get_fulltext(paper_id, db=self.db)
            if cached:
                return FullTextResult(paper_id, cached, origin="cache")

        tried: list[str] = []
        last_oa_error = ""

        # 1) Europe PMC（覆盖 800 万+ OA 全文，带 JATS 正文）
        if paper.pmcid or paper.is_open_access:
            tried.append("Europe PMC")
            try:
                client = await self._client("europepmc")
                text = await client.fulltext(paper)
            except Exception as exc:
                logger.debug("Europe PMC 全文失败：%s", exc)
                text = ""
            if text:
                return self._store(paper, text, "europepmc", self._epmc_url(paper), persist)

        # 2) PubMed 的 PMC 子集
        if paper.pmcid:
            tried.append("PubMed PMC")
            try:
                client = await self._client("pubmed")
                text = await client.fulltext(paper)
            except Exception as exc:
                logger.debug("PubMed PMC 全文失败：%s", exc)
                text = ""
            if text:
                return self._store(
                    paper, text, "pubmed-pmc", f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper.pmcid}/", persist
                )

        # 3) Unpaywall：按 DOI 找**合法**的开放获取副本。
        #    很多文献只有 DOI（既无 PMCID 也没标 OA），这条是它们唯一的希望；
        #    Unpaywall 只返回 OA 链接，不涉及任何认证绕过。
        if paper.doi:
            tried.append("Unpaywall")
            try:
                client = await self._client("unpaywall")
                location = await client.best_oa_location(paper.doi)
            except Exception as exc:
                logger.debug("Unpaywall 查询失败（%s）：%s", paper.doi, exc)
                location = None
            if location is not None and location.preferred_url:
                url = location.preferred_url
                logger.info(
                    "Unpaywall 找到 OA 副本：%s（%s / %s）",
                    url,
                    location.host_type or "未知来源",
                    location.version or "版本未知",
                )
                oa_paper = replace(
                    paper,
                    full_text_url=url,
                    is_open_access=True,
                    note=(paper.note or "").strip(),
                )
                result = await self._fetch_pdf(oa_paper, persist=persist)
                if result.ok:
                    return result
                last_oa_error = result.error or ""

        # 4) 开放获取 PDF（仅有链接且明确 OA 时才下载）
        if paper.full_text_url and paper.is_open_access:
            tried.append("OA PDF")
            result = await self._fetch_pdf(paper, persist=persist)
            if result.ok:
                return result
            return FullTextResult(
                paper_id,
                content="",
                error=f"{result.error or 'PDF 全文获取失败'}（已尝试：{'、'.join(tried)}）",
            )

        # 到这里说明四种路径都没走通，按实际情况给出准确原因
        if last_oa_error:
            reason = f"Unpaywall 找到了开放获取链接，但下载失败：{last_oa_error}"
        elif paper.pmcid:
            reason = (
                f"PMC（{paper.pmcid}）没有提供 JATS 正文，且没有其他可下载的 OA 链接。"
                "常见于会议摘要、勘误、社论等本身就没有正文全文的条目。"
            )
        elif paper.is_open_access:
            reason = "标记为开放获取，但既没有 PMCID 也没有可下载的 OA 全文链接。"
        else:
            reason = "该文献非开放获取，仅保留元数据与出版商链接（不绕过付费墙）。"
        return FullTextResult(
            paper_id, content="", error=f"{reason}（已尝试：{'、'.join(tried) or '无可用路径'}）"
        )

    async def _download(self, client: httpx.AsyncClient, url: str) -> tuple[bytes, str, str]:
        """下载一个 URL。返回 ``(数据, content_type, 错误)``；错误非空表示失败。"""
        try:
            response = await client.get(url)
        except Exception as exc:
            return b"", "", f"PDF 下载失败：{type(exc).__name__}: {exc}"

        if response.status_code in {401, 403}:
            return b"", "", (
                f"出版商拒绝了自动下载（HTTP {response.status_code}）。"
                "该链接是面向人工浏览的页面，程序不会去绕过其访问控制；"
                "如需该文全文，请点击文献卡片上的链接手动获取。"
            )
        if response.status_code == 405:
            return b"", "", "该链接不支持直接下载（HTTP 405），可能是落地页而非 PDF"
        if response.status_code >= 400:
            return b"", "", f"PDF 下载失败：HTTP {response.status_code}"

        return response.content, (response.headers.get("content-type") or "").lower(), ""

    async def _fetch_pdf(self, paper: Paper, *, persist: bool) -> FullTextResult:
        """下载并解析开放获取 PDF。

        两个要点：

        1. **出版商普遍拒绝自动化下载**（实测 ScienceDirect 403、部分机构仓储 405），
           这是对方的风控策略，不是本程序的缺陷，也不应该去绕过；
        2. **数据源给的链接常常是文章网页而不是 PDF**（实测占失败原因的 27%）。
           遇到 HTML 时再解析一层：用出版商为学术搜索声明的标准
           ``citation_pdf_url`` 元数据定位真正的 PDF 地址后重试。
           —— 实测能显著提高成功率。
        """
        url = paper.full_text_url
        paper_id = paper.paper_id or 0
        if not url.lower().startswith(("http://", "https://")):
            return FullTextResult(paper_id, error=f"非法全文链接：{url}")
        if not PDF_AVAILABLE:
            return FullTextResult(paper_id, error=PDF_NOTE)

        data = b""
        content_type = ""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=15.0),
            follow_redirects=True,
            headers={
                # 部分站点对 Accept 也做校验；用通配更稳
                "Accept": "application/pdf, text/html, application/octet-stream, */*",
                "User-Agent": "MedScholarAgent/1.0 (academic research; contact via config)",
            },
        ) as client:
            data, content_type, error = await self._download(client, url)
            if error:
                return FullTextResult(paper_id, error=error)

            # 不是 PDF 而是网页 → 尝试从页面里找出真正的 PDF 地址
            if not data[:5].startswith(b"%PDF") and ("html" in content_type or data[:200].lstrip()[:1] == b"<"):
                try:
                    html = data.decode("utf-8", "replace")
                except Exception:  # pragma: no cover
                    html = ""
                pdf_url = extract_pdf_url(html, url)
                if pdf_url and pdf_url != url:
                    logger.debug("从落地页解析到 PDF 地址：%s", pdf_url)
                    data2, content_type2, error2 = await self._download(client, pdf_url)
                    if not error2 and data2[:5].startswith(b"%PDF"):
                        data, content_type, url = data2, content_type2, pdf_url

        if len(data) > _MAX_PDF_BYTES:
            return FullTextResult(paper_id, error="PDF 体积过大，已跳过")
        if not data[:5].startswith(b"%PDF"):
            hint = "链接指向网页而非 PDF（页面里也没有找到 citation_pdf_url 声明）" \
                if "html" in content_type else "返回内容不是 PDF"
            return FullTextResult(paper_id, error=hint)

        try:
            text = extract_pdf_text(data)
        except Exception as exc:
            return FullTextResult(paper_id, error=f"PDF 解析失败：{exc}")

        if not text:
            return FullTextResult(paper_id, error="PDF 无可提取文本（可能是扫描件）")

        if persist and paper.paper_id:
            self._save_pdf_file(paper, data)
        return self._store(paper, text, "pdf", url, persist)

    def _save_pdf_file(self, paper: Paper, data: bytes) -> Path | None:
        try:
            target = self.config.fulltext_dir / f"{paper.paper_id}.pdf"
            target.write_bytes(data)
            return target
        except OSError as exc:  # pragma: no cover
            logger.debug("PDF 落盘失败：%s", exc)
            return None

    @staticmethod
    def _epmc_url(paper: Paper) -> str:
        if paper.pmcid:
            return f"https://europepmc.org/article/PMC/{paper.pmcid}"
        if paper.pmid:
            return f"https://europepmc.org/article/MED/{paper.pmid}"
        return paper.url

    def _store(
        self,
        paper: Paper,
        text: str,
        origin: str,
        source_url: str,
        persist: bool,
    ) -> FullTextResult:
        if persist and paper.paper_id:
            try:
                save_fulltext(
                    paper.paper_id,
                    text,
                    origin=origin,
                    source_url=source_url,
                    db=self.db,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("全文入库失败 %s：%s", paper.paper_id, exc)
        return FullTextResult(paper.paper_id or 0, text, origin=origin, source_url=source_url)

    # ------------------------------------------------------------ 批量抓取
    async def warm_fulltext(
        self,
        papers: list[Paper],
        *,
        limit: int = 5,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> int:
        """为前若干篇开放获取文献预取全文（供 Writer 深度引用）。"""
        fetched = 0
        candidates = [p for p in papers if p.paper_id and (p.pmcid or p.is_open_access)][:limit]
        for paper in candidates:
            result = await self.fetch_fulltext(paper)
            if result.ok:
                fetched += 1
                if emit:
                    await emit(
                        "status",
                        {
                            "message": f"已获取全文：{paper.title[:48]}（{len(result.content)} 字）"
                        },
                    )
        return fetched

    # ---------------------------------------------------------------- 速读
    async def summarize(
        self,
        paper: Paper,
        *,
        topic: str = "",
        focus: str = "",
        llm: Any = None,
    ) -> str:
        """生成单篇文献速读笔记（基于摘要 + 已入库全文）。"""
        from ..llm.client import get_llm
        from ..llm.prompts import SUMMARY_SYSTEM, summary_user

        body = paper.abstract or ""
        if paper.paper_id:
            full = get_fulltext(paper.paper_id, db=self.db)
            if full:
                body = full[:12000]

        if not body.strip():
            return "该文献没有可用的摘要或全文，无法生成速读笔记。"

        client = llm or get_llm(self.config)
        await client.start()
        text = await client.chat(
            [{"role": "user", "content": summary_user(body, topic=topic, focus=focus)}],
            system=SUMMARY_SYSTEM,
            temperature=0.2,
        )
        return text.strip()

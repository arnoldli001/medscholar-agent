"""PDF 解析基础设施（PyMuPDF 适配 + 落地页 PDF 定位 + 失败归类）。

这些函数原在 ``agent/reader.py``，属于基础设施关注点（调用第三方 PDF 库、
解析 HTML 元信息、把错误字符串归类），不是智能体的决策逻辑。
原来 ``importers/__init__.py``（Zotero 附件导入需要读 PDF）不得不在函数体里
``from ..agent.reader import extract_pdf_text``，形成了一条
基础设施反向依赖应用层的隐藏耦合（懒加载 import，人工 review 很难发现）。
这条违规是 ``scripts/check_arch.py`` 跑出来的。

搬到 ``importers/`` 后依赖方向就正了：application → infrastructure。
放在这里而不是新开一个包：导入器与 Reader 都需要它，而它本身没有业务语义，
属于"把外部格式转成文本"这一类适配器。
"""

from __future__ import annotations

import re
from html import unescape as html_unescape
from urllib.parse import urljoin

__all__ = [
    "PDF_AVAILABLE",
    "PDF_NOTE",
    "classify_fulltext_error",
    "extract_pdf_text",
    "extract_pdf_url",
]

# ---------------------------------------------------------------------------
# 失败原因归类
# ---------------------------------------------------------------------------
#: (正则, 可读标签, 是否永久)
#:
#: "永久" = 再试也不会成功（文献本身没正文、出版商长期拒绝、非开放获取）；
#: "可重试" = 网络抖动、对方 5xx，下次值得再试。
#: 顺序敏感：更具体的规则放前面。
_ERROR_RULES: tuple[tuple[str, str, bool], ...] = (
    (r"没有提供 JATS 正文", "文献本身没有正文（会议摘要 / 勘误 / 社论等）", True),
    (r"出版商拒绝了自动下载|HTTP 403", "出版商风控拒绝自动下载（403）", True),
    (r"不支持直接下载|HTTP 405", "链接是落地页而非 PDF（405）", True),
    (r"链接指向网页而非 PDF", "链接指向网页而非 PDF（未找到 citation_pdf_url）", True),
    (r"返回内容不是 PDF", "返回内容不是 PDF", True),
    (r"PDF 无可提取文本", "PDF 是扫描件，无法提取文本", True),
    (r"PDF 解析失败", "PDF 解析失败（文件可能损坏）", True),
    (r"PDF 体积过大", "PDF 体积过大，已跳过", True),
    (r"非开放获取", "非开放获取（合规跳过）", True),
    (r"既没有 PMCID", "标称 OA 但缺 PMCID 与下载链接", True),
    (r"非法全文链接", "全文链接格式非法", True),
    (r"HTTP 5\d\d|服务端错误", "对方服务端错误（5xx，可重试）", False),
    (r"网络错误|ReadTimeout|ConnectError|Timeout|timed out", "网络超时/连接失败（可重试）", False),
    (r"PDF 下载失败", "PDF 下载失败（其他）", False),
)


def classify_fulltext_error(message: str | None) -> tuple[str, bool]:
    """把全文抓取的错误文本归类。

    Returns:
        ``(可读标签, 是否永久失败)``。永久失败的下次会被跳过，不再白跑。
    """
    text = message or ""
    for pattern, label, permanent in _ERROR_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return label, permanent
    return "其他原因", False


# ---------------------------------------------------------------------------
# PDF 后端探测
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_RE = re.compile(r"\n{3,}")
_MAX_PDF_BYTES = 40 * 1024 * 1024


def _pdf_available() -> tuple[bool, str]:
    """探测 PDF 解析后端。

    PyMuPDF 1.24+ 推荐 ``import pymupdf``；旧的 ``import fitz`` 已标记为弃用
    （实测 1.28.2 会打印 DeprecationWarning）。这里优先用新名字，旧版回退到 ``fitz``。
    """
    try:
        import pymupdf  # type: ignore

        return True, f"PyMuPDF {getattr(pymupdf, '__version__', '')}".strip()
    except ImportError:
        pass
    try:
        import fitz  # type: ignore

        return True, f"PyMuPDF(fitz) {getattr(fitz, '__version__', '')}".strip()
    except ImportError:
        return False, (
            "未安装 PyMuPDF，无法解析 PDF 全文。安装命令：\n"
            "    .python\\python.exe -m pip install pymupdf"
        )


def _import_pymupdf():
    """返回已加载的 PyMuPDF 模块（优先新包名）。"""
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except ImportError:
        import fitz  # type: ignore

        return fitz


PDF_AVAILABLE, PDF_NOTE = _pdf_available()


def _normalize(text: str) -> str:
    text = _WS_RE.sub(" ", text or "")
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RE.sub("\n\n", text).strip()


def extract_pdf_text(data: bytes, *, max_pages: int = 80) -> str:
    """从 PDF 字节流提取文本（仅取前若干页，医学论文正文通常足够）。"""
    if not PDF_AVAILABLE:
        raise RuntimeError(PDF_NOTE)
    pymupdf = _import_pymupdf()

    chunks: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            if page_index >= max_pages:
                break
            chunks.append(page.get_text("text"))
    return _normalize("\n".join(chunks))


# ---------------------------------------------------------------------------
# 从落地页定位真正的 PDF
# ---------------------------------------------------------------------------

_PDF_META_RE = re.compile(
    r"<meta[^>]+(?:name|property)\s*=\s*[\"']citation_pdf_url[\"'][^>]*>",
    re.IGNORECASE,
)
_CONTENT_ATTR_RE = re.compile(r"content\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
#: 兜底：页面里明显的 ".pdf" 链接
_PDF_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+\.pdf(?:\?[^"']*)?)["']""", re.IGNORECASE)


def extract_pdf_url(html: str, base_url: str) -> str:
    """从文章落地页里找出真正的 PDF 地址。

    很多数据源（OpenAlex / Crossref）给出的 ``full_text_url`` 其实是文章网页
    而不是 PDF，直接下载网页当然拿不到——实测这一条占失败原因的 27%。

    出版商为了被学术搜索收录，普遍会在页面里声明标准的
    ``<meta name="citation_pdf_url">``（Google Scholar 规范）。用它定位 PDF
    是公开、正当的做法，不需要绕过任何访问控制。

    >>> extract_pdf_url('<meta name="citation_pdf_url" content="/a.pdf">', 'https://x.org/p')
    'https://x.org/a.pdf'
    """
    if not html:
        return ""
    match = _PDF_META_RE.search(html)
    if match:
        content = _CONTENT_ATTR_RE.search(match.group(0))
        if content:
            return urljoin(base_url, html_unescape(content.group(1).strip()))
    # 兜底：找页面上第一个 .pdf 链接
    href = _PDF_HREF_RE.search(html)
    if href:
        return urljoin(base_url, html_unescape(href.group(1).strip()))
    return ""

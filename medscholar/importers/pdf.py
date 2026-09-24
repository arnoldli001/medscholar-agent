"""PDF 基础设施：PyMuPDF 适配、落地页 PDF 定位、失败归类（外部格式→文本的适配器，
无业务语义）。放 importers/ 是为消除 importers 懒加载 agent.reader 造成的
基础设施反向依赖应用层的耦合（scripts/check_arch.py 检出）。
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
#: (正则, 可读标签, 是否永久)。永久=重试也不会成功（无正文/403/非OA）；
#: 可重试=网络抖动/5xx。顺序敏感，更具体的规则放前面。
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
    """探测 PyMuPDF 后端：1.24+ 优先 import pymupdf，旧版回退 import fitz（fitz 已弃用）。"""
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
    """从落地页找真正 PDF 地址：很多数据源给的 full_text_url 是网页而非 PDF（实测占
    失败原因 27%），优先用页面公开声明的 <meta name="citation_pdf_url">（Google Scholar
    规范，正当手段不绕访问控制），再兜底找首个 .pdf 链接。

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

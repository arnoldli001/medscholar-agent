"""导出层：参考文献、综述草稿、文献表格与知识库备份。

所有导出都写成 UTF-8（含 BOM 的 CSV 便于 Excel 直接打开），
并统一落到 ``<data_dir>/exports/`` 下。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..cite import detect_style, format_records, format_reference_list
from ..config import AppConfig, get_config
from ..models import Paper

logger = logging.getLogger(__name__)

__all__ = [
    "EXPORT_FORMATS",
    "export_references",
    "export_csv",
    "export_json",
    "export_markdown",
    "export_docx_compatible_html",
    "export_bundle",
    "write_export",
    "safe_filename",
]

#: 导出格式 → (扩展名, 说明)
EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "bibtex": (".bib", "BibTeX（LaTeX / Zotero / EndNote）"),
    "ris": (".ris", "RIS（EndNote / NoteExpress）"),
    "apa7": (".txt", "APA 第 7 版参考文献表"),
    "vancouver": (".txt", "Vancouver 参考文献表"),
    "gb7714": (".txt", "GB/T 7714-2015 参考文献表"),
    "csv": (".csv", "文献表格（Excel 可直接打开）"),
    "json": (".json", "完整结构化数据"),
    "markdown": (".md", "综述草稿（Markdown）"),
    "html": (".html", "综述草稿（HTML，可粘贴进 Word）"),
}

_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str, *, fallback: str = "export", max_len: int = 80) -> str:
    """把任意标题转成安全的文件名。"""
    cleaned = _UNSAFE_RE.sub("_", str(name or "")).strip(" ._")
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


# ------------------------------------------------------------------ 参考文献
def export_references(papers: Sequence[Paper], style: str = "gb7714") -> str:
    """导出参考文献表（BibTeX / RIS 返回条目序列）。"""
    style = detect_style(style)
    return format_records(papers, style)


# ---------------------------------------------------------------------- 表格
_CSV_COLUMNS: tuple[tuple[str, str], ...] = (
    ("paper_id", "编号"),
    ("title", "标题"),
    ("authors", "作者"),
    ("journal", "期刊"),
    ("pub_year", "年份"),
    ("cited_by_count", "被引"),
    ("doi", "DOI"),
    ("pmid", "PMID"),
    ("source", "来源"),
    ("is_open_access", "开放获取"),
    ("keywords", "关键词"),
    ("abstract", "摘要"),
)


def export_csv(papers: Sequence[Paper]) -> str:
    """导出为 CSV（UTF-8 BOM，Excel 双击不乱码）。"""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([label for _, label in _CSV_COLUMNS])
    for paper in papers:
        row = []
        for key, _ in _CSV_COLUMNS:
            value = getattr(paper, key, "")
            if isinstance(value, list):
                value = "; ".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = "是" if value else "否"
            elif value is None:
                value = ""
            row.append(str(value).replace("\n", " "))
        writer.writerow(row)
    return "\ufeff" + buffer.getvalue()


def export_json(papers: Sequence[Paper], *, indent: int = 2) -> str:
    """导出为结构化 JSON。"""
    return json.dumps([p.to_dict() for p in papers], ensure_ascii=False, indent=indent)


# ------------------------------------------------------------------ 综述草稿
def export_markdown(
    *,
    title: str,
    content: str,
    references: Sequence[Paper] = (),
    style: str = "gb7714",
    meta: Mapping[str, Any] | None = None,
) -> str:
    """把综述草稿组装成完整的 Markdown 文档（正文 + 参考文献表）。"""
    lines = [f"# {title}", ""]
    info = dict(meta or {})
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    footer_bits = [f"生成时间：{stamp}", "生成工具：MedScholar Agent"]
    if info.get("topic"):
        footer_bits.insert(0, f"课题：{info['topic']}")
    if info.get("paper_count"):
        footer_bits.insert(1, f"文献数：{info['paper_count']}")
    lines.append("> " + " ｜ ".join(footer_bits))
    lines.append("")
    lines.append(content.strip())
    lines.append("")
    if references:
        lines.append("## 参考文献")
        lines.append("")
        lines.append(format_reference_list(references, style))
        lines.append("")
    return "\n".join(lines)


def export_docx_compatible_html(
    *,
    title: str,
    content: str,
    references: Sequence[Paper] = (),
    style: str = "gb7714",
) -> str:
    """生成可直接粘贴进 Word 的 HTML（保留标题层级与参考文献表）。"""
    body_html = _markdown_to_html(content)
    refs_html = ""
    if references:
        items = "".join(
            f"<p style='margin:0 0 6pt 0;text-indent:-21pt;padding-left:21pt;'>{_escape(line)}</p>"
            for line in format_reference_list(references, style).splitlines()
            if line.strip()
        )
        refs_html = f"<h2>参考文献</h2>{items}"
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_escape(title)}</title></head>"
        "<body style=\"font-family:'Times New Roman','SimSun',serif;font-size:12pt;"
        'line-height:1.6;max-width:800px;margin:0 auto;">'
        f"<h1 style='text-align:center;'>{_escape(title)}</h1>"
        f"{body_html}{refs_html}</body></html>"
    )


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _markdown_to_html(text: str) -> str:
    """极简 Markdown → HTML（只处理本项目会生成的语法）。"""
    html_parts: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("### "):
            html_parts.append(f"<h3>{_escape(line[4:])}</h3>")
        elif line.startswith("## "):
            html_parts.append(f"<h2>{_escape(line[3:])}</h2>")
        elif line.startswith("# "):
            html_parts.append(f"<h1>{_escape(line[2:])}</h1>")
        elif line.startswith(("- ", "* ")):
            html_parts.append(f"<p style='margin-left:24pt;'>• {_escape(line[2:])}</p>")
        else:
            html_parts.append(f"<p style='text-indent:24pt;'>{_escape(line)}</p>")
    return "".join(html_parts)


# ---------------------------------------------------------------------- 落盘
def write_export(
    content: str,
    *,
    name: str,
    fmt: str,
    config: AppConfig | None = None,
    directory: Path | None = None,
) -> Path:
    """把导出内容写入磁盘，返回文件路径。"""
    cfg = config or get_config()
    target_dir = directory or cfg.export_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    fmt_key = str(fmt).lower().lstrip(".")
    suffix = EXPORT_FORMATS.get(fmt_key, (f".{fmt_key or 'txt'}", ""))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_filename(name)}_{stamp}{suffix}"
    path = target_dir / filename
    path.write_text(content, encoding="utf-8")
    logger.info("已导出 %s（%d 字符）", path, len(content))
    return path


def export_bundle(
    papers: Sequence[Paper],
    *,
    name: str,
    styles: Iterable[str] = ("bibtex", "ris", "gb7714"),
    config: AppConfig | None = None,
) -> dict[str, str]:
    """一次性导出多种参考文献格式，返回 ``{格式: 文件路径}``。"""
    out: dict[str, str] = {}
    for style in styles:
        key = detect_style(style)
        content = export_references(papers, key)
        if not content.strip():
            continue
        if key == "csv":
            content = export_csv(papers)
        path = write_export(content, name=f"{name}_{key}", fmt=key, config=config)
        out[key] = str(path)
    return out

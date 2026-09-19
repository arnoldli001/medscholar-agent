"""引用格式化与导出。"""

from __future__ import annotations

from .styles import (
    STYLE_LABELS,
    STYLES,
    citation_key,
    detect_style,
    format_authors,
    format_citation,
    format_inline,
    format_records,
    format_reference_list,
    split_author,
    to_bibtex,
    to_ris,
)

__all__ = [
    "STYLES",
    "STYLE_LABELS",
    "detect_style",
    "split_author",
    "format_authors",
    "format_citation",
    "format_inline",
    "format_reference_list",
    "format_records",
    "citation_key",
    "to_bibtex",
    "to_ris",
]

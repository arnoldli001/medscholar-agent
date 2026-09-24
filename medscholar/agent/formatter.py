"""Formatter Agent：引用格式化与成稿校验（需求 3.1）。

把正文里的数字引用标记按目标格式渲染（数字制 ``[1]``，作者-年份制
``(Zhang, 2023)``）；生成参考文献表（APA 7th / Vancouver / GB-T 7714 /
BibTeX / RIS）；校验引用完整性——正文引用了但参考文献表没有的编号、
参考文献表里有但正文从未引用的条目；并落盘导出多种格式。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..cite import (
    STYLE_LABELS,
    detect_style,
    format_citation,
    format_inline,
    format_reference_list,
)
from ..config import AppConfig, get_config
from ..export.exporters import export_bundle, export_markdown, write_export
from ..models import Paper
from .writer import extract_citations

logger = logging.getLogger(__name__)

__all__ = ["FormatterAgent", "CitationReport", "FORMATTER_STYLES"]

#: 数字制样式（正文使用 [n]）
NUMERIC_STYLES = {"vancouver", "gb7714"}
FORMATTER_STYLES = tuple(STYLE_LABELS)

_CITATION_GROUP_RE = re.compile(r"[\[【]\s*\d{1,3}(?:\s*[,，\-–]\s*\d{1,3})*\s*[\]】]")


@dataclass(slots=True)
class CitationReport:
    """引用一致性检查结果。"""

    cited: list[int] = field(default_factory=list)
    missing_from_list: list[int] = field(default_factory=list)  # 正文引了但列表没有
    never_cited: list[int] = field(default_factory=list)        # 列表有但正文没引
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cited": sorted(set(self.cited)),
            "missing_from_list": sorted(set(self.missing_from_list)),
            "never_cited": sorted(set(self.never_cited)),
            "ok": self.ok,
        }

    def summary(self) -> str:
        if self.ok:
            return f"引用校验通过：正文引用 {len(set(self.cited))} 篇，全部可对应参考文献表。"
        bits = []
        if self.missing_from_list:
            bits.append(f"正文引用了不存在的编号 {sorted(set(self.missing_from_list))}")
        if self.never_cited:
            bits.append(f"参考文献表中有 {len(set(self.never_cited))} 条未被正文引用")
        return "引用校验发现问题：" + "；".join(bits)


class FormatterAgent:
    """格式智能体。"""

    def __init__(self, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    # ------------------------------------------------------------ 引用校验
    @staticmethod
    def validate_citations(
        draft: str, valid_ids: Sequence[int], *, draft_citations: Sequence[int] | None = None
    ) -> CitationReport:
        """检查正文引用编号与参考文献表的对应关系。"""
        valid = set(valid_ids)
        cited = list(draft_citations) if draft_citations is not None else extract_citations(draft)
        missing = [n for n in cited if n not in valid]
        never = [n for n in valid if n not in set(cited)]
        return CitationReport(
            cited=cited,
            missing_from_list=missing,
            never_cited=never,
            ok=not missing,
        )

    # ---------------------------------------------------------- 正文样式化
    def restyle_inline(
        self,
        draft: str,
        entries: Sequence[tuple[int, Paper]],
        style: str = "gb7714",
    ) -> str:
        """把 ``[n]`` 形式的引用标记渲染成目标样式的文内引用。

        数字制样式（Vancouver / GB-T 7714）保持 ``[n]``；
        作者-年份制样式（APA / Chicago）改写为 ``(Zhang, 2023)``。
        """
        style = detect_style(style)
        index_to_paper = {index: paper for index, paper in entries}

        if style in NUMERIC_STYLES or style in {"bibtex", "ris"}:
            return draft

        def replace(match: re.Match[str]) -> str:
            numbers: list[int] = []
            for token in re.split(r"[,，]", match.group(0).strip("[]【】")):
                token = token.strip()
                if token.isdigit():
                    numbers.append(int(token))
            parts: list[str] = []
            for number in numbers:
                paper = index_to_paper.get(number)
                if paper is None:
                    continue
                rendered = format_inline(paper, style, index=number)
                # format_inline 对作者-年份制返回 "(X, Y)"，这里剥掉括号再合并
                parts.append(rendered.strip("()"))
            if not parts:
                return ""
            return "(" + "; ".join(parts) + ")"

        return _CITATION_GROUP_RE.sub(replace, draft or "")

    # -------------------------------------------------------------- 参考文献
    def build_references(
        self,
        entries: Sequence[tuple[int, Paper]],
        style: str = "gb7714",
        *,
        only_cited: str | None = None,
    ) -> str:
        """生成参考文献表。

        Args:
            only_cited: 传入正文时，只输出正文真正引用过的条目（推荐，
                避免"参考文献表里有但正文没引"的常见问题）。
        """
        style = detect_style(style)
        papers = [paper for _index, paper in entries]
        if only_cited is not None:
            cited = set(extract_citations(only_cited))
            kept = [paper for index, paper in entries if index in cited]
            if kept:
                papers = kept
        return format_reference_list(papers, style)

    def reference_entries(
        self, entries: Sequence[tuple[int, Paper]], style: str = "gb7714"
    ) -> list[dict[str, Any]]:
        """结构化参考文献列表（供右栏「引用列表」渲染并支持点击跳转）。

        ``text`` 必须是完整的参考文献文本，而不是文内引用短标：早期版本
        误用了 ``format_inline``，导致数字制样式下这个字段只剩一个 ``[1]``，
        前端引用列表因此只显示编号、看不到文献信息。
        """
        style = detect_style(style)
        out: list[dict[str, Any]] = []
        for index, paper in entries:
            out.append(
                {
                    "index": index,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "text": format_citation(paper, style, index=index),
                    "inline": format_inline(paper, style, index=index),
                    "citation_label": paper.citation_label,
                    "journal": paper.journal,
                    "pub_year": paper.pub_year,
                    "doi": paper.doi,
                }
            )
        return out

    # ------------------------------------------------------------------ 导出
    def export(
        self,
        *,
        draft: str,
        entries: Sequence[tuple[int, Paper]],
        style: str = "gb7714",
        name: str = "review",
        title: str = "",
        formats: Sequence[str] = ("markdown",),
        meta: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """导出成稿（Markdown / HTML / 参考文献格式）。返回 ``{格式: 路径}``。"""
        style = detect_style(style)
        papers = [paper for _index, paper in entries]
        results: dict[str, str] = {}

        for fmt in formats:
            key = str(fmt).lower()
            if key in {"markdown", "md"}:
                content = export_markdown(
                    title=title or name,
                    content=draft,
                    references=papers,
                    style=style,
                    meta=meta,
                )
                results["markdown"] = str(
                    write_export(content, name=name, fmt="markdown", config=self.config)
                )
            elif key == "html":
                from ..export.exporters import export_docx_compatible_html

                content = export_docx_compatible_html(
                    title=title or name, content=draft, references=papers, style=style
                )
                results["html"] = str(
                    write_export(content, name=name, fmt="html", config=self.config)
                )
            elif key in {"bibtex", "ris"}:
                results.update(
                    export_bundle(papers, name=name, styles=[key], config=self.config)
                )
        return results

    # ------------------------------------------------------------------ 自查
    def selfcheck(self, draft: str, entries: Sequence[tuple[int, Paper]]) -> dict[str, Any]:
        """不依赖 LLM 的成稿体检：引用、空章节、异常短段。"""
        report = self.validate_citations(draft, [index for index, _ in entries])
        issues: list[dict[str, Any]] = []

        if report.missing_from_list:
            issues.append(
                {
                    "severity": "high",
                    "type": "越界引用",
                    "detail": f"正文出现参考文献表中不存在的编号：{sorted(set(report.missing_from_list))}",
                    "suggestion": "删除或修正这些引用标记",
                }
            )
        if report.never_cited:
            issues.append(
                {
                    "severity": "low",
                    "type": "未引用条目",
                    "detail": f"参考文献表中有 {len(set(report.never_cited))} 条未被正文引用",
                    "suggestion": "已在导出时自动过滤，或将相关文献补写进正文",
                }
            )

        for section in _split_sections(draft):
            head, body = section
            if not body.strip():
                issues.append(
                    {
                        "severity": "medium",
                        "type": "空章节",
                        "detail": f"章节「{head}」内容为空",
                        "suggestion": "补充内容或删除该章节",
                    }
                )
            elif len(body.strip()) < 60 and head:
                issues.append(
                    {
                        "severity": "low",
                        "type": "内容过短",
                        "detail": f"章节「{head}」仅 {len(body.strip())} 字",
                        "suggestion": "考虑合并到相邻章节",
                    }
                )

        return {
            "citations": report.to_dict(),
            "issues": issues,
            "verdict": "pass" if not any(i["severity"] == "high" for i in issues) else "revise",
            "char_count": len(draft or ""),
        }


def _split_sections(draft: str) -> list[tuple[str, str]]:
    """按 ``## `` 标题切分草稿。"""
    sections: list[tuple[str, str]] = []
    current_title = ""
    buffer: list[str] = []
    for line in (draft or "").splitlines():
        if line.startswith("## "):
            if current_title or buffer:
                sections.append((current_title, "\n".join(buffer)))
            current_title = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            continue
        else:
            buffer.append(line)
    if current_title or buffer:
        sections.append((current_title, "\n".join(buffer)))
    return sections

"""引用格式化引擎。

支持需求文档要求的全部导出格式：

============ ==========================================================
``apa7``     APA 第 7 版（作者-年份制）
``vancouver`` Vancouver（数字制，医学期刊最常用）
``gb7714``   GB/T 7714-2015（中文期刊与学位论文标准）
``bibtex``   BibTeX（LaTeX / Zotero / EndNote 通用）
``ris``      RIS（EndNote / NoteExpress / Zotero 通用）
``chicago``  Chicago 作者-年份制（部分社科期刊）
============ ==========================================================

作者姓名在 :class:`~medscholar.models.Paper` 中统一按「姓 名」顺序存储
（见各 API 客户端的 ``to_family_first`` 归一化），因此本模块可以稳定地
提取姓氏，而不必猜测姓名顺序。
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models import Paper

__all__ = [
    "STYLES",
    "STYLE_LABELS",
    "split_author",
    "format_authors",
    "format_citation",
    "format_inline",
    "format_reference_list",
    "citation_key",
    "to_bibtex",
    "to_ris",
    "detect_style",
]

_CJK_RE = re.compile(r"[\u3400-\u9fff]")

STYLES: tuple[str, ...] = ("apa7", "vancouver", "gb7714", "chicago", "bibtex", "ris")

STYLE_LABELS: dict[str, str] = {
    "apa7": "APA 7th",
    "vancouver": "Vancouver",
    "gb7714": "GB/T 7714-2015",
    "chicago": "Chicago (author-date)",
    "bibtex": "BibTeX",
    "ris": "RIS",
}

_ALIASES = {
    "apa": "apa7", "apa7": "apa7", "apa-7": "apa7", "apa 7th": "apa7",
    "vancouver": "vancouver", "van": "vancouver",
    "gb": "gb7714", "gb7714": "gb7714", "gb/t 7714": "gb7714", "gbt7714": "gb7714",
    "gb/t7714-2015": "gb7714", "国标": "gb7714",
    "chicago": "chicago",
    "bibtex": "bibtex", "bib": "bibtex",
    "ris": "ris",
}


def detect_style(name: str | None, *, default: str = "gb7714") -> str:
    """把用户输入的样式名归一化。"""
    if not name:
        return default
    key = str(name).strip().lower()
    return _ALIASES.get(key, key if key in STYLES else default)


# ------------------------------------------------------------------ 作者姓名
def split_author(name: str) -> tuple[str, str]:
    """把「姓 名」形式的作者串拆成 ``(姓, 名)``。

    >>> split_author("Zhang Wei")
    ('Zhang', 'Wei')
    >>> split_author("van der Berg Jan")
    ('van der Berg', 'Jan')
    >>> split_author("王伟")
    ('王伟', '')
    """
    name = (name or "").strip().strip(".,;")
    if not name:
        return "", ""
    if _CJK_RE.search(name):
        # 中文姓名整体作为姓氏处理（GB/T 7714 保留全名）
        return name, ""

    # 复姓/带前缀的西方姓氏：van, von, de, del, der, la, le, di, da, bin, ibn …
    particles = {
        "van", "von", "de", "del", "della", "der", "den", "la", "le", "du",
        "di", "da", "dos", "bin", "ibn", "ter", "ten", "op", "st", "mac", "mc",
    }
    tokens = [t for t in re.split(r"\s+", name) if t]
    if len(tokens) == 1:
        return tokens[0], ""
    if tokens[0].lower() in particles and len(tokens) >= 3:
        return " ".join(tokens[:-1]), tokens[-1]
    return tokens[0], " ".join(tokens[1:])


def _initials(given: str, *, sep: str = ". ", join: str = "") -> str:
    """把名转成缩写形式：``Wei`` → ``W.``；``Jean-Pierre`` → ``J.-P.``"""
    given = (given or "").strip()
    if not given:
        return ""
    parts = [p for p in re.split(r"[\s\-]+", given) if p]
    out: list[str] = []
    for part in parts:
        out.append(part[0].upper() + sep)
    return join.join(out).strip()


def format_authors(
    authors: Sequence[str],
    style: str,
    *,
    language: str = "",
) -> str:
    """按样式格式化作者列表。"""
    style = detect_style(style)
    names = [a for a in (authors or []) if str(a).strip()]
    if not names:
        return "佚名" if language.startswith("zh") or style == "gb7714" else "Anonymous"

    if style == "gb7714":
        shown = names[:3]
        rendered = [str(n).strip() for n in shown]
        text = ", ".join(rendered)
        return text + (", 等" if len(names) > 3 else "")

    if style == "vancouver":
        shown = names[:6]
        rendered = []
        for name in shown:
            family, given = split_author(name)
            initials = _initials(given, sep="", join="")
            rendered.append(f"{family} {initials}".strip())
        text = ", ".join(rendered)
        return text + ", et al." if len(names) > 6 else text

    if style in {"apa7", "chicago"}:
        limit = 20
        rendered = []
        for name in names[:limit]:
            family, given = split_author(name)
            initials = _initials(given)
            rendered.append(f"{family}, {initials}".strip().rstrip(",") if initials else family)
        if len(names) > limit:
            rendered = rendered[:19] + ["…", rendered[-1]]
        if len(rendered) == 1:
            return rendered[0]
        return ", ".join(rendered[:-1]) + ", & " + rendered[-1]

    # 默认退化为 Vancouver
    return format_authors(names, "vancouver", language=language)


# ------------------------------------------------------------------ 单条文献
def _year(paper: Paper) -> str:
    return str(paper.pub_year) if paper.pub_year else "n.d."


def _journal(paper: Paper) -> str:
    return (paper.journal or "").strip() or "（期刊信息缺失）"


def _pages(paper: Paper) -> str:
    pages = (paper.pages or "").strip()
    if pages:
        return pages
    if paper.volume or paper.issue:
        return ""
    return ""


def _doi_url(paper: Paper) -> str:
    return f"https://doi.org/{paper.doi}" if paper.doi else ""


def format_citation(paper: Paper, style: str = "gb7714", *, index: int | None = None) -> str:
    """格式化单条参考文献。"""
    style = detect_style(style)
    if style == "bibtex":
        return to_bibtex(paper)
    if style == "ris":
        return to_ris(paper)

    authors = format_authors(paper.authors, style, language=paper.language)
    year = _year(paper)
    title = (paper.title or "").strip().rstrip(".")
    journal = _journal(paper)
    is_zh = bool(_CJK_RE.search(title))
    doi_url = _doi_url(paper)

    if style == "apa7":
        parts = [f"{authors} ({year}).", f"{title}."]
        source = journal
        if paper.volume:
            source += f", {paper.volume}"
            if paper.issue:
                source += f"({paper.issue})"
        pages = _pages(paper)
        if pages:
            source += f", {pages}"
        parts.append(source + ".")
        if doi_url:
            parts.append(doi_url)
        return " ".join(p for p in parts if p).strip()

    if style == "vancouver":
        prefix = f"{index}. " if index else ""
        authors_text = format_authors(paper.authors, "vancouver", language=paper.language)
        if not authors_text.endswith("."):
            authors_text += "."
        source = f"{journal}."
        if year:
            source += f" {year}"
        if paper.volume:
            source += f";{paper.volume}"
            if paper.issue:
                source += f"({paper.issue})"
        pages = _pages(paper)
        if pages:
            source += f":{pages}"
        source += "."
        tail = f" doi:{paper.doi}" if paper.doi else ""
        return f"{prefix}{authors_text} {title}. {source}{tail}".strip()

    if style == "chicago":
        volume_part = ""
        if paper.volume:
            volume_part = f" {paper.volume}"
            if paper.issue:
                volume_part += f"({paper.issue})"
        pages = _pages(paper)
        pages_part = f": {pages}" if pages else ""
        text = (
            f"{authors.rstrip('.')}. {year}. “{title}.” "
            f"{journal}{volume_part}{pages_part}."
        )
        return f"{text} {doi_url}".strip()

    # ---- GB/T 7714-2015
    prefix = f"[{index}] " if index else ""
    type_marker = _gb_type_marker(paper)
    parts = [f"{authors}.", f"{title}{type_marker}."]
    if is_zh:
        source = f"{journal}"
        if year:
            source += f", {year}"
        if paper.volume:
            source += f", {paper.volume}"
            if paper.issue:
                source += f"({paper.issue})"
        pages = _pages(paper)
        if pages:
            source += f": {pages}"
        parts.append(source + ".")
    else:
        source = f"{journal}"
        if year:
            source += f", {year}"
        if paper.volume:
            source += f", {paper.volume}"
            if paper.issue:
                source += f"({paper.issue})"
        pages = _pages(paper)
        if pages:
            source += f": {pages}"
        parts.append(source + ".")
    text = " ".join(p for p in parts if p).strip()
    if doi_url:
        text += f" DOI: {paper.doi}."
    return f"{prefix}{text}".strip()


def _gb_type_marker(paper: Paper) -> str:
    """GB/T 7714 文献类型标识：``[J]`` 期刊、``[D]`` 学位论文、``[C]`` 会议、``[M]`` 专著。"""
    ptype = (paper.publication_type or "").lower()
    if any(k in ptype for k in ("dissertation", "thesis")):
        return "[D]"
    if any(k in ptype for k in ("proceedings", "conference")):
        return "[C]"
    if any(k in ptype for k in ("book", "monograph")):
        return "[M]"
    if any(k in ptype for k in ("preprint",)):
        return "[EB/OL]"
    return "[J]"


def format_inline(
    paper: Paper, style: str = "gb7714", *, index: int | None = None
) -> str:
    """文内引用短标（用于正文中的引用标记）。"""
    style = detect_style(style)
    if style in {"vancouver", "gb7714", "bibtex", "ris"}:
        return f"[{index}]" if index else "[?]"
    family = split_author(paper.authors[0])[0] if paper.authors else "佚名"
    year = _year(paper)
    if len(paper.authors) >= 3:
        return f"({family} et al., {year})"
    if len(paper.authors) == 2:
        family2 = split_author(paper.authors[1])[0]
        return f"({family} & {family2}, {year})"
    return f"({family}, {year})"


def format_reference_list(
    papers: Sequence[Paper],
    style: str = "gb7714",
    *,
    numbered: bool | None = None,
    sort: str = "cited",
) -> str:
    """生成整份参考文献表。

    Args:
        numbered: 是否加序号；``None`` 时按样式决定（数字制加序号）。
        sort: ``cited`` 保持传入顺序（正文引用顺序）；``author`` 按作者排序；
              ``year`` 按年份降序。
    """
    style = detect_style(style)
    items = [p for p in papers if p]
    if style in {"bibtex", "ris"}:
        joiner = "\n\n" if style == "bibtex" else "\n"
        return joiner.join(format_citation(p, style) for p in items)

    if sort == "author":
        items = sorted(items, key=lambda p: (split_author(p.authors[0])[0].lower() if p.authors else "zzz"))
    elif sort == "year":
        items = sorted(items, key=lambda p: -(p.pub_year or 0))

    if numbered is None:
        numbered = style in {"vancouver", "gb7714", "chicago"}

    lines: list[str] = []
    for i, paper in enumerate(items, start=1):
        lines.append(format_citation(paper, style, index=i if numbered else None))
    return "\n".join(lines)


# ---------------------------------------------------------------- BibTeX / RIS
_BIBTEX_UNSAFE = re.compile(r"[^a-zA-Z0-9]")
_BIBTEX_STOP = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by",
    "study", "trial", "analysis", "effect", "effects", "review",
}


def citation_key(paper: Paper, *, taken: set[str] | None = None) -> str:
    """生成 BibTeX 引用键：``姓+年份+首个实词``，冲突时追加 a/b/c。"""
    family = split_author(paper.authors[0])[0] if paper.authors else "anon"
    family = _BIBTEX_UNSAFE.sub("", family).lower() or "anon"
    year = str(paper.pub_year or "nd")
    words = [
        w for w in _BIBTEX_UNSAFE.sub(" ", paper.title or "").split()
        if w.lower() not in _BIBTEX_STOP
    ]
    head = (words[0].lower() if words else "paper")[:12]
    base = f"{family}{year}{head}"
    if taken is None:
        return base
    if base not in taken:
        taken.add(base)
        return base
    for suffix in "abcdefghij":
        candidate = f"{base}{suffix}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    return base


def _bibtex_escape(text: str) -> str:
    return (
        (text or "")
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def to_bibtex(paper: Paper, *, key: str | None = None) -> str:
    """输出 BibTeX 条目。"""
    key = key or citation_key(paper)
    ptype = (paper.publication_type or "").lower()
    entry = "inproceedings" if "conference" in ptype or "proceedings" in ptype else (
        "phdthesis" if "dissertation" in ptype or "thesis" in ptype else "article"
    )

    fields: list[tuple[str, str]] = []
    if paper.authors:
        fields.append(("author", _bibtex_escape(" and ".join(paper.authors))))
    fields.append(("title", _bibtex_escape(paper.title)))
    if paper.journal:
        fields.append(("journal" if entry == "article" else "booktitle", _bibtex_escape(paper.journal)))
    if paper.pub_year:
        fields.append(("year", str(paper.pub_year)))
    if paper.volume:
        fields.append(("volume", _bibtex_escape(paper.volume)))
    if paper.issue:
        fields.append(("number", _bibtex_escape(paper.issue)))
    if paper.pages:
        fields.append(("pages", _bibtex_escape(paper.pages.replace("-", "--"))))
    if paper.doi:
        fields.append(("doi", paper.doi))
    if paper.url:
        fields.append(("url", paper.url))
    if paper.keywords:
        fields.append(("keywords", _bibtex_escape(", ".join(paper.keywords))))
    if paper.note:
        fields.append(("note", _bibtex_escape(paper.note)))

    width = max((len(k) for k, _ in fields), default=4)
    body = ",\n".join(f"  {k.ljust(width)} = {{{v}}}" for k, v in fields)
    return f"@{entry}{{{key},\n{body}\n}}"


def to_ris(paper: Paper) -> str:
    """输出 RIS 条目（EndNote / NoteExpress / Zotero 可直接导入）。"""
    lines: list[str] = []
    ptype = (paper.publication_type or "").lower()
    kind = "CONF" if "conference" in ptype or "proceedings" in ptype else (
        "THES" if "dissertation" in ptype or "thesis" in ptype else "JOUR"
    )
    lines.append(f"TY  - {kind}")
    for author in paper.authors:
        lines.append(f"AU  - {author}")
    lines.append(f"TI  - {paper.title}")
    if paper.journal:
        lines.append(f"JO  - {paper.journal}")
        lines.append(f"T2  - {paper.journal}")
    if paper.pub_year:
        lines.append(f"PY  - {paper.pub_year}")
    if paper.volume:
        lines.append(f"VL  - {paper.volume}")
    if paper.issue:
        lines.append(f"IS  - {paper.issue}")
    if paper.pages:
        first, _, last = paper.pages.partition("-")
        lines.append(f"SP  - {first.strip()}")
        if last:
            lines.append(f"EP  - {last.strip()}")
    if paper.doi:
        lines.append(f"DO  - {paper.doi}")
    if paper.url:
        lines.append(f"UR  - {paper.url}")
    if paper.abstract:
        lines.append("AB  - " + paper.abstract.replace("\n", " ")[:4000])
    for keyword in paper.keywords:
        lines.append(f"KW  - {keyword}")
    if paper.note:
        lines.append(f"N1  - {paper.note}")
    if paper.pmid:
        lines.append(f"AN  - PMID:{paper.pmid}")
    lines.append("ER  - ")
    return "\n".join(lines)


def format_records(
    papers: Sequence[Paper], style: str, *, key_prefix: str = ""
) -> str:
    """批量导出（BibTeX / RIS 用空行分隔；其他样式生成参考文献表）。"""
    style = detect_style(style)
    if style == "bibtex":
        taken: set[str] = set()
        return "\n\n".join(
            to_bibtex(p, key=citation_key(p, taken=taken)) for p in papers if p
        )
    if style == "ris":
        return "\n\n".join(to_ris(p) for p in papers if p)
    return format_reference_list(papers, style)

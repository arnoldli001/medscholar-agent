"""文献题录文件解析：RIS / BibTeX / EndNote 标记格式 / CSV。

为什么要做这个：Web of Science、Scopus、Embase、Cochrane、CNKI、万方
都支持把检索结果**导出**成 RIS / BibTeX / 标记文本，而这些数据库本身
没有开放 API。让用户"从学校订阅合法导出 → 导入本地库"，
既拿到了这些库的题录，又不涉及任何抓取或认证绕过。

设计原则：解析器只负责"文本 → Paper 列表"，不做网络、不碰数据库，
因此可以单独测试，也便于将来接新的导出格式。
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..models import Paper, coerce_int

logger = logging.getLogger(__name__)

__all__ = [
    "parse_ris",
    "parse_bibtex",
    "parse_tagged",
    "parse_wos_plain",
    "looks_like_wos_plain",
    "parse_csv",
    "detect_format",
    "parse_any",
    "SUPPORTED_FORMATS",
]

SUPPORTED_FORMATS = ("ris", "bibtex", "tagged", "wos", "csv")


# --------------------------------------------------------------------- RIS
#: RIS 标签 → Paper 字段。键是 RIS 标签，值是我们内部字段名。
_RIS_MAP: dict[str, str] = {
    "TI": "title", "T1": "title", "CT": "title", "BT": "title",
    "AU": "authors", "A1": "authors", "A2": "authors", "ED": "authors",
    "AB": "abstract", "N2": "abstract",
    # 期刊名：不同数据库各用各的标签（WoS 用 SO，Scopus 用 JO，PubMed 用 JF/JA）
    "JO": "journal", "JF": "journal", "JA": "journal", "T2": "journal",
    "J2": "journal", "SO": "journal",
    "PY": "pub_year", "DA": "pub_year", "Y1": "pub_year", "PD": "pub_year",
    # DOI：RIS 规范是 DO，但 Web of Science 用 DI。少了这个键，
    # WoS 导出的文献会全部丢掉 DOI —— 去重和全文获取都会跟着失效。
    "DO": "doi", "DI": "doi",
    "AN": "source_id",
    "UR": "url", "L1": "full_text_url", "L2": "full_text_url",
    "KW": "keywords", "DE": "keywords",
    "VL": "volume",
    "IS": "issue",
    # 页码：WoS 用 BP/EP，其它多用 SP/EP
    "SP": "pages", "BP": "pages", "EP": "pages_end",
    "LA": "language",
    "M3": "publication_type", "DT": "publication_type",
    "PM": "pmid",
    "PMC": "pmcid",
    "SN": "issn",
    # 被引次数：WoS 用 TC/Z9
    "TC": "cited_by_count", "Z9": "cited_by_count",
}
_RIS_TYPE_MAP: dict[str, str] = {
    "JOUR": "journal-article",
    "JFULL": "journal-article",
    "EJOUR": "journal-article",
    "CONF": "conference-paper",
    "CPAPER": "conference-paper",
    "BOOK": "book",
    "CHAP": "book-chapter",
    "THES": "thesis",
    "RPRT": "report",
    "NEWS": "news",
    "ELEC": "web",
}

#: 作者字段里可能出现 "Smith, John A." 或 "John A. Smith"；
#: 也常见 RIS 用 "Smith, J.A." 形式，这里保持原样（models 会统一清洗）。
_RIS_LINE_RE = re.compile(r"^([A-Z][A-Z0-9])\s{1,2}-\s?(.*)$")

#: 可以出现多次的字段（每行一个值），续行不能并入这些字段。
_REPEATABLE_FIELDS = frozenset({"authors", "keywords"})


@dataclass
class _Record:
    fields: dict[str, list[str]]
    ris_type: str = ""

    def first(self, key: str) -> str:
        values = self.fields.get(key) or []
        return values[0].strip() if values else ""

    def all(self, key: str) -> list[str]:
        return [v.strip() for v in (self.fields.get(key) or []) if v.strip()]


def parse_ris(text: str, *, default_source: str = "import") -> list[Paper]:
    """解析 RIS 文本（Web of Science / Scopus / Embase / EndNote 导出通用）。"""
    records: list[_Record] = []
    current: _Record | None = None

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        match = _RIS_LINE_RE.match(line)
        if not match:
            # 续行：接到上一个标签的值后面
            if current is not None and current.fields:
                last_key = next(reversed(current.fields))
                tail = current.fields[last_key]
                if tail:
                    tail[-1] = f"{tail[-1]} {line.strip()}".strip()
            continue

        tag, value = match.group(1), match.group(2).strip()
        if tag == "TY":
            current = _Record(fields={}, ris_type=value.upper())
            records.append(current)
            continue
        if tag == "ER":
            current = None
            continue
        if current is None:
            # 没有 TY 开头的裸记录：宽容处理，直接建一条
            current = _Record(fields={})
            records.append(current)

        field = _RIS_MAP.get(tag)
        if field:
            current.fields.setdefault(field, []).append(value)
        else:
            current.fields.setdefault(f"_{tag}", []).append(value)

    papers: list[Paper] = []
    for record in records:
        paper = _record_to_paper(record, default_source)
        if paper is not None:
            papers.append(paper)
    return papers


def _record_to_paper(record: _Record, default_source: str) -> Paper | None:
    title = record.first("title")
    if not title:
        return None

    pages = record.first("pages")
    pages_end = record.first("pages_end")
    if pages and pages_end and pages_end not in pages:
        pages = f"{pages}-{pages_end}"

    url = record.first("url")
    full_text_url = record.first("full_text_url")
    doi = record.first("doi")
    if not url and doi:
        url = f"https://doi.org/{doi}"

    paper = Paper(
        title=title,
        source=default_source,
        abstract=record.first("abstract"),
        authors=record.all("authors"),
        journal=record.first("journal"),
        pub_year=coerce_int(record.first("pub_year")),
        doi=doi or None,
        pmid=record.first("pmid") or None,
        pmcid=record.first("pmcid") or None,
        keywords=_split_keywords(record.all("keywords")),
        volume=record.first("volume"),
        issue=record.first("issue"),
        pages=pages,
        language=record.first("language"),
        publication_type=_RIS_TYPE_MAP.get(record.ris_type, record.first("publication_type")),
        url=url,
        full_text_url=full_text_url,
        source_id=record.first("source_id"),
        cited_by_count=coerce_int(record.first("cited_by_count")) or 0,
        is_open_access=bool(full_text_url),
    )
    return paper


# ------------------------------------------------------------------ BibTeX
_BIB_ENTRY_RE = re.compile(r"@([a-zA-Z]+)\s*[{(]", re.MULTILINE)
_BIB_TYPE_MAP: dict[str, str] = {
    "article": "journal-article",
    "inproceedings": "conference-paper",
    "conference": "conference-paper",
    "book": "book",
    "inbook": "book-chapter",
    "incollection": "book-chapter",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "report",
    "misc": "other",
    "unpublished": "other",
    "online": "web",
}

#: BibTeX 里常见的 LaTeX 转义 → Unicode。只处理标题/作者里高频出现的那些，
#: 不做完整 LaTeX 渲染（那是另一个量级的工程）。
_LATEX_MAP: dict[str, str] = {
    r"\\&": "&", r"\\%": "%", r"\\$": "$", r"\\#": "#", r"\\_": "_",
    r"\\{": "{", r"\\}": "}", r"~": " ", r"\\,": "", r"\\ ": " ",
    r"--": "-", r"---": "—",
    r'\\"a': "ä", r'\\"o': "ö", r'\\"u': "ü", r'\\"A': "Ä", r'\\"O': "Ö", r'\\"U': "Ü",
    r"\\'e": "é", r"\\'a": "á", r"\\'i": "í", r"\\'o": "ó", r"\\'u": "ú",
    r"\\`e": "è", r"\\`a": "à", r"\\~n": "ñ", r"\\c{c}": "ç", r"\\ss": "ß",
    r"\\aa": "å", r"\\o": "ø",
}
_LATEX_BRACE_RE = re.compile(r"[{}]")
_SHORT_JOURNAL_RE = re.compile(r"\\[a-zA-Z]+\s*")


def _latex_to_text(value: str) -> str:
    text = value or ""
    for pattern, replacement in _LATEX_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = _LATEX_BRACE_RE.sub("", text)
    text = _SHORT_JOURNAL_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_bibtex(text: str, *, default_source: str = "import") -> list[Paper]:
    """解析 BibTeX（Web of Science / Scopus / Zotero / Google Scholar 导出）。"""
    papers: list[Paper] = []
    for entry_type, body in _iter_bib_entries(text or ""):
        fields = _parse_bib_fields(body)
        title = _latex_to_text(fields.get("title", ""))
        if not title:
            continue

        authors = _split_bib_authors(_latex_to_text(fields.get("author", "")))
        year = coerce_int(_first_year(fields.get("year", "") or fields.get("date", "")))
        doi = (fields.get("doi") or "").strip() or None
        url = (fields.get("url") or "").strip()
        journal = _latex_to_text(fields.get("journal") or fields.get("journaltitle") or "")
        if not journal:
            journal = _latex_to_text(fields.get("booktitle") or "")

        keywords = [
            _latex_to_text(k)
            for k in re.split(r"[;,]", fields.get("keywords", "") or "")
            if k.strip()
        ]

        paper = Paper(
            title=title,
            source=default_source,
            abstract=_latex_to_text(fields.get("abstract", "")),
            authors=authors,
            journal=journal,
            pub_year=year,
            doi=doi,
            pmid=(fields.get("pmid") or "").strip() or None,
            pmcid=(fields.get("pmcid") or "").strip() or None,
            keywords=keywords,
            volume=(fields.get("volume") or "").strip(),
            issue=(fields.get("number") or "").strip(),
            pages=(fields.get("pages") or "").strip(),
            publication_type=_BIB_TYPE_MAP.get(entry_type.lower(), ""),
            url=url or (f"https://doi.org/{doi}" if doi else ""),
            is_open_access=bool(fields.get("eprint") or ""),
        )
        papers.append(paper)
    return papers


def _iter_bib_entries(text: str) -> Iterable[tuple[str, str]]:
    """按花括号/引号配平切出每个条目，返回 ``(类型, 条目体)``。"""
    index = 0
    while True:
        match = _BIB_ENTRY_RE.search(text, index)
        if not match:
            return
        entry_type = match.group(1)
        start = match.end() - 1
        opener = text[start]
        closer = "}" if opener == "{" else ")"
        depth = 0
        in_quote = False
        escaped = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_quote = not in_quote
                continue
            if in_quote:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    yield entry_type, text[start + 1 : pos]
                    index = pos + 1
                    break
        else:
            return


def _parse_bib_fields(body: str) -> dict[str, str]:
    """解析 ``key = {value}, key = "value"`` 形式的字段。"""
    # 跳过引用键（第一个逗号之前的部分）
    if "," in body:
        body = body.split(",", 1)[1]

    fields: dict[str, str] = {}
    pos = 0
    length = len(body)
    while pos < length:
        eq = body.find("=", pos)
        if eq < 0:
            break
        key = body[pos:eq].strip().strip(",").strip().lower()
        cursor = eq + 1
        while cursor < length and body[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        opener = body[cursor]
        if opener == "{":
            depth = 0
            value_start = cursor + 1
            in_quote = False
            escaped = False
            idx = cursor
            while idx < length:
                ch = body[idx]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"' and not escaped:
                    in_quote = not in_quote
                elif not in_quote and ch == "{":
                    depth += 1
                elif not in_quote and ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                idx += 1
            value = body[value_start:idx]
            pos = idx + 1
        elif opener == '"':
            idx = cursor + 1
            escaped = False
            while idx < length:
                ch = body[idx]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    break
                idx += 1
            value = body[cursor + 1 : idx]
            pos = idx + 1
        else:
            idx = cursor
            while idx < length and body[idx] not in ",\n":
                idx += 1
            value = body[cursor:idx]
            pos = idx

        if key:
            fields[key] = value.strip()
    return fields


def _split_bib_authors(raw: str) -> list[str]:
    """``"Smith, J. and Doe, A."`` → ``["Smith, J.", "Doe, A."]``。"""
    if not raw.strip():
        return []
    parts = re.split(r"\s+and\s+", raw)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "others"]


def _first_year(raw: str) -> str:
    match = re.search(r"(1[6-9]\d{2}|20\d{2})", raw or "")
    return match.group(1) if match else ""


# --------------------------------------------------- EndNote / RefMan 标记格式
#: CNKI、万方、维普等的 "EndNote" 导出就是这种 ``%标签 值`` 格式
_TAGGED_MAP: dict[str, str] = {
    "0": "publication_type",
    "A": "authors",
    "T": "title",
    "J": "journal",
    "B": "journal",
    "D": "pub_year",
    "R": "doi",
    "K": "keywords",
    "X": "abstract",
    "U": "url",
    "V": "volume",
    "N": "issue",
    "P": "pages",
    "L": "language",
    "W": "source_id",
    "@": "issn",
}
_TAGGED_TYPE_MAP: dict[str, str] = {
    "journal article": "journal-article",
    "conference paper": "conference-paper",
    "book": "book",
    "thesis": "thesis",
    "report": "report",
    "web page": "web",
}


def parse_tagged(text: str, *, default_source: str = "import") -> list[Paper]:
    """解析 ``%0 Journal Article`` 这类标记格式（CNKI/万方/EndNote 导出）。"""
    papers: list[Paper] = []
    for record in _split_tagged_records(text):
        title = (record.get("title") or [""])[0].strip()
        if not title:
            continue
        doi = (record.get("doi") or [""])[0].strip() or None
        url = (record.get("url") or [""])[0].strip()
        paper = Paper(
            title=title,
            source=default_source,
            abstract=(record.get("abstract") or [""])[0].strip(),
            authors=[a.strip() for a in (record.get("authors") or []) if a.strip()],
            journal=(record.get("journal") or [""])[0].strip(),
            pub_year=coerce_int((record.get("pub_year") or [""])[0]),
            doi=doi,
            keywords=_split_keywords(record.get("keywords") or []),
            volume=(record.get("volume") or [""])[0].strip(),
            issue=(record.get("issue") or [""])[0].strip(),
            pages=(record.get("pages") or [""])[0].strip(),
            language=(record.get("language") or [""])[0].strip(),
            publication_type=_TAGGED_TYPE_MAP.get(
                (record.get("publication_type") or [""])[0].strip().lower(), ""
            ),
            url=url or (f"https://doi.org/{doi}" if doi else ""),
            source_id=(record.get("source_id") or [""])[0].strip(),
        )
        papers.append(paper)
    return papers


def _split_keywords(values: list[str]) -> list[str]:
    """关键词常写成 ``a; b; c`` 或 ``a, b``，要拆开。

    CNKI/万方的标记格式把整串关键词放在一个 ``%K`` 里，
    不拆开的话"关键词"就变成一整句，检索命中率会变差。
    """
    out: list[str] = []
    for value in values:
        for part in re.split(r"[;,；，]", value or ""):
            text = part.strip()
            if text:
                out.append(text)
    return out


def _split_tagged_records(text: str) -> list[dict[str, list[str]]]:
    """按 ``%0`` 切分标记格式记录（有些导出没有 %0，就整体当一条）。"""
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    seen_zero = False

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("%"):
            continue
        tag, _, value = line[1:].partition(" ")
        tag = tag.strip()
        if tag == "0":
            seen_zero = True
            if current:
                records.append(current)
            current = {}
        field = _TAGGED_MAP.get(tag)
        if field:
            current.setdefault(field, []).append(value.strip())

    if current:
        records.append(current)
    if not seen_zero and len(records) > 1:
        # 没有 %0 分隔符却切出多条 → 说明 %0 不是记录边界，退回整体一条
        merged: dict[str, list[str]] = {}
        for record in records:
            for key, values in record.items():
                merged.setdefault(key, []).extend(values)
        return [merged]
    return records


# ------------------------------------------- Web of Science 纯文本导出（无短横线）
#: WoS 默认的 "Plain Text" 导出长这样：``AU Zhang, Wei``（标签后只有一个空格），
#: 续行以三个空格缩进，记录以 ``ER`` 结束。它**不是** RIS，很容易被误判。
_WOS_LINE_RE = re.compile(r"^([A-Z][A-Z0-9])\s(.*)$")
_WOS_CONT_RE = re.compile(r"^\s{2,}\S")
#: 纯文本导出里各标签的含义（与 RIS 大同小异，但期刊名是 SO、页码是 BP/EP）
_WOS_TAGS: frozenset[str] = frozenset(
    {"PT", "AU", "TI", "SO", "VL", "IS", "BP", "EP", "PY", "DI", "AB", "LA",
     "DT", "DE", "SN", "UR", "PU", "PD", "PG", "TC", "Z9", "UT", "AF", "EM",
     "C1", "RP", "NR", "TC", "WE", "SC", "ID", "FX"}
)
_WOS_TYPE_MAP: dict[str, str] = {
    "J": "journal-article",
    "C": "conference-paper",
    "B": "book",
    "S": "book",
    "P": "patent",
    "R": "report",
    "D": "dataset",
}


def looks_like_wos_plain(text: str) -> bool:
    """判断是否是 WoS 纯文本导出。

    特征：以 ``FN``/``VR``/``PT`` 起头，且有 ``AU``/``TI``/``SO`` 这类两字符标签，
    但**没有** RIS 的 ``标签 + 短横线`` 形式。
    """
    body = (text or "").lstrip("\ufeff")
    if not body.strip():
        return False
    head = "\n".join(body.splitlines()[:12])
    if _RIS_LINE_RE.search(head):
        return False
    if not re.search(r"^(FN|VR|PT)\s", head, re.MULTILINE):
        return False
    return bool(re.search(r"^(AU|TI|SO)\s", head, re.MULTILINE))


def parse_wos_plain(text: str, *, default_source: str = "import") -> list[Paper]:
    """解析 Web of Science 的纯文本导出（保存为 .txt 的那种）。"""
    records: list[_Record] = []
    current: _Record | None = None
    last_tag = ""

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        if line.strip() == "EF":  # 文件结束
            break

        if _WOS_CONT_RE.match(line) and current is not None and last_tag:
            field = _RIS_MAP.get(last_tag)
            if field:
                values = current.fields.setdefault(field, [])
                if field in _REPEATABLE_FIELDS:
                    # WoS 纯文本只给第一位作者写 `AU`，其余作者是缩进的续行；
                    # 因此可重复字段的续行是**新的一项**，不是上一项的延续。
                    values.append(line.strip())
                elif values:
                    values[-1] = f"{values[-1]} {line.strip()}".strip()
            continue

        match = _WOS_LINE_RE.match(line)
        if not match:
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag not in _WOS_TAGS:
            continue

        if tag == "PT":
            current = _Record(fields={}, ris_type=value.upper())
            records.append(current)
            last_tag = tag
            continue
        if tag == "ER":
            current = None
            last_tag = ""
            continue
        if current is None:
            current = _Record(fields={})
            records.append(current)
        last_tag = tag

        field = _RIS_MAP.get(tag)
        if field:
            current.fields.setdefault(field, []).append(value)
            if tag == "PT":
                current.ris_type = value.upper()
        else:
            current.fields.setdefault(f"_{tag}", []).append(value)

    papers: list[Paper] = []
    for record in records:
        paper = _record_to_paper(record, default_source)
        if paper is not None:
            # WoS 的 `PT J`（文献类型代码）比 `DT Article` 更规范，优先采用
            mapped = _WOS_TYPE_MAP.get(record.ris_type)
            if mapped:
                paper.publication_type = mapped
            papers.append(paper)
    return papers


# ---------------------------------------------------------------------- CSV
#: CSV 表头 → 内部字段（大小写与空格不敏感）
_CSV_HEADERS: dict[str, str] = {
    "title": "title", "标题": "title", "题名": "title", "ti": "title", "article title": "title",
    "author": "authors", "authors": "authors", "作者": "authors", "au": "authors",
    "abstract": "abstract", "摘要": "abstract", "ab": "abstract",
    "journal": "journal", "期刊": "journal", "来源": "journal", "刊名": "journal",
    "so": "journal", "source title": "journal", "publication name": "journal",
    "year": "pub_year", "年份": "pub_year", "年": "pub_year", "py": "pub_year",
    "publication year": "pub_year", "date": "pub_year",
    "doi": "doi", "di": "doi",
    "pmid": "pmid", "pubmed id": "pmid",
    "keywords": "keywords", "关键词": "keywords", "author keywords": "keywords",
    "volume": "volume", "卷": "volume",
    "issue": "issue", "期": "issue",
    "pages": "pages", "页码": "pages", "page start": "pages",
    "url": "url", "链接": "url", "link": "url",
    "language": "language", "语种": "language",
    "document type": "publication_type", "文献类型": "publication_type",
    "cited by": "cited_by_count", "被引频次": "cited_by_count", " citations": "cited_by_count",
}


def parse_csv(text: str, *, default_source: str = "import") -> list[Paper]:
    """解析 CSV/TSV（Web of Science 和 Scopus 的 "导出为表格" 就是这种）。"""
    body = (text or "").lstrip("\ufeff")
    if not body.strip():
        return []

    sample = body[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(body), dialect=dialect)
    if not reader.fieldnames:
        return []

    mapping: dict[str, str] = {}
    for name in reader.fieldnames:
        key = (name or "").strip().lower()
        if key in _CSV_HEADERS:
            mapping[name] = _CSV_HEADERS[key]

    if "title" not in mapping.values():
        return []

    papers: list[Paper] = []
    for row in reader:
        data: dict[str, Any] = {}
        for column, field in mapping.items():
            value = (row.get(column) or "").strip()
            if not value:
                continue
            if field in {"authors", "keywords"}:
                data.setdefault(field, [])
                data[field].extend(
                    part.strip() for part in re.split(r"[;|]", value) if part.strip()
                )
            elif field == "cited_by_count":
                data[field] = coerce_int(value) or 0
            elif field == "pub_year":
                data[field] = coerce_int(value)
            else:
                data[field] = value

        title = str(data.get("title") or "").strip()
        if not title:
            continue
        doi = str(data.get("doi") or "").strip() or None
        papers.append(
            Paper(
                title=title,
                source=default_source,
                abstract=str(data.get("abstract") or ""),
                authors=list(data.get("authors") or []),
                journal=str(data.get("journal") or ""),
                pub_year=data.get("pub_year"),
                doi=doi,
                pmid=str(data.get("pmid") or "") or None,
                keywords=list(data.get("keywords") or []),
                volume=str(data.get("volume") or ""),
                issue=str(data.get("issue") or ""),
                pages=str(data.get("pages") or ""),
                language=str(data.get("language") or ""),
                publication_type=str(data.get("publication_type") or ""),
                url=str(data.get("url") or "") or (f"https://doi.org/{doi}" if doi else ""),
                cited_by_count=int(data.get("cited_by_count") or 0),
            )
        )
    return papers


# ------------------------------------------------------------------ 格式识别
def detect_format(text: str, filename: str = "") -> str:
    """判断题录文件格式，返回 ``ris/bibtex/tagged/wos/csv`` 之一。

    顺序很重要：**先按内容排除**再按扩展名判断。
    踩过的坑：WoS 的纯文本导出存成 .txt，第一行恰好同时含 "title" 和逗号，
    被误判成 CSV；所以 BibTeX / WoS 这些有强特征的格式必须先识别。
    """
    body = (text or "").lstrip("\ufeff")
    name = (filename or "").lower()
    extension = name.rsplit(".", 1)[-1] if "." in name else ""

    # 1) 强特征优先，与扩展名无关
    if _BIB_ENTRY_RE.search(body):
        return "bibtex"
    if re.search(r"^TY\s{1,2}-", body, re.MULTILINE) and re.search(
        r"^ER\s{1,2}-", body, re.MULTILINE
    ):
        return "ris"
    if looks_like_wos_plain(body):
        return "wos"
    if re.search(r"^%[0-9A-Z@]", body, re.MULTILINE):
        return "tagged"
    if re.search(r"^TY\s{1,2}-", body, re.MULTILINE):
        return "ris"  # 偶尔缺 ER

    # 2) 再看扩展名
    if extension == "ris":
        return "ris"
    if extension in {"bib", "bibtex"}:
        return "bibtex"
    if extension in {"enw", "ref", "note"}:
        return "tagged"
    if extension in {"csv", "tsv"}:
        return "csv"
    if extension == "txt" and _looks_like_csv(body):
        return "csv"
    if _looks_like_csv(body):
        return "csv"
    return ""


def _looks_like_csv(text: str) -> bool:
    """CSV 判定要**严**：必须有像表头的首行，且命中已知列名。

    否则 WoS/BibTeX 的文本导出会被误判（第一行常含 "title" 和逗号）。
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    header = lines[0].lower()
    if header.lstrip().startswith(("@", "%", "TY ", "FN ", "PT ")):
        return False
    separator = max((",", "\t", ";"), key=header.count)
    if header.count(separator) < 1:
        return False
    columns = [c.strip().strip('"') for c in header.split(separator)]
    known = sum(1 for c in columns if c in _CSV_HEADERS)
    # 至少要认出两列，且列名占首行的多数，避免把散文误当表头
    return known >= 2 and known * 2 >= len(columns)


def parse_any(
    text: str, *, filename: str = "", default_source: str = "import"
) -> tuple[list[Paper], str]:
    """自动识别格式并解析，返回 ``(papers, 实际使用的格式)``。"""
    fmt = detect_format(text, filename)
    if not fmt:
        return [], ""
    parsers = {
        "ris": parse_ris,
        "bibtex": parse_bibtex,
        "tagged": parse_tagged,
        "wos": parse_wos_plain,
        "csv": parse_csv,
    }
    parser = parsers.get(fmt)
    if parser is None:
        return [], ""
    return parser(text, default_source=default_source), fmt

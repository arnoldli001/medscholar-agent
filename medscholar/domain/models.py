"""MedScholar Agent 数据模型。

所有学术 API 客户端统一返回 :class:`Paper`，数据库读写与 Agent 层也只依赖该结构，
因此新增数据源时不需要改动下游任何代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping

from .text import (
    clean_abstract,
    normalize_doi,
    normalize_text,
    title_fingerprint,
)

__all__ = [
    "Paper",
    "ScoredPaper",
    "SearchLogEntry",
    "SOURCE_LABELS",
    "coerce_int",
    "coerce_str_list",
]

#: 数据源显示名
SOURCE_LABELS: dict[str, str] = {
    "pubmed": "PubMed",
    "europepmc": "Europe PMC",
    "s2": "Semantic Scholar",
    "openalex": "OpenAlex",
    "arxiv": "arXiv",
    "cnki": "CNKI",
    "crossref": "Crossref",
    "doaj": "DOAJ",
    "core": "CORE",
    "import": "导入",
    "zotero": "Zotero",
    "bulk": "官方批量包",
    "local": "本地库",
    "manual": "手动录入",
}


def coerce_int(value: Any) -> int | None:
    """尽力把任意值转成 int，失败返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        # 处理 "2021-05-01" / "2021" 这类日期
        if "-" in text:
            text = text.split("-")[0]
        return int(float(text))
    except (TypeError, ValueError):
        return None


def coerce_str_list(value: Any) -> list[str]:
    """把作者/关键词等字段统一成去空、去重、保序的字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item is None:
            continue
        if isinstance(item, Mapping):
            # 常见于 OpenAlex / Crossref 的 {display_name: ...} / {family:, given:}
            name = (
                item.get("display_name")
                or item.get("name")
                or item.get("family")
                or item.get("literal")
                or ""
            )
            given = item.get("given") or item.get("first") or ""
            if name and given and item.get("family"):
                name = f"{name} {given}".strip()
            item = name
        text = normalize_text(str(item)).strip(" ,;")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


@dataclass(slots=True)
class Paper:
    """一篇文献的规范化元数据。

    ``papers`` 表字段与本类一一对应；``raw`` 仅用于调试，不落库。
    """

    title: str
    source: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_year: int | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    cited_by_count: int = 0
    is_open_access: bool = False
    full_text_url: str = ""
    full_text_path: str = ""
    url: str = ""
    source_id: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publication_type: str = ""
    language: str = ""
    note: str = ""
    #: 该文献在哪些数据源中被检索到（跨库去重后合并；不落库）
    found_in: list[str] = field(default_factory=list)
    paper_id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    # ------------------------------------------------------------------ 构造
    def __post_init__(self) -> None:
        self.title = normalize_text(self.title)
        self.abstract = clean_abstract(self.abstract)
        self.authors = coerce_str_list(self.authors)
        self.mesh_terms = coerce_str_list(self.mesh_terms)
        self.keywords = coerce_str_list(self.keywords)
        self.journal = normalize_text(self.journal)
        self.doi = normalize_doi(self.doi)
        self.pmid = (str(self.pmid).strip() or None) if self.pmid else None
        self.pmcid = (str(self.pmcid).strip() or None) if self.pmcid else None
        self.pub_year = coerce_int(self.pub_year)
        self.cited_by_count = coerce_int(self.cited_by_count) or 0
        self.is_open_access = bool(self.is_open_access)
        self.source = (self.source or "manual").strip().lower()
        self.full_text_url = normalize_text(self.full_text_url)
        self.full_text_path = normalize_text(self.full_text_path)
        self.url = normalize_text(self.url)
        self.source_id = normalize_text(self.source_id)
        self.volume = normalize_text(self.volume)
        self.issue = normalize_text(self.issue)
        self.pages = normalize_text(self.pages)
        self.publication_type = normalize_text(self.publication_type)
        self.language = normalize_text(self.language)
        self.note = normalize_text(self.note)

    # -------------------------------------------------------------- 去重与显示
    @property
    def dedup_key(self) -> str:
        """跨库去重键：DOI 优先，其次 PMID，最后标题指纹。"""
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        fp = title_fingerprint(self.title)
        return f"title:{fp}" if fp else f"src:{self.source}:{self.title[:80]}"

    @property
    def embed_text(self) -> str:
        """用于生成向量的文本：标题 + 摘要（+ 期刊/年份作为弱信号）。"""
        parts = [self.title]
        if self.mesh_terms:
            parts.append("MeSH: " + "; ".join(self.mesh_terms[:12]))
        if self.abstract:
            parts.append(self.abstract)
        return "\n".join(p for p in parts if p).strip()

    @property
    def short_authors(self) -> str:
        """作者简写：前三作者 + et al. / 等。"""
        if not self.authors:
            return "佚名"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:3]) + ", et al."

    @property
    def citation_label(self) -> str:
        """文内引用短标签，如 ``Zhang 2023``。"""
        first = self.authors[0] if self.authors else "佚名"
        # 中文姓名取姓氏（首字），西文取末词（姓氏）
        surname = first.split()[-1] if " " in first else first
        year = self.pub_year or "n.d."
        return f"{surname} {year}"

    def to_row(self) -> dict[str, Any]:
        """转换为可直接写入 SQLite 的扁平字典（列表字段存 JSON 字符串）。"""
        import json

        row = asdict(self)
        for key in ("authors", "mesh_terms", "keywords"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        row["is_open_access"] = 1 if self.is_open_access else 0
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Paper":
        """从 SQLite 行（sqlite3.Row 或 dict）还原 Paper。

        对缺失字段保持宽容（缺列取默认值），这样 SELECT 列表变化不会引发崩溃。
        """
        import json

        def raw(key: str, default: Any = None) -> Any:
            try:
                return row[key]
            except (KeyError, IndexError):
                return default

        def _load(value: Any) -> list[str]:
            if not value:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return [str(value)]
            return parsed if isinstance(parsed, list) else [str(parsed)]

        return cls(
            paper_id=raw("paper_id"),
            title=raw("title") or "",
            source=raw("source") or "manual",
            abstract=raw("abstract") or "",
            authors=_load(raw("authors")),
            journal=raw("journal") or "",
            pub_year=raw("pub_year"),
            pmid=raw("pmid"),
            pmcid=raw("pmcid"),
            doi=raw("doi"),
            mesh_terms=_load(raw("mesh_terms")),
            keywords=_load(raw("keywords")),
            cited_by_count=raw("cited_by_count") or 0,
            is_open_access=bool(raw("is_open_access")),
            full_text_url=raw("full_text_url") or "",
            full_text_path=raw("full_text_path") or "",
            url=raw("url") or "",
            source_id=raw("source_id") or "",
            volume=raw("volume") or "",
            issue=raw("issue") or "",
            pages=raw("pages") or "",
            publication_type=raw("publication_type") or "",
            language=raw("language") or "",
            note=raw("note") or "",
            created_at=raw("created_at"),
            updated_at=raw("updated_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """面向 API/前端的 JSON 友好字典（不含 raw）。"""
        data = asdict(self)
        data["dedup_key"] = self.dedup_key
        data["citation_label"] = self.citation_label
        data["short_authors"] = self.short_authors
        data["source_label"] = SOURCE_LABELS.get(self.source, self.source)
        return data


@dataclass(slots=True)
class ScoredPaper:
    """混合检索结果：文献 + 融合分数 + 各路排名明细。"""

    paper: Paper
    score: float = 0.0
    fts_rank: int | None = None
    vector_rank: int | None = None
    fts_score: float | None = None
    vector_distance: float | None = None
    matched_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = self.paper.to_dict()
        data.update(
            score=round(self.score, 6),
            fts_rank=self.fts_rank,
            vector_rank=self.vector_rank,
            fts_score=None if self.fts_score is None else round(self.fts_score, 4),
            vector_distance=(
                None if self.vector_distance is None else round(self.vector_distance, 4)
            ),
            matched_by=self.matched_by,
        )
        return data


@dataclass(slots=True)
class SearchLogEntry:
    """一次检索的记录（``search_logs`` 表）。"""

    query: str
    source: str
    result_count: int
    new_count: int = 0
    duration_ms: int = 0
    error: str = ""
    id: int | None = None
    created_at: str | None = None

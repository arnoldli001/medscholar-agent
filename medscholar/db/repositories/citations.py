"""引用关系（citations 表）：写入某篇文献的参考文献、查它引了谁 / 谁引了它。

单独拆出来是因为它是一张纯关系表，读写都是"按标识符找本地 ID + 落一条边"，
与文献主表的去重富化、检索排序没有任何耦合，放在自己的文件里才好单独测。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..connect import Database
from ._common import _db

__all__ = [
    "add_citation",
    "add_citations",
    "get_references",
    "get_citing_papers",
]


def add_citation(
    citing_paper_id: int,
    *,
    cited_paper_id: int | None = None,
    cited_external_id: str | None = None,
    cited_title: str = "",
    source: str = "",
    db: Database | None = None,
) -> None:
    database = _db(db)
    with database.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO citations"
            "(citing_paper_id, cited_paper_id, cited_external_id, cited_title, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (citing_paper_id, cited_paper_id, cited_external_id, cited_title or None, source),
        )


def add_citations(
    citing_paper_id: int,
    references: Sequence[Mapping[str, Any]],
    *,
    source: str = "s2",
    db: Database | None = None,
) -> int:
    """批量写入某篇文献的参考文献。``references`` 每项含 doi/pmid/title。"""
    database = _db(db)
    count = 0
    with database.transaction() as conn:
        for ref in references:
            doi = (ref.get("doi") or "").lower() or None
            pmid = str(ref.get("pmid") or "") or None
            local_id = None
            if doi:
                r = conn.execute("SELECT paper_id FROM papers WHERE doi = ?", (doi,)).fetchone()
                local_id = int(r["paper_id"]) if r else None
            if local_id is None and pmid:
                r = conn.execute("SELECT paper_id FROM papers WHERE pmid = ?", (pmid,)).fetchone()
                local_id = int(r["paper_id"]) if r else None
            external = doi or pmid or ref.get("s2_id") or ref.get("title")
            conn.execute(
                "INSERT OR IGNORE INTO citations"
                "(citing_paper_id, cited_paper_id, cited_external_id, cited_title, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    citing_paper_id,
                    local_id,
                    str(external)[:400] if external else None,
                    (ref.get("title") or "")[:400] or None,
                    source,
                ),
            )
            count += 1
    return count


def get_references(paper_id: int, *, db: Database | None = None) -> list[dict[str, Any]]:
    """某篇文献引用了谁（含已入库的本地记录）。"""
    rows = _db(db).query(
        "SELECT c.cited_paper_id, c.cited_external_id, c.cited_title, c.source, "
        "       p.title AS local_title, p.pub_year, p.journal, p.doi AS local_doi "
        "FROM citations c LEFT JOIN papers p ON p.paper_id = c.cited_paper_id "
        "WHERE c.citing_paper_id = ?",
        (paper_id,),
    )
    return [dict(r) for r in rows]


def get_citing_papers(paper_id: int, *, db: Database | None = None) -> list[dict[str, Any]]:
    """谁引用了这篇文献（本地库范围内）。"""
    rows = _db(db).query(
        "SELECT p.paper_id, p.title, p.pub_year, p.journal, p.doi, c.source "
        "FROM citations c JOIN papers p ON p.paper_id = c.citing_paper_id "
        "WHERE c.cited_paper_id = ?",
        (paper_id,),
    )
    return [dict(r) for r in rows]

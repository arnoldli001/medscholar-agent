"""仓储层共享的内部助手：数据库句柄、FTS 影子表同步、日志器与 papers 过滤 SQL。

这里只放被多个业务边界复用的私有零件：``_db`` 几乎每个仓储函数都要用；
``_fts_sync`` / ``_fts_delete`` 同时服务文献写入与全文保存；``_FTS_COLUMNS``
同时决定 FTS 写入列序与 BM25 权重列序；``_FILTER_SQL`` / ``_build_filters``
同时服务文献列表与各路检索。

单独成模块，让这些零件只存在一份。如果留在 papers 或 search 里，
另一个模块就得反向 import 它，会产生循环依赖。
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Mapping, Sequence

from ..connect import Database, get_db
from ...textutil import segment_cjk

# 记录器沿用原 repo.py 的模块名：下游若按 "medscholar.db.repo" 过滤/采集日志，行为不变。
logger = logging.getLogger("medscholar.db.repo")

__all__ = [
    "_db",
    "_fts_sync",
    "_fts_delete",
    "_build_filters",
    "_FTS_COLUMNS",
    "_FILTER_SQL",
    "logger",
]

# papers_fts 的列顺序（务必与 db/schema.sql 保持一致）
_FTS_COLUMNS: tuple[str, ...] = (
    "title",
    "abstract",
    "authors",
    "journal",
    "mesh_terms",
    "keywords",
)

_FILTER_SQL = {
    "year_from": "p.pub_year >= ?",
    "year_to": "p.pub_year <= ?",
    "journal": "p.journal LIKE ?",
    "min_cited": "p.cited_by_count >= ?",
    "open_access": "p.is_open_access = 1",
    "has_abstract": "p.abstract IS NOT NULL AND p.abstract <> ''",
    "source": "p.source = ?",
}


def _db(db: Database | None) -> Database:
    return db or get_db()


def _fts_sync(
    conn: sqlite3.Connection,
    fts_table: str,
    columns: Sequence[str],
    paper_id: int,
    values: Mapping[str, str],
) -> None:
    """写入/更新一条 FTS 记录：先删后插，中文在写入前逐字切分。"""
    conn.execute(f"DELETE FROM {fts_table} WHERE paper_id = ?", (paper_id,))
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"INSERT INTO {fts_table}(paper_id, {', '.join(columns)}) "
        f"VALUES (?, {placeholders})",
        (paper_id, *[segment_cjk(values.get(col, "")) for col in columns]),
    )


def _fts_delete(conn: sqlite3.Connection, fts_table: str, paper_id: int) -> None:
    conn.execute(f"DELETE FROM {fts_table} WHERE paper_id = ?", (paper_id,))


def _build_filters(filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for key, sql in _FILTER_SQL.items():
        value = filters.get(key)
        if value is None or value is False or value == "":
            continue
        if key == "journal":
            clauses.append(sql)
            params.append(f"%{value}%")
        elif key == "open_access":
            clauses.append(sql)
        else:
            clauses.append(sql)
            params.append(value)
    sources = filters.get("sources")
    if sources:
        marks = ",".join("?" for _ in sources)
        clauses.append(f"p.source IN ({marks})")
        params.extend(list(sources))
    return (" AND ".join(clauses), params)

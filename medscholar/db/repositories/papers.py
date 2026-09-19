"""文献主表（papers）的读写：入库/去重富化、取回、分页列举、统计与删除。

单独拆出来是因为它是最底层的数据边界：混合检索要读它的元数据过滤、向量与全文回填
要读它挑候选、课题要按它分页。把它放底层，兄弟模块可以依赖它，而它不依赖任何兄弟模块。

## 关于 ``embed=True``：为什么改成"注入钩子"

入库时顺带嵌入是**便利**，不是数据层的职责。原来它直接
``from ...embedding.pipeline import embed_paper``，于是产生了
``db.repositories.papers ↔ embedding.pipeline`` 的**循环依赖**
（pipeline 又要 import 数据层来读文献），依赖图里出现了环。

这里改成依赖倒置：数据层只留一个空的钩子，由嵌入层在导入时注册自己
（``medscholar/embedding/__init__.py`` → :func:`register_embed_hooks`）。
数据层因此**完全不认识嵌入层**，环被打破，而 ``embed=True`` 的调用方式不变。

钩子没注册却传了 ``embed=True`` 时会**明确报错**而不是静默跳过 ——
"以为嵌入了其实没有"会让向量检索悄悄少掉一批文献，比报错难查得多。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..connect import Database
from ...models import Paper
from ...textutil import title_fingerprint
from ._common import _FTS_COLUMNS, _build_filters, _db, _fts_delete, _fts_sync, logger

__all__ = [
    "insert_paper",
    "insert_papers",
    "get_paper",
    "get_papers_by_ids",
    "find_paper_id",
    "count_papers",
    "list_papers",
    "delete_papers",
    "register_embed_hooks",
]

# ---------------------------------------------------------------------------
# 嵌入钩子（依赖倒置点）
# ---------------------------------------------------------------------------

#: (paper_id, database) -> None：单篇嵌入
_EMBED_ONE: Callable[[int, Database], Any] | None = None
#: (ids, database) -> report
_EMBED_MANY: Callable[[Sequence[int], Database], Any] | None = None


def register_embed_hooks(
    embed_one: Callable[[int, Database], Any],
    embed_many: Callable[[Sequence[int], Database], Any],
) -> None:
    """由嵌入层注册"入库后自动嵌入"的实现。

    只在 :mod:`medscholar.embedding` 导入时调用一次。这样数据层不必 import 嵌入层，
    依赖方向从"数据层 → 嵌入层 → 数据层"的环变成"嵌入层 → 数据层"的单向依赖。
    """
    global _EMBED_ONE, _EMBED_MANY
    _EMBED_ONE, _EMBED_MANY = embed_one, embed_many


def _require_hooks() -> tuple[
    Callable[[int, Database], Any], Callable[[Sequence[int], Database], Any]
]:
    if _EMBED_ONE is None or _EMBED_MANY is None:
        raise RuntimeError(
            "调用了 embed=True，但嵌入层尚未注册钩子。\n"
            "    修复方式：在调用前 import medscholar.embedding（它会自动注册），\n"
            "    或者不要传 embed=True，改为自行调用 embedding.run_embedding_pipeline()。\n"
            "    之所以报错而不是静默跳过：静默跳过会让向量检索少掉一批文献，"
            "表现为「检索结果莫名变少」，比直接失败难查得多。"
        )
    return _EMBED_ONE, _EMBED_MANY


_PAPER_COLUMNS = (
    "paper_id, pmid, pmcid, doi, title, abstract, authors, journal, pub_year, "
    "source, source_id, mesh_terms, keywords, cited_by_count, is_open_access, "
    "full_text_path, full_text_url, url, volume, issue, pages, publication_type, "
    "language, note, created_at, updated_at"
)


def _decode_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _row_to_paper(row: Mapping[str, Any]) -> Paper:
    return Paper.from_row(row)


def _merge_into(existing: Paper, incoming: Paper) -> dict[str, Any]:
    """把新抓到的记录合并进已有记录：补空字段 + 取更优值。"""

    def richer(a: str, b: str) -> str:
        return b if len(b or "") > len(a or "") else (a or "")

    def union(a: Iterable[str], b: Iterable[str]) -> list[str]:
        out = list(a)
        seen = {x.lower() for x in out}
        for item in b:
            if item.lower() not in seen:
                out.append(item)
                seen.add(item.lower())
        return out

    return {
        "pmid": existing.pmid or incoming.pmid,
        "pmcid": existing.pmcid or incoming.pmcid,
        "doi": existing.doi or incoming.doi,
        "title": richer(existing.title, incoming.title) or existing.title,
        "abstract": richer(existing.abstract, incoming.abstract) or None,
        "authors": json.dumps(union(existing.authors, incoming.authors), ensure_ascii=False),
        "journal": existing.journal or incoming.journal or None,
        "pub_year": existing.pub_year or incoming.pub_year,
        "mesh_terms": json.dumps(
            union(existing.mesh_terms, incoming.mesh_terms), ensure_ascii=False
        ),
        "keywords": json.dumps(union(existing.keywords, incoming.keywords), ensure_ascii=False),
        "cited_by_count": max(existing.cited_by_count, incoming.cited_by_count),
        "is_open_access": 1 if (existing.is_open_access or incoming.is_open_access) else 0,
        "full_text_url": existing.full_text_url or incoming.full_text_url or None,
        "full_text_path": existing.full_text_path or None,
        "url": existing.url or incoming.url or None,
        "volume": existing.volume or incoming.volume or None,
        "issue": existing.issue or incoming.issue or None,
        "pages": existing.pages or incoming.pages or None,
        "publication_type": existing.publication_type or incoming.publication_type or None,
        "language": existing.language or incoming.language or None,
    }


def _find_existing_id(conn: sqlite3.Connection, paper: Paper) -> int | None:
    """按 DOI → PMID → 标题指纹 的顺序查找已存在的文献。"""
    if paper.doi:
        row = conn.execute("SELECT paper_id FROM papers WHERE doi = ?", (paper.doi,)).fetchone()
        if row:
            return int(row["paper_id"])
    if paper.pmid:
        row = conn.execute("SELECT paper_id FROM papers WHERE pmid = ?", (paper.pmid,)).fetchone()
        if row:
            return int(row["paper_id"])
    fp = title_fingerprint(paper.title)
    if fp:
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE title_key = ? LIMIT 1", (fp,)
        ).fetchone()
        if row:
            return int(row["paper_id"])
    return None


def insert_paper(
    paper: Paper, *, db: Database | None = None, embed: bool = False
) -> tuple[int, bool]:
    """插入或**富化**一篇文献。

    Returns:
        ``(paper_id, created)``；``created=False`` 表示命中已有记录并被合并更新。
    """
    database = _db(db)
    fp = title_fingerprint(paper.title)

    with database.transaction() as conn:
        existing_id = _find_existing_id(conn, paper)

        if existing_id is None:
            cur = conn.execute(
                "INSERT INTO papers (pmid, pmcid, doi, title, abstract, authors, journal, "
                "pub_year, source, source_id, mesh_terms, keywords, cited_by_count, "
                "is_open_access, full_text_path, full_text_url, url, volume, issue, pages, "
                "publication_type, language, title_key) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    paper.pmid,
                    paper.pmcid,
                    paper.doi,
                    paper.title,
                    paper.abstract or None,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.journal or None,
                    paper.pub_year,
                    paper.source,
                    paper.source_id or None,
                    json.dumps(paper.mesh_terms, ensure_ascii=False),
                    json.dumps(paper.keywords, ensure_ascii=False),
                    paper.cited_by_count,
                    1 if paper.is_open_access else 0,
                    paper.full_text_path or None,
                    paper.full_text_url or None,
                    paper.url or None,
                    paper.volume or None,
                    paper.issue or None,
                    paper.pages or None,
                    paper.publication_type or None,
                    paper.language or None,
                    fp or None,
                ),
            )
            paper_id = int(cur.lastrowid or 0)
            created = True
            row = conn.execute(
                f"SELECT {_PAPER_COLUMNS} FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        else:
            paper_id = existing_id
            created = False
            row = conn.execute(
                f"SELECT {_PAPER_COLUMNS} FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            merged = _merge_into(_row_to_paper(row), paper)
            if paper.source and paper.source not in str(row["source"] or ""):
                merged["source"] = row["source"]
            assignments = ", ".join(f"{k} = ?" for k in merged)
            conn.execute(
                f"UPDATE papers SET {assignments}, updated_at = datetime('now') "
                f"WHERE paper_id = ?",
                (*merged.values(), paper_id),
            )
            row = conn.execute(
                f"SELECT {_PAPER_COLUMNS} FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()

        _fts_sync(
            conn,
            "papers_fts",
            _FTS_COLUMNS,
            paper_id,
            {
                "title": row["title"] or "",
                "abstract": row["abstract"] or "",
                "authors": " ".join(_decode_list(row["authors"])),
                "journal": row["journal"] or "",
                "mesh_terms": " ".join(_decode_list(row["mesh_terms"])),
                "keywords": " ".join(_decode_list(row["keywords"])),
            },
        )

    if embed:
        embed_one, _ = _require_hooks()
        try:
            embed_one(paper_id, database)
        except Exception as exc:  # 嵌入失败绝不应阻断入库
            logger.warning("文献 %s 自动嵌入失败：%s", paper_id, exc)

    return paper_id, created


def insert_papers(
    papers: Iterable[Paper], *, db: Database | None = None, embed: bool = False
) -> dict[str, Any]:
    """批量入库。返回 ``{new, updated, ids, new_ids, errors}``。"""
    database = _db(db)
    new_ids: list[int] = []
    all_ids: list[int] = []
    updated = 0
    errors: list[str] = []

    for paper in papers:
        if not paper.title:
            continue
        try:
            paper_id, created = insert_paper(paper, db=database)
        except sqlite3.Error as exc:
            errors.append(f"{paper.title[:60]}: {exc}")
            continue
        all_ids.append(paper_id)
        if created:
            new_ids.append(paper_id)
        else:
            updated += 1

    if embed and new_ids:
        _, embed_many = _require_hooks()
        report = embed_many(new_ids, database)
        embedded = report.to_dict()
    else:
        embedded = {}

    return {
        "new": len(new_ids),
        "updated": updated,
        "total": len(all_ids),
        "ids": all_ids,
        "new_ids": new_ids,
        "errors": errors,
        "embedded": embedded,
    }


def get_paper(paper_id: int, *, db: Database | None = None) -> Paper | None:
    row = _db(db).query_one(
        f"SELECT {_PAPER_COLUMNS} FROM papers WHERE paper_id = ?", (paper_id,)
    )
    return _row_to_paper(row) if row else None


def get_papers_by_ids(ids: Sequence[int], *, db: Database | None = None) -> dict[int, Paper]:
    if not ids:
        return {}
    database = _db(db)
    out: dict[int, Paper] = {}
    # 分批避免超过 SQLite 变量上限（默认 999）
    for start in range(0, len(ids), 500):
        chunk = list(ids[start : start + 500])
        marks = ",".join("?" for _ in chunk)
        for row in database.query(
            f"SELECT {_PAPER_COLUMNS} FROM papers WHERE paper_id IN ({marks})", chunk
        ):
            paper = _row_to_paper(row)
            out[paper.paper_id] = paper  # type: ignore[index]
    return out


def find_paper_id(
    *, doi: str | None = None, pmid: str | None = None, title: str | None = None,
    db: Database | None = None,
) -> int | None:
    database = _db(db)
    if doi:
        row = database.query_one("SELECT paper_id FROM papers WHERE doi = ?", (doi.lower(),))
        if row:
            return int(row["paper_id"])
    if pmid:
        row = database.query_one("SELECT paper_id FROM papers WHERE pmid = ?", (str(pmid),))
        if row:
            return int(row["paper_id"])
    if title:
        fp = title_fingerprint(title)
        if fp:
            row = database.query_one(
                "SELECT paper_id FROM papers WHERE title_key = ? LIMIT 1", (fp,)
            )
            if row:
                return int(row["paper_id"])
    return None


def count_papers(
    *,
    filters: Mapping[str, Any] | None = None,
    project_id: int | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    where, params = _build_filters(filters)
    join = ""
    if project_id is not None:
        join = " JOIN project_papers pp ON pp.paper_id = p.paper_id AND pp.project_id = ?"
        params = [project_id, *params]
    sql = f"SELECT COUNT(*) FROM papers p{join}"
    if where:
        sql += f" WHERE {where}"
    return int(database.scalar(sql, params, default=0))


def list_papers(
    *,
    filters: Mapping[str, Any] | None = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "created_desc",
    project_id: int | None = None,
    db: Database | None = None,
) -> list[Paper]:
    """按条件分页列出文献。"""
    database = _db(db)
    where, params = _build_filters(filters)
    join = ""
    if project_id is not None:
        join = " JOIN project_papers pp ON pp.paper_id = p.paper_id AND pp.project_id = ?"
        params = [project_id, *params]

    order = {
        "created_desc": "p.created_at DESC, p.paper_id DESC",
        "created_asc": "p.created_at ASC, p.paper_id ASC",
        "year_desc": "p.pub_year DESC NULLS LAST, p.paper_id DESC",
        "year_asc": "p.pub_year ASC NULLS LAST, p.paper_id ASC",
        "cited_desc": "p.cited_by_count DESC, p.paper_id DESC",
        "title_asc": "p.title ASC",
    }.get(order_by, "p.created_at DESC, p.paper_id DESC")

    cols = ", ".join(f"p.{c.strip()}" for c in _PAPER_COLUMNS.split(","))
    sql = f"SELECT {cols} FROM papers p{join}"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
    params = [*params, limit, offset]
    return [_row_to_paper(r) for r in database.query(sql, params)]


def delete_papers(ids: Sequence[int], *, db: Database | None = None) -> int:
    """删除文献（同时清理 FTS 与向量索引）。"""
    if not ids:
        return 0
    database = _db(db)
    removed = 0
    with database.transaction() as conn:
        for start in range(0, len(ids), 500):
            chunk = list(ids[start : start + 500])
            marks = ",".join("?" for _ in chunk)
            for pid in chunk:
                _fts_delete(conn, "papers_fts", pid)
                _fts_delete(conn, "fulltext_fts", pid)
            conn.execute(f"DELETE FROM paper_embeddings WHERE paper_id IN ({marks})", chunk)
            cur = conn.execute(f"DELETE FROM papers WHERE paper_id IN ({marks})", chunk)
            removed += cur.rowcount or 0
    return removed

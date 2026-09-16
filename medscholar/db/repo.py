"""数据库仓储层。

对外暴露需求文档要求的核心函数：
    ``insert_paper()`` / ``search_fts()`` / ``search_vector()`` / ``hybrid_search()``

检索策略（需求 2.4 节）：
    1. FTS5 BM25 关键词召回（标题权重最高，MeSH/关键词次之）
    2. sqlite-vec KNN 语义召回（向量已 L2 归一化，距离与余弦距离单调一致）
    3. RRF 融合（默认 k=60）
    4. 元数据过滤（年份 / 期刊 / 被引 / 开放获取 / 课题）
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import struct
from typing import Any, Iterable, Mapping, Sequence

from ..config import get_config
from ..models import Paper, ScoredPaper, SearchLogEntry
from ..textutil import build_match_query, segment_cjk, title_fingerprint
from .connect import Database, get_db

logger = logging.getLogger(__name__)

__all__ = [
    "insert_paper",
    "insert_papers",
    "get_paper",
    "get_papers_by_ids",
    "find_paper_id",
    "list_papers",
    "delete_papers",
    "count_papers",
    "search_fts",
    "search_vector",
    "hybrid_search",
    "rrf_fuse",
    "store_embedding",
    "store_embeddings",
    "papers_missing_embeddings",
    "clear_embeddings",
    "save_fulltext",
    "get_fulltext",
    "search_fulltext",
    "add_citation",
    "add_citations",
    "get_references",
    "get_citing_papers",
    "log_search",
    "recent_searches",
    "create_project",
    "list_projects",
    "get_project",
    "update_project",
    "delete_project",
    "add_papers_to_project",
    "remove_paper_from_project",
    "list_project_papers",
    "create_session",
    "list_sessions",
    "get_session",
    "rename_session",
    "delete_session",
    "add_message",
    "list_messages",
    "save_artifact",
    "get_artifact",
    "list_artifacts",
    "delete_artifact",
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

_PAPER_COLUMNS = (
    "paper_id, pmid, pmcid, doi, title, abstract, authors, journal, pub_year, "
    "source, source_id, mesh_terms, keywords, cited_by_count, is_open_access, "
    "full_text_path, full_text_url, url, volume, issue, pages, publication_type, "
    "language, note, created_at, updated_at"
)


# ============================================================== 内部工具
def _db(db: Database | None) -> Database:
    return db or get_db()


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


def serialize_vector(vector: Sequence[float]) -> bytes:
    """float32 小端序列化（sqlite-vec 要求）。"""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """L2 归一化，使 L2 距离与余弦距离单调一致，两种向量后端结果可对齐。"""
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm <= 1e-12:
        return [0.0] * len(vector)
    return [float(v) / norm for v in vector]


# ============================================================ 文献写入
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
        from ..embedding.pipeline import embed_paper

        try:
            embed_paper(paper_id, db=database)
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
        from ..embedding.pipeline import run_embedding_pipeline

        report = run_embedding_pipeline(ids=new_ids, db=database)
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


_FILTER_SQL = {
    "year_from": "p.pub_year >= ?",
    "year_to": "p.pub_year <= ?",
    "journal": "p.journal LIKE ?",
    "min_cited": "p.cited_by_count >= ?",
    "open_access": "p.is_open_access = 1",
    "has_abstract": "p.abstract IS NOT NULL AND p.abstract <> ''",
    "source": "p.source = ?",
}


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


# ============================================================== 检索
def search_fts(
    query: str,
    *,
    limit: int = 100,
    filters: Mapping[str, Any] | None = None,
    db: Database | None = None,
    table: str = "papers_fts",
) -> list[tuple[int, float]]:
    """FTS5 BM25 关键词检索。返回 ``[(paper_id, bm25_score), ...]``，分数越小越相关。

    中文检索采用**逐级放宽**策略，因为单一级别无法同时兼顾精度与召回：

    1. ``phrase`` —— 连续子串精确匹配（精度最高，但「治疗卒中后抑郁」匹配不到
       「治疗**脑**卒中后抑郁」）；
    2. ``bigram`` —— 重叠二元组取 AND（对词序调换、中间插入修饰语更宽容）；
    3. ``or``     —— 二元组取 OR（召回兜底，靠 BM25 排序压住噪声）。

    任何一级返回非空结果就停止，因此常见查询仍只跑一次 FTS（毫秒级）。
    """
    if not query or not query.strip():
        return []
    database = _db(db)
    cfg = get_config()
    weights = cfg.retrieval.bm25_weights
    weight_args = [0.0] + [float(weights.get(col, 1.0)) for col in _FTS_COLUMNS]
    weight_sql = ", ".join(f"{w:.3f}" for w in weight_args)

    where, params = _build_filters(filters)
    sql = (
        f"SELECT f.paper_id AS paper_id, bm25({table}, {weight_sql}) AS score "
        f"FROM {table} f JOIN papers p ON p.paper_id = f.paper_id "
        f"WHERE {table} MATCH ?"
    )
    if where:
        sql += f" AND {where}"
    sql += " ORDER BY score LIMIT ?"

    strategies: list[str] = []
    for candidate in (
        build_match_query(query, cjk="phrase"),
        build_match_query(query, cjk="bigram"),
        build_match_query(query, mode="or", cjk="bigram"),
    ):
        if candidate and candidate not in strategies:
            strategies.append(candidate)

    for index, expression in enumerate(strategies):
        try:
            rows = database.query(sql, [expression, *params, limit])
        except sqlite3.OperationalError as exc:
            logger.warning("FTS 检索失败（%s，策略 %d）：%s", query, index + 1, exc)
            continue
        if rows:
            if index:
                logger.debug("FTS 采用第 %d 级放宽策略命中 %d 条：%s", index + 1, len(rows), query)
            return [(int(r["paper_id"]), float(r["score"])) for r in rows]
    return []


def search_fulltext(
    query: str, *, limit: int = 50, db: Database | None = None
) -> list[tuple[int, float]]:
    """在已入库的开放获取全文中检索。"""
    return search_fts(query, limit=limit, db=db, table="fulltext_fts")


def search_vector(
    embedding: Sequence[float],
    *,
    limit: int = 100,
    filters: Mapping[str, Any] | None = None,
    db: Database | None = None,
) -> list[tuple[int, float]]:
    """向量 KNN 检索。返回 ``[(paper_id, distance), ...]``，距离越小越相似。"""
    if not embedding:
        return []
    database = _db(db)
    vector = normalize_vector(embedding)
    dim = len(vector)

    if database.vec_available:
        try:
            rows = database.query(
                "SELECT paper_id, distance FROM paper_embeddings "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (serialize_vector(vector), limit),
            )
        except sqlite3.Error as exc:
            logger.warning("vec0 KNN 失败，回退纯 Python：%s", exc)
            rows = []
        else:
            results = [(int(r["paper_id"]), float(r["distance"])) for r in rows]
            return _apply_vector_filters(results, filters, database, limit)

    # ---- 纯 Python 回退：全量取出后做点积（向量已归一化，L2² = 2 - 2·cos）
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore

    rows = database.query(
        "SELECT paper_id, embedding, dim FROM paper_embeddings WHERE dim = ?", (dim,)
    )
    if not rows:
        return []

    if np is not None:
        ids = [int(r["paper_id"]) for r in rows]
        mat = np.frombuffer(
            b"".join(bytes(r["embedding"]) for r in rows), dtype="<f4"
        ).reshape(len(ids), dim)
        q = np.asarray(vector, dtype="<f4")
        sims = mat @ q  # 归一化向量的点积即余弦相似度
        order = np.argsort(-sims)[: max(limit * 4, limit)]
        scored = [(ids[i], float(math.sqrt(max(0.0, 2.0 - 2.0 * float(sims[i]))))) for i in order]
    else:
        q = vector
        scored = []
        for r in rows:
            vec = deserialize_vector(r["embedding"])
            if len(vec) != dim:
                continue
            dot = 0.0
            for a, b in zip(vec, q):
                dot += a * b
            scored.append((int(r["paper_id"]), math.sqrt(max(0.0, 2.0 - 2.0 * dot))))
        scored.sort(key=lambda x: x[1])
        scored = scored[: max(limit * 4, limit)]

    return _apply_vector_filters(scored, filters, database, limit)


def _apply_vector_filters(
    results: list[tuple[int, float]],
    filters: Mapping[str, Any] | None,
    database: Database,
    limit: int,
) -> list[tuple[int, float]]:
    """向量召回后的元数据过滤（KNN 子查询不能直接与过滤条件共存，故在 Python 侧完成）。"""
    if not filters:
        return results[:limit]
    where, params = _build_filters(filters)
    if not where:
        return results[:limit]
    ids = [pid for pid, _ in results]
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    allowed = {
        int(r["paper_id"])
        for r in database.query(
            f"SELECT p.paper_id FROM papers p WHERE p.paper_id IN ({marks}) AND {where}",
            [*ids, *params],
        )
    }
    return [(pid, dist) for pid, dist in results if pid in allowed][:limit]


def rrf_fuse(
    rankings: Sequence[Sequence[tuple[int, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：``score = Σ w_i / (k + rank_i)``。

    >>> rrf_fuse([[(1, -3.0), (2, -2.0)], [(2, 0.1), (3, 0.2)]], k=60)
    [(2, 0.03278688524590164), (1, 0.01639344262295082), (3, 0.016129032258064516)]
    """
    scores: dict[int, float] = {}
    for idx, ranking in enumerate(rankings):
        weight = float(weights[idx]) if weights and idx < len(weights) else 1.0
        for rank, (paper_id, _score) in enumerate(ranking, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_search(
    query: str,
    *,
    embedding: Sequence[float] | None = None,
    top_k: int | None = None,
    filters: Mapping[str, Any] | None = None,
    db: Database | None = None,
    fts_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list[ScoredPaper]:
    """BM25 + 向量 + RRF 融合的混合检索。

    ``embedding`` 为 ``None`` 时只走关键词（例如模型未就绪或离线场景）。
    """
    database = _db(db)
    cfg = get_config()
    top_k = top_k or cfg.retrieval.top_k
    fts_limit = max(cfg.retrieval.fts_candidates, top_k * 3)
    vec_limit = max(cfg.retrieval.vector_candidates, top_k * 3)

    fts_hits = search_fts(query, limit=fts_limit, filters=filters, db=database)
    vec_hits = (
        search_vector(embedding, limit=vec_limit, filters=filters, db=database)
        if embedding
        else []
    )

    if not fts_hits and not vec_hits:
        return []

    fused = rrf_fuse([fts_hits, vec_hits], k=cfg.retrieval.rrf_k, weights=[fts_weight, vector_weight])

    fts_rank = {pid: i for i, (pid, _) in enumerate(fts_hits, start=1)}
    fts_score = {pid: s for pid, s in fts_hits}
    vec_rank = {pid: i for i, (pid, _) in enumerate(vec_hits, start=1)}
    vec_dist = {pid: d for pid, d in vec_hits}

    fused = fused[:top_k]
    papers = get_papers_by_ids([pid for pid, _ in fused], db=database)

    results: list[ScoredPaper] = []
    for paper_id, score in fused:
        paper = papers.get(paper_id)
        if paper is None:
            continue
        if score < cfg.retrieval.min_score:
            continue
        matched = []
        if paper_id in fts_rank:
            matched.append("bm25")
        if paper_id in vec_rank:
            matched.append("vector")
        results.append(
            ScoredPaper(
                paper=paper,
                score=score,
                fts_rank=fts_rank.get(paper_id),
                vector_rank=vec_rank.get(paper_id),
                fts_score=fts_score.get(paper_id),
                vector_distance=vec_dist.get(paper_id),
                matched_by="+".join(matched),
            )
        )
    return results


# ============================================================ 向量嵌入
def store_embedding(
    paper_id: int,
    vector: Sequence[float],
    *,
    db: Database | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """写入/覆盖一篇文献的嵌入向量（自动 L2 归一化）。"""
    database = _db(db)
    vec = normalize_vector(vector)
    payload = serialize_vector(vec)

    def _write(c: sqlite3.Connection) -> None:
        c.execute("DELETE FROM paper_embeddings WHERE paper_id = ?", (paper_id,))
        if database.vec_available:
            c.execute(
                "INSERT INTO paper_embeddings(paper_id, embedding) VALUES (?, ?)",
                (paper_id, payload),
            )
        else:
            c.execute(
                "INSERT INTO paper_embeddings(paper_id, dim, embedding) VALUES (?, ?, ?)",
                (paper_id, len(vec), payload),
            )

    if conn is not None:
        _write(conn)
    else:
        with database.transaction() as c:
            _write(c)


def store_embeddings(
    items: Sequence[tuple[int, Sequence[float]]], *, db: Database | None = None
) -> int:
    """批量写入嵌入。"""
    if not items:
        return 0
    database = _db(db)
    with database.transaction() as conn:
        for paper_id, vector in items:
            vec = normalize_vector(vector)
            conn.execute("DELETE FROM paper_embeddings WHERE paper_id = ?", (paper_id,))
            if database.vec_available:
                conn.execute(
                    "INSERT INTO paper_embeddings(paper_id, embedding) VALUES (?, ?)",
                    (paper_id, serialize_vector(vec)),
                )
            else:
                conn.execute(
                    "INSERT INTO paper_embeddings(paper_id, dim, embedding) VALUES (?, ?, ?)",
                    (paper_id, len(vec), serialize_vector(vec)),
                )
    return len(items)


def papers_missing_embeddings(
    *, limit: int = 500, ids: Sequence[int] | None = None, db: Database | None = None
) -> list[int]:
    """找出尚未生成嵌入的文献（增量嵌入管道用）。"""
    database = _db(db)
    sql = (
        "SELECT p.paper_id FROM papers p "
        "LEFT JOIN paper_embeddings e ON e.paper_id = p.paper_id "
        "WHERE e.paper_id IS NULL"
    )
    params: list[Any] = []
    if ids:
        chunk_ids = list(ids)[:900]
        marks = ",".join("?" for _ in chunk_ids)
        sql += f" AND p.paper_id IN ({marks})"
        params.extend(chunk_ids)
    sql += " ORDER BY p.paper_id LIMIT ?"
    params.append(limit)
    return [int(r["paper_id"]) for r in database.query(sql, params)]


def clear_embeddings(*, db: Database | None = None) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM paper_embeddings")
    return cur.rowcount or 0


# ============================================================== 全文
def save_fulltext(
    paper_id: int,
    content: str,
    *,
    origin: str = "",
    source_url: str = "",
    db: Database | None = None,
) -> None:
    """保存开放获取全文并建立 FTS 索引。"""
    if not content or not content.strip():
        return
    database = _db(db)
    content = content.strip()
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO paper_fulltext(paper_id, content, char_count, origin, source_url, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(paper_id) DO UPDATE SET content = excluded.content, "
            "char_count = excluded.char_count, origin = excluded.origin, "
            "source_url = excluded.source_url, fetched_at = datetime('now')",
            (paper_id, content, len(content), origin, source_url),
        )
        _fts_sync(conn, "fulltext_fts", ("content",), paper_id, {"content": content})


def get_fulltext(paper_id: int, *, db: Database | None = None) -> str:
    row = _db(db).query_one(
        "SELECT content FROM paper_fulltext WHERE paper_id = ?", (paper_id,)
    )
    return str(row["content"]) if row else ""


def fulltext_candidates(
    *, limit: int = 20, only_missing: bool = True, db: Database | None = None
) -> list[Paper]:
    """挑出「最可能抓到开放获取全文」的文献，供全文回填使用。

    排序策略直接决定回填成功率：

    1. **有 PMCID 的排最前** —— Europe PMC / PMC 的 JATS 全文接口稳定可用，实测成功率最高；
    2. 其次是只有 OA 链接的 —— 要去出版商站点下 PDF，实测常被 403/405 拒绝
       （ScienceDirect、部分机构仓储会挡自动化请求），属于正常现象，不该硬闯；
    3. 同类内按被引数降序，优先补齐重要的文献。

    另外会跳过两类：已有全文的、以及**已确认永久取不到**的
    （见 :func:`record_fulltext_attempt`）。因此本操作可以**反复执行**，
    不会把没有正文的会议摘要之类反复重试。
    """
    database = _db(db)
    sql = (
        "SELECT p.paper_id FROM papers p "
        "WHERE (p.is_open_access = 1 OR (p.pmcid IS NOT NULL AND p.pmcid <> ''))"
    )
    if only_missing:
        sql += " AND p.paper_id NOT IN (SELECT paper_id FROM paper_fulltext)"
        sql += (
            " AND p.paper_id NOT IN ("
            "   SELECT paper_id FROM fulltext_attempts WHERE permanent = 1)"
        )
    sql += (
        " ORDER BY (CASE WHEN p.pmcid IS NOT NULL AND p.pmcid <> '' THEN 0 ELSE 1 END),"
        "          p.cited_by_count DESC, p.paper_id ASC"
        " LIMIT ?"
    )
    ids = [int(r["paper_id"]) for r in database.query(sql, (limit,))]
    return [p for p in (get_paper(i, db=database) for i in ids) if p is not None]


def record_fulltext_attempt(
    paper_id: int, error: str, *, permanent: bool, db: Database | None = None
) -> None:
    """记录一次全文抓取失败，供下次跳过。

    ``permanent=True`` 表示"确定取不到"（文献本身没有正文、出版商长期拒绝、
    非开放获取等），下次 :func:`fulltext_candidates` 会直接跳过；
    网络超时、5xx 这类**可重试**的错误则不标记，留待下次再试。
    """
    database = _db(db)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO fulltext_attempts(paper_id, attempts, last_error, permanent, checked_at) "
            "VALUES (?, 1, ?, ?, datetime('now')) "
            "ON CONFLICT(paper_id) DO UPDATE SET "
            "  attempts = attempts + 1, last_error = excluded.last_error, "
            "  permanent = excluded.permanent, checked_at = datetime('now')",
            (paper_id, (error or "")[:500], 1 if permanent else 0),
        )


def clear_fulltext_attempts(*, db: Database | None = None) -> int:
    """清空尝试记录（用户想强行重试全部时使用）。"""
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM fulltext_attempts")
    return cur.rowcount or 0


def fulltext_failure_stats(*, db: Database | None = None) -> dict[str, int]:
    """按原因归类统计已确认取不到的文献数。"""
    rows = _db(db).query(
        "SELECT permanent, COUNT(*) AS n FROM fulltext_attempts GROUP BY permanent"
    )
    out = {"permanent": 0, "retryable": 0}
    for row in rows:
        out["permanent" if row["permanent"] else "retryable"] = int(row["n"])
    return out


# ============================================================== 引用关系
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


# ============================================================== 检索历史
def log_search(entry: SearchLogEntry, *, db: Database | None = None) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO search_logs(query, source, result_count, new_count, duration_ms, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry.query,
                entry.source,
                entry.result_count,
                entry.new_count,
                entry.duration_ms,
                entry.error or None,
            ),
        )
        return int(cur.lastrowid or 0)


def recent_searches(*, limit: int = 20, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT id, query, source, result_count, new_count, duration_ms, error, created_at "
        "FROM search_logs ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


# ============================================================== 课题管理
def create_project(
    name: str, *, description: str = "", keywords: Sequence[str] | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO projects(name, description, keywords) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET description = excluded.description, "
            "keywords = excluded.keywords, updated_at = datetime('now')",
            (name.strip(), description, json.dumps(list(keywords or []), ensure_ascii=False)),
        )
        row = conn.execute("SELECT id FROM projects WHERE name = ?", (name.strip(),)).fetchone()
        return int(row["id"])


def list_projects(*, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT pr.id, pr.name, pr.description, pr.keywords, pr.created_at, pr.updated_at, "
        "       (SELECT COUNT(*) FROM project_papers pp WHERE pp.project_id = pr.id) AS paper_count "
        "FROM projects pr ORDER BY pr.updated_at DESC, pr.id DESC"
    )
    out = []
    for r in rows:
        item = dict(r)
        item["keywords"] = _decode_list(item.get("keywords"))
        out.append(item)
    return out


def get_project(project_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
    if not row:
        return None
    item = dict(row)
    item["keywords"] = _decode_list(item.get("keywords"))
    return item


def update_project(
    project_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    keywords: Sequence[str] | None = None,
    db: Database | None = None,
) -> bool:
    sets, params = [], []
    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if keywords is not None:
        sets.append("keywords = ?")
        params.append(json.dumps(list(keywords), ensure_ascii=False))
    if not sets:
        return False
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            f"UPDATE projects SET {', '.join(sets)}, updated_at = datetime('now') WHERE id = ?",
            [*params, project_id],
        )
        return (cur.rowcount or 0) > 0


def delete_project(project_id: int, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return (cur.rowcount or 0) > 0


def add_papers_to_project(
    project_id: int, paper_ids: Sequence[int], *, note: str = "", db: Database | None = None
) -> int:
    if not paper_ids:
        return 0
    database = _db(db)
    added = 0
    with database.transaction() as conn:
        for pid in paper_ids:
            cur = conn.execute(
                "INSERT OR IGNORE INTO project_papers(project_id, paper_id, note) VALUES (?, ?, ?)",
                (project_id, pid, note),
            )
            added += cur.rowcount or 0
        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?", (project_id,)
        )
    return added


def remove_paper_from_project(
    project_id: int, paper_id: int, *, db: Database | None = None
) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "DELETE FROM project_papers WHERE project_id = ? AND paper_id = ?",
            (project_id, paper_id),
        )
        return (cur.rowcount or 0) > 0


def list_project_papers(
    project_id: int, *, limit: int = 200, db: Database | None = None
) -> list[Paper]:
    return list_papers(project_id=project_id, limit=limit, db=db)


# ============================================================== 会话与消息
def create_session(
    *, title: str = "新会话", project_id: int | None = None, topic: str = "",
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO chat_sessions(title, project_id, topic) VALUES (?, ?, ?)",
            (title[:200], project_id, topic),
        )
        return int(cur.lastrowid or 0)


def list_sessions(*, limit: int = 50, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT s.id, s.title, s.project_id, s.topic, s.created_at, s.updated_at, "
        "       (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS message_count "
        "FROM chat_sessions s ORDER BY s.updated_at DESC, s.id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def get_session(session_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
    return dict(row) if row else None


def rename_session(session_id: int, title: str, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title[:200], session_id),
        )
        return (cur.rowcount or 0) > 0


def delete_session(session_id: int, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return (cur.rowcount or 0) > 0


def add_message(
    session_id: int,
    role: str,
    content: str,
    *,
    meta: Mapping[str, Any] | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, meta) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(dict(meta or {}), ensure_ascii=False)),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,)
        )
        return int(cur.lastrowid or 0)


def list_messages(
    session_id: int, *, limit: int = 200, db: Database | None = None
) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT id, session_id, role, content, meta, created_at FROM chat_messages "
        "WHERE session_id = ? ORDER BY id LIMIT ?",
        (session_id, limit),
    )
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["meta"] = json.loads(item.get("meta") or "{}")
        except (TypeError, ValueError):
            item["meta"] = {}
        out.append(item)
    return out


# ============================================================== 运行记录
def upsert_run(
    run_id: str,
    *,
    session_id: int | None = None,
    topic: str = "",
    phase: str = "pending",
    status: str = "running",
    papers: int = 0,
    citations: int = 0,
    artifact_id: int | None = None,
    error: str = "",
    db: Database | None = None,
) -> None:
    """写入/更新一条运行记录（跨服务重启可追溯）。"""
    database = _db(db)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO agent_runs(run_id, session_id, topic, phase, status, papers, "
            "                       citations, artifact_id, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "  phase = excluded.phase, status = excluded.status, "
            "  papers = excluded.papers, citations = excluded.citations, "
            "  artifact_id = COALESCE(excluded.artifact_id, agent_runs.artifact_id), "
            "  error = excluded.error, updated_at = datetime('now')",
            (run_id, session_id, topic[:500], phase, status, papers, citations,
             artifact_id, (error or "")[:1000]),
        )


def get_run(run_id: str, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
    return dict(row) if row else None


def list_runs(*, limit: int = 30, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT r.*, a.title AS artifact_title, a.fmt AS artifact_fmt "
        "FROM agent_runs r LEFT JOIN artifacts a ON a.id = r.artifact_id "
        "ORDER BY r.created_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def mark_interrupted_runs(*, db: Database | None = None) -> int:
    """把上次退出时仍在进行中的运行标记为「已中断」。

    服务启动时调用。这样用户刷新页面就能看到"那次运行被中断了"，
    而不是对着一个永远等不到内容的草稿页签发呆。
    """
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "UPDATE agent_runs SET status = 'interrupted', "
            "  error = CASE WHEN error = '' THEN '服务在运行过程中重启，本次运行已中断' ELSE error END, "
            "  updated_at = datetime('now') "
            "WHERE status IN ('running', 'awaiting_approval')"
        )
    return cur.rowcount or 0


def delete_run(run_id: str, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        conn.execute("DELETE FROM run_steps WHERE run_id = ?", (run_id,))
        cur = conn.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))
        return (cur.rowcount or 0) > 0


# ================================================== 阶段快照（断点续跑）
def save_run_step(
    run_id: str,
    phase: str,
    payload: Mapping[str, Any],
    *,
    db: Database | None = None,
) -> None:
    """保存某个阶段结束时的状态快照（覆盖同一阶段的上一次）。"""
    database = _db(db)
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO run_steps(run_id, phase, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, phase) DO UPDATE SET "
            "  payload = excluded.payload, created_at = datetime('now')",
            (run_id, phase, blob),
        )


def get_run_step(
    run_id: str, phase: str, *, db: Database | None = None
) -> dict[str, Any] | None:
    row = _db(db).query_one(
        "SELECT phase, payload, created_at FROM run_steps WHERE run_id = ? AND phase = ?",
        (run_id, phase),
    )
    if not row:
        return None
    try:
        data = json.loads(row["payload"] or "{}")
    except json.JSONDecodeError:
        data = {}
    return {"phase": row["phase"], "data": data, "created_at": row["created_at"]}


def list_run_steps(run_id: str, *, db: Database | None = None) -> list[dict[str, Any]]:
    """按阶段顺序返回所有快照。"""
    rows = _db(db).query(
        "SELECT phase, payload, created_at FROM run_steps WHERE run_id = ?",
        (run_id,),
    )
    order = ["plan", "execute", "reflect", "synthesize", "review"]
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            data = {}
        out.append({"phase": row["phase"], "data": data, "created_at": row["created_at"]})
    out.sort(key=lambda item: order.index(item["phase"]) if item["phase"] in order else 99)
    return out


def latest_run_for_session(session_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one(
        "SELECT * FROM agent_runs WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    )
    return dict(row) if row else None


def run_step_phases(
    run_ids: Sequence[str], *, db: Database | None = None
) -> dict[str, list[str]]:
    """一次查出多份运行各自的已完成阶段（按流水线顺序）。

    用来判断"这次运行还能不能接着跑"。注意不能用 agent_runs.phase 判断：
    被中断的运行那一列往往是 await_approval，并不属于流水线阶段。
    这里用单条 IN 查询，避免按运行逐条查（N+1）。
    """
    ids = [str(r) for r in run_ids if r]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = _db(db).query(
        f"SELECT run_id, phase FROM run_steps WHERE run_id IN ({placeholders})",
        tuple(ids),
    )
    order = ["plan", "execute", "reflect", "synthesize", "review"]
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["run_id"], []).append(row["phase"])
    for phases in out.values():
        phases.sort(key=lambda p: order.index(p) if p in order else 99)
    return out


# ============================================================== 产物
def save_artifact(
    *,
    title: str,
    content: str,
    kind: str = "draft",
    fmt: str = "markdown",
    session_id: int | None = None,
    meta: Mapping[str, Any] | None = None,
    artifact_id: int | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    payload = json.dumps(dict(meta or {}), ensure_ascii=False)
    with database.transaction() as conn:
        if artifact_id:
            conn.execute(
                "UPDATE artifacts SET title = ?, content = ?, kind = ?, fmt = ?, meta = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (title[:300], content, kind, fmt, payload, artifact_id),
            )
            return artifact_id
        cur = conn.execute(
            "INSERT INTO artifacts(session_id, kind, title, content, fmt, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, kind, title[:300], content, fmt, payload),
        )
        return int(cur.lastrowid or 0)


def get_artifact(artifact_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
    if not row:
        return None
    item = dict(row)
    try:
        item["meta"] = json.loads(item.get("meta") or "{}")
    except (TypeError, ValueError):
        item["meta"] = {}
    return item


def list_artifacts(
    *, session_id: int | None = None, limit: int = 50, db: Database | None = None
) -> list[dict[str, Any]]:
    database = _db(db)
    if session_id is None:
        rows = database.query(
            "SELECT id, session_id, kind, title, fmt, created_at, updated_at, "
            "       length(content) AS char_count FROM artifacts "
            "ORDER BY updated_at DESC, id DESC LIMIT ?",
            (limit,),
        )
    else:
        rows = database.query(
            "SELECT id, session_id, kind, title, fmt, created_at, updated_at, "
            "       length(content) AS char_count FROM artifacts "
            "WHERE session_id = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
            (session_id, limit),
        )
    return [dict(r) for r in rows]


def delete_artifact(artifact_id: int, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        return (cur.rowcount or 0) > 0

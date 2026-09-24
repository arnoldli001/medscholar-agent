"""检索：FTS5 BM25 关键词召回、向量 KNN 语义召回、RRF 融合与检索日志。

检索质量是这个项目最常被调整的部分（放宽分级、权重、候选规模、RRF 的 k）。
单独放在一个文件里，调参时不用在一堆课题/会话/产物代码里翻找。
检索日志也放在这里：它记录每次检索的结果，与检索同生命周期。

检索策略（需求 2.4 节）：
    1. FTS5 BM25 关键词召回（标题权重最高，MeSH/关键词次之）
    2. sqlite-vec KNN 语义召回（向量已 L2 归一化，距离与余弦距离单调一致）
    3. RRF 融合（默认 k=60）
    4. 元数据过滤（年份 / 期刊 / 被引 / 开放获取 / 课题）
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any, Mapping, Sequence

from ...config import get_config
from ..connect import Database
from ...models import ScoredPaper, SearchLogEntry
from ...textutil import build_match_query
from ._common import _FTS_COLUMNS, _build_filters, _db, logger
from .embeddings import deserialize_vector, normalize_vector, serialize_vector
from .papers import get_papers_by_ids

__all__ = [
    "search_fts",
    "search_fulltext",
    "search_vector",
    "rrf_fuse",
    "hybrid_search",
    "log_search",
    "recent_searches",
]


def search_fts(
    query: str,
    *,
    limit: int = 100,
    filters: Mapping[str, Any] | None = None,
    db: Database | None = None,
    table: str = "papers_fts",
) -> list[tuple[int, float]]:
    """FTS5 BM25 关键词检索。返回 ``[(paper_id, bm25_score), ...]``，分数越小越相关。

    中文检索采用逐级放宽策略，单一级别无法同时兼顾精度与召回：

    1. ``phrase`` —— 连续子串精确匹配（精度最高，但「治疗卒中后抑郁」匹配不到
       「治疗脑卒中后抑郁」）；
    2. ``bigram`` —— 重叠二元组取 AND（对词序调换、中间插入修饰语更宽容）；
    3. ``or``     —— 二元组取 OR（召回兜底，靠 BM25 排序压住噪声）。

    任何一级返回非空结果就停止，常见查询仍只跑一次 FTS（毫秒级）。
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


def search_log_summary(
    *,
    since: str | None = None,
    queries: Sequence[str] | None = None,
    db: Database | None = None,
) -> dict[str, Any]:
    """按数据源汇总检索日志：命中数、新增数、失败数、涉及多少条检索式。

    用于 PRISMA 流程里的"识别（Identification）"环节：
    那一步要求写出每个数据库各检索到多少条，这个数字本来就在日志里，
    不用让研究者拿 Excel 手工数。

    Args:
        since: ISO 时间下界（``search_logs.created_at`` 是 UTC 的 ``datetime('now')`` 文本）。
        queries: 只统计这些检索式（一次综述通常固定一组检索式）。

    Note:
        ``result_count`` 是该源本次返回的条数，同一篇文献被多个源返回会被重复计入：
        PRISMA 识别阶段统计的就是"检索到的记录数"，去重发生在下一步
        （``duplicates_removed``）。把这两件事混在一起，会让 PRISMA 数字与流程图对不上。
    """
    sql = [
        "SELECT source,",
        "       SUM(result_count) AS result_count,",
        "       SUM(new_count)    AS new_count,",
        "       SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) AS failures,",
        "       COUNT(*)          AS searches",
        "FROM search_logs WHERE 1=1",
    ]
    params: list[Any] = []
    if since:
        sql.append("AND created_at >= ?")
        params.append(since)
    if queries:
        marks = ",".join("?" for _ in queries)
        sql.append(f"AND query IN ({marks})")
        params.extend(queries)
    sql.append("GROUP BY source ORDER BY result_count DESC")

    rows = _db(db).query(" ".join(sql), tuple(params))
    by_source = {str(r["source"] or "unknown"): int(r["result_count"] or 0) for r in rows}
    return {
        "by_source": by_source,
        "total_results": sum(by_source.values()),
        "total_new": sum(int(r["new_count"] or 0) for r in rows),
        "total_searches": sum(int(r["searches"] or 0) for r in rows),
        "failures": sum(int(r["failures"] or 0) for r in rows),
        "queries": sorted({str(r["query"]) for r in recent_searches(limit=200, db=db)})
        if not queries
        else list(queries),
    }

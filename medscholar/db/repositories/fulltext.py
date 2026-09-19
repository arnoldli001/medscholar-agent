"""开放获取全文：正文与 FTS 索引的保存/读取、全文回填候选筛选、抓取失败记录。

单独拆出来是因为全文是"慢、会失败、要能反复跑"的独立流程：它有自己的表
（paper_fulltext / fulltext_attempts）、自己的重试语义（永久失败 vs 可重试），
和文献元数据的生命周期完全不同，混在一起会让"入库"看起来比实际复杂。
"""

from __future__ import annotations

from ..connect import Database
from ...models import Paper
from ._common import _db, _fts_sync
from .papers import get_paper

__all__ = [
    "save_fulltext",
    "get_fulltext",
    "fulltext_candidates",
    "record_fulltext_attempt",
    "clear_fulltext_attempts",
    "fulltext_failure_stats",
]


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

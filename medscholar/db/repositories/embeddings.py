"""嵌入向量：float32 序列化、L2 归一化，以及 paper_embeddings 表的读写。

"向量长什么样"是一个独立的契约：检索侧（KNN 与纯 Python 回退）
和嵌入管道都要按同一份序列化/归一化规则来，混在检索代码里就会出现两处真相，
所以单独拆出来。
"""

from __future__ import annotations

import math
import sqlite3
import struct
from typing import Any, Sequence

from ..connect import Database
from ._common import _db

__all__ = [
    "serialize_vector",
    "deserialize_vector",
    "normalize_vector",
    "store_embedding",
    "store_embeddings",
    "papers_missing_embeddings",
    "clear_embeddings",
]


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

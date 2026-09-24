"""运行记录、阶段快照与产物：agent_runs / run_steps / artifacts。

这三张表服务的是"一次研究运行能不能续跑、草稿在哪"，
不参与文献、检索与向量逻辑，只被 Agent 运行时与草稿页读写；
生命周期也一致（运行被删时它的阶段快照一并消失），所以单独拆出来。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..connect import Database
from ._common import _db

__all__ = [
    "upsert_run",
    "get_run",
    "list_runs",
    "mark_interrupted_runs",
    "delete_run",
    "save_run_step",
    "get_run_step",
    "list_run_steps",
    "latest_run_for_session",
    "run_step_phases",
    "save_artifact",
    "get_artifact",
    "list_artifacts",
    "delete_artifact",
]


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

    服务启动时调用。用户刷新页面能看到"那次运行被中断了"，
    而不是对着一个永远等不到内容的草稿页签。
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

    用来判断"这次运行还能不能接着跑"。不能用 agent_runs.phase 判断：
    被中断的运行那一列往往是 await_approval，不属于流水线阶段。
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
